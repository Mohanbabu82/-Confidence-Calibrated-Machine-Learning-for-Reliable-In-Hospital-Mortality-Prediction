"""Final, single-use held-out test-set evaluation.

================================================================================
TEST-SET USAGE POLICY — READ BEFORE USE
================================================================================
This module is the ONLY authorized place in this project where test-set
LABELS are used for evaluation. Every other script/notebook (src/preprocess.py,
src/train_models.py, notebooks 01-04) loads the test split via
src.preprocess.load_and_split_cohort() and immediately discards it
(`del test_df`) specifically so that no hyperparameter, calibration-method,
referral-rate, or decision-threshold choice is ever informed by test-set
performance.

By the time this module runs, every modeling decision is already frozen:
  - model hyperparameters                 (src/train_models.py, validation-selected)
  - calibration method per model          (src/selective_prediction.py, validation-ECE-selected)
  - referral-rate grid + decision threshold (fixed in configs/config.yaml)

This module only COMPUTES metrics against those already-frozen choices — it
never searches over alternatives using test performance. `run_final_evaluation()`
writes a sentinel file the first time it runs and logs a loud (non-blocking)
warning on subsequent runs, so repeated test-set evaluation is visible in the
logs rather than silently normalized. Re-running to fix an implementation bug
is fine; re-running to pick a better-looking result is a protocol violation.
================================================================================

Every number in every output table is computed directly from saved test
predictions produced by the models/calibrators already fit in notebooks
01-04 — no value here is simulated, guessed, hardcoded, or otherwise
fabricated. Where a metric cannot be computed from the available data (a
single-class sample, or a subgroup too small for a stable estimate), the
result is NaN with an explanatory note, never a placeholder number.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.calibration import brier_score, calibration_intercept_slope, expected_calibration_error
from src.config import load_config
from src.models import MLPClassifierTorch
from src.preprocess import IDENTIFIER_COLUMNS, drop_identifier_columns, load_and_split_cohort
from src.reproducibility import DEFAULT_SEED, get_logger, log_run_metadata, set_global_seed
from src.selective_prediction import compute_accepted_mask, select_best_calibration_method

METRIC_NAMES = [
    "auroc", "auprc", "sensitivity", "specificity", "precision", "recall",
    "f1_score", "balanced_accuracy", "brier_score", "ece",
    "calibration_intercept", "calibration_slope",
]


# ---------------------------------------------------------------------------
# Point-estimate metrics
# ---------------------------------------------------------------------------

def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def compute_point_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, ece_bins: int = 10
) -> dict[str, float]:
    """Point estimates for the full final-evaluation metric suite.

    Returns NaN (never a fabricated fallback) for any metric that is
    undefined given the data — e.g. AUROC when only one class is present.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, float] = {}
    single_class = len(np.unique(y_true)) < 2

    if single_class:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    else:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))

    tp, tn, fp, fn = _confusion_counts(y_true, y_pred)
    sensitivity = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    precision = _safe_divide(tp, tp + fp)
    recall = sensitivity  # recall === sensitivity (positive-class recall); reported under both names per project brief

    metrics["sensitivity"] = sensitivity
    metrics["specificity"] = specificity
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1_score"] = (
        float("nan")
        if (np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0)
        else 2 * precision * recall / (precision + recall)
    )
    metrics["balanced_accuracy"] = (
        float("nan") if (np.isnan(sensitivity) or np.isnan(specificity)) else (sensitivity + specificity) / 2
    )

    metrics["brier_score"] = brier_score(y_true, y_prob)
    metrics["ece"] = expected_calibration_error(y_true, y_prob, n_bins=ece_bins)

    if single_class:
        metrics["calibration_intercept"] = float("nan")
        metrics["calibration_slope"] = float("nan")
    else:
        with warnings.catch_warnings():
            # Calibration-slope instability (from a near-unregularized fit
            # on a small/low-variance resample) is already surfaced once
            # during actual calibration fitting in notebook 03; suppress
            # the repeat here to avoid thousands of duplicate warnings
            # across bootstrap resamples.
            warnings.simplefilter("ignore")
            intercept, slope = calibration_intercept_slope(y_true, y_prob)
        metrics["calibration_intercept"] = intercept
        metrics["calibration_slope"] = slope

    return metrics


# ---------------------------------------------------------------------------
# Stratified bootstrap
# ---------------------------------------------------------------------------

def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One stratified bootstrap resample: resample WITH replacement
    separately within each class, preserving the exact original class
    counts (not just proportions) in every resample.
    """
    y_true = np.asarray(y_true)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    resampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True) if len(pos_idx) else np.array([], dtype=int)
    resampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True) if len(neg_idx) else np.array([], dtype=int)
    return np.concatenate([resampled_pos, resampled_neg])


def bootstrap_metrics_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    ece_bins: int = 10,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Point estimate + (1-alpha) stratified-bootstrap CI for every metric
    in METRIC_NAMES. Computed from ONE bootstrap loop (not one loop per
    metric) for efficiency: 1000 resamples x all metrics per resample.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    point = compute_point_metrics(y_true, y_prob, threshold=threshold, ece_bins=ece_bins)

    result = dict(point)

    if len(np.unique(y_true)) < 2 or len(y_true) < 2:
        # Cannot bootstrap a single-class or near-empty sample; NaN CIs, not fabricated ones.
        for name in METRIC_NAMES:
            result[f"{name}_ci_lower"] = float("nan")
            result[f"{name}_ci_upper"] = float("nan")
        return result

    rng = np.random.default_rng(seed)
    boot_values = {name: np.full(n_bootstrap, np.nan) for name in METRIC_NAMES}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for b in range(n_bootstrap):
            idx = stratified_bootstrap_indices(y_true, rng)
            resample_metrics = compute_point_metrics(y_true[idx], y_prob[idx], threshold=threshold, ece_bins=ece_bins)
            for name in METRIC_NAMES:
                boot_values[name][b] = resample_metrics[name]

    lower_q = 100 * (alpha / 2)
    upper_q = 100 * (1 - alpha / 2)
    for name in METRIC_NAMES:
        values = boot_values[name]
        values = values[~np.isnan(values)]
        if len(values) == 0:
            result[f"{name}_ci_lower"] = float("nan")
            result[f"{name}_ci_upper"] = float("nan")
        else:
            result[f"{name}_ci_lower"] = float(np.percentile(values, lower_q))
            result[f"{name}_ci_upper"] = float(np.percentile(values, upper_q))

    return result


# ---------------------------------------------------------------------------
# Referral-coverage evaluation (reuses src.selective_prediction's referral logic)
# ---------------------------------------------------------------------------

def evaluate_at_referral_rate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    referral_rate: float,
    threshold: float = 0.5,
    ece_bins: int = 10,
    n_bootstrap: int = 1000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Evaluate the full metric suite (+ bootstrap CI) on the accepted
    (non-referred) subset at one referral rate. The referral decision
    itself is NOT re-derived per bootstrap resample — it is fixed once
    from the observed predictions (as in src.selective_prediction), and
    the bootstrap resamples within that fixed accepted subset only, to
    estimate uncertainty in the metrics reported for that subset.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    accepted_mask = compute_accepted_mask(y_prob, referral_rate)
    n_accepted = int(accepted_mask.sum())

    result: dict[str, Any] = {
        "referral_rate": referral_rate,
        "coverage": n_accepted / n,
        "n_accepted": n_accepted,
        "n_referred": n - n_accepted,
    }

    if n_accepted == 0:
        for name in METRIC_NAMES:
            result[name] = float("nan")
            result[f"{name}_ci_lower"] = float("nan")
            result[f"{name}_ci_upper"] = float("nan")
        return result

    metrics = bootstrap_metrics_with_ci(
        y_true[accepted_mask], y_prob[accepted_mask],
        threshold=threshold, ece_bins=ece_bins, n_bootstrap=n_bootstrap, seed=seed,
    )
    result.update(metrics)
    return result


# ---------------------------------------------------------------------------
# Subgroup evaluation
# ---------------------------------------------------------------------------

def derive_age_group(age_series: pd.Series, cutoffs: list[float] = (65,)) -> pd.Series:
    """Bin age into groups using fixed cutoffs (default: <65 vs >=65, the
    standard elderly-ICU-mortality-risk threshold). Adjust `cutoffs` per
    your study's inclusion criteria if a different threshold applies.
    """
    cutoffs = sorted(cutoffs)
    bounds = [-np.inf] + list(cutoffs) + [np.inf]
    labels = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if lo == -np.inf:
            labels.append(f"<{hi:g}")
        elif hi == np.inf:
            labels.append(f">={lo:g}")
        else:
            labels.append(f"{lo:g}-{hi:g}")
    return pd.cut(age_series, bins=bounds, labels=labels, right=False)


def evaluate_subgroups(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subgroup_df: pd.DataFrame,
    subgroup_columns: list[str],
    threshold: float = 0.5,
    ece_bins: int = 10,
    n_bootstrap: int = 1000,
    seed: int = DEFAULT_SEED,
    min_size: int = 20,
    min_events: int = 5,
) -> pd.DataFrame:
    """Evaluate the metric suite within each level of each requested
    subgroup column (post-hoc fairness/equity auditing — see
    docs/data_dictionary.md's fairness note on race/insurance). Subgroup
    levels with fewer than `min_size` cases, fewer than `min_events`
    positive cases, or only one class present are SKIPPED (NaN + a `note`
    explaining why) — never approximated or fabricated.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rows = []

    for col in subgroup_columns:
        if col not in subgroup_df.columns:
            rows.append({
                "subgroup_variable": col, "subgroup_level": "N/A", "n": 0, "n_events": 0,
                "note": f"column '{col}' not present in test data — skipped",
                **{name: float("nan") for name in METRIC_NAMES},
                **{f"{name}_ci_lower": float("nan") for name in METRIC_NAMES},
                **{f"{name}_ci_upper": float("nan") for name in METRIC_NAMES},
            })
            continue

        for level, _ in subgroup_df.groupby(col, observed=True):
            mask = (subgroup_df[col] == level).to_numpy()
            n = int(mask.sum())
            n_events = int(y_true[mask].sum()) if n > 0 else 0

            row: dict[str, Any] = {
                "subgroup_variable": col, "subgroup_level": str(level), "n": n, "n_events": n_events,
            }

            if n < min_size or n_events < min_events or n_events == n:
                row["note"] = f"skipped: n={n} (min {min_size}), n_events={n_events} (min {min_events}, and must be < n)"
                for name in METRIC_NAMES:
                    row[name] = float("nan")
                    row[f"{name}_ci_lower"] = float("nan")
                    row[f"{name}_ci_upper"] = float("nan")
            else:
                row["note"] = ""
                metrics = bootstrap_metrics_with_ci(
                    y_true[mask], y_prob[mask],
                    threshold=threshold, ece_bins=ece_bins, n_bootstrap=n_bootstrap, seed=seed,
                )
                row.update(metrics)

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model / calibrator loading (mirrors notebooks 03/04's loading logic)
# ---------------------------------------------------------------------------

def load_trained_models(config: dict[str, Any]) -> dict[str, Any]:
    models_dir = Path(config["training"]["models_output_dir"])
    seed = config["project"]["random_seed"]

    logreg_model = joblib.load(models_dir / "logistic_regression.joblib")
    lgbm_model = joblib.load(models_dir / "lightgbm.joblib")

    mlp_checkpoint = torch.load(models_dir / "mlp_torch.pt", weights_only=True)
    mlp_model = MLPClassifierTorch(
        input_dim=mlp_checkpoint["input_dim"], hidden_dims=mlp_checkpoint["hidden_dims"], seed=seed
    )
    mlp_model.load_state_dict(mlp_checkpoint["state_dict"])
    mlp_model.eval()

    return {"logistic_regression": logreg_model, "lightgbm": lgbm_model, "mlp_torch": mlp_model}


def compute_raw_test_predictions(
    models: dict[str, Any], X_test: pd.DataFrame
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Uncalibrated test-set predictions for all three models, plus the
    MLP's raw pre-sigmoid logits (needed for temperature scaling).
    """
    raw_prob = {
        "logistic_regression": models["logistic_regression"].predict_proba(X_test)[:, 1],
        "lightgbm": models["lightgbm"].predict_proba(X_test)[:, 1],
    }
    with torch.no_grad():
        mlp_logits = models["mlp_torch"](torch.tensor(X_test.to_numpy(dtype=np.float32))).squeeze().numpy()
    raw_prob["mlp_torch"] = 1.0 / (1.0 + np.exp(-mlp_logits))
    return raw_prob, mlp_logits


def apply_calibration_method(
    config: dict[str, Any], model_name: str, method: str, raw_prob: np.ndarray, mlp_logits: np.ndarray | None
) -> np.ndarray:
    """Apply an already-fitted (train/val-only) calibrator to test-set
    predictions. Never refits a calibrator here.
    """
    if method == "uncalibrated":
        return raw_prob

    cal_dir = Path(config["final_evaluation"]["calibration_models_output_dir"])

    if method == "temperature":
        from src.calibration import TemperatureScaler

        checkpoint = torch.load(cal_dir / f"calibration_{model_name}_temperature.pt", weights_only=True)
        scaler = TemperatureScaler()
        scaler.load_state_dict(checkpoint["state_dict"])
        return scaler.transform(mlp_logits)

    scaler = joblib.load(cal_dir / f"calibration_{model_name}_{method}.joblib")
    return scaler.transform(raw_prob)


def available_calibration_methods(model_name: str) -> list[str]:
    """All calibration methods evaluated per model (uncalibrated + fitted
    calibrators); temperature scaling applies to the MLP only.
    """
    methods = ["uncalibrated", "platt", "isotonic"]
    if model_name == "mlp_torch":
        methods.append("temperature")
    return methods


# ---------------------------------------------------------------------------
# Raw per-case test predictions (byproduct of THIS SAME authorized run —
# not a second, independent touch of test labels). Persisted so downstream
# consumers (e.g. notebooks/06_publication_outputs.ipynb) can build
# ROC/PR/reliability/subgroup-calibration figures from the exact same
# already-computed, already-authorized test predictions, rather than
# recomputing test-set predictions a second time.
# ---------------------------------------------------------------------------

def save_test_predictions(
    config: dict[str, Any],
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    raw_prob: dict[str, np.ndarray],
    calibrated_prob_per_model: dict[str, np.ndarray],
    best_method_per_model: dict[str, str],
    subgroup_source_df: pd.DataFrame,
    logger,
) -> list[Path]:
    fe_config = config["final_evaluation"]
    id_col = config["preprocessing"]["id_column"]
    subgroup_cols = [c for c in fe_config["subgroup_columns"] if c in subgroup_source_df.columns]

    out_dir = Path(fe_config["tables_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []

    for model_name in ["logistic_regression", "lightgbm", "mlp_torch"]:
        pred_df = pd.DataFrame({
            id_col: test_df[id_col].reset_index(drop=True),
            "y_true": y_test,
            "y_prob_uncalibrated": raw_prob[model_name],
            "y_prob_calibrated": calibrated_prob_per_model[model_name],
            "calibration_method": best_method_per_model[model_name],
        })
        for col in subgroup_cols:
            pred_df[col] = subgroup_source_df[col].reset_index(drop=True)

        out_path = out_dir / f"test_predictions_{model_name}.csv"
        pred_df.to_csv(out_path, index=False)
        written_paths.append(out_path)
        logger.info("Raw test predictions for %s saved to %s", model_name, out_path)

    return written_paths


# ---------------------------------------------------------------------------
# Test-set usage marker (soft safeguard, not a hard block)
# ---------------------------------------------------------------------------

def _check_and_record_test_usage(marker_path: str | Path, logger) -> None:
    marker_path = Path(marker_path)
    if marker_path.exists():
        prior = json.loads(marker_path.read_text(encoding="utf-8"))
        logger.warning(
            "TEST-SET USAGE MARKER ALREADY EXISTS (first evaluated at %s, %d prior run(s)). "
            "Re-running this script to fix an implementation bug is fine. Re-running it to "
            "search over model/calibration/threshold choices based on test results is a "
            "held-out-set protocol violation — do not do that.",
            prior.get("first_evaluated_utc", "unknown"), prior.get("run_count", 1),
        )
        prior["run_count"] = prior.get("run_count", 1) + 1
        prior["last_evaluated_utc"] = datetime.now(timezone.utc).isoformat()
        marker_path.write_text(json.dumps(prior, indent=2), encoding="utf-8")
    else:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(
            json.dumps({
                "first_evaluated_utc": datetime.now(timezone.utc).isoformat(),
                "last_evaluated_utc": datetime.now(timezone.utc).isoformat(),
                "run_count": 1,
            }, indent=2),
            encoding="utf-8",
        )
        logger.info("Test-set usage marker created at %s (first final evaluation).", marker_path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_final_evaluation(config: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    config = config or load_config()
    seed = config["project"]["random_seed"]
    set_global_seed(seed)
    logger = get_logger(log_file=config["logging"]["log_file"])
    fe_config = config["final_evaluation"]

    _check_and_record_test_usage(fe_config["test_usage_marker_path"], logger)

    logger.info("=== evaluate_final.py started (seed=%d, n_bootstrap=%d) — TEST SET IN USE ===",
                seed, fe_config["n_bootstrap"])

    target_col = config["preprocessing"]["target_column"]
    threshold = fe_config["decision_threshold"]
    ece_bins = fe_config["ece_bins"]
    n_bootstrap = fe_config["n_bootstrap"]

    train_df, val_df, test_df = load_and_split_cohort(config)
    del train_df, val_df  # this stage evaluates on TEST only
    logger.info("Test set loaded: n=%d", len(test_df))

    preprocessor = joblib.load(config["preprocessing"]["preprocessor_output_path"])
    test_features_df = drop_identifier_columns(test_df, IDENTIFIER_COLUMNS)
    X_test = preprocessor.transform(test_features_df)
    y_test = test_features_df[target_col].reset_index(drop=True).to_numpy()

    models = load_trained_models(config)
    raw_prob, mlp_logits = compute_raw_test_predictions(models, X_test)

    calibration_metrics_df = pd.read_csv(config["calibration"]["metrics_table_path"])
    sp_config = config["selective_prediction"]

    best_method_per_model = {}
    calibrated_prob_per_model = {}
    for model_name in ["logistic_regression", "lightgbm", "mlp_torch"]:
        candidates = sp_config["calibration_candidates"][model_name]
        best_method = select_best_calibration_method(calibration_metrics_df, model_name, candidates)
        best_method_per_model[model_name] = best_method
        calibrated_prob_per_model[model_name] = apply_calibration_method(
            config, model_name, best_method, raw_prob[model_name], mlp_logits
        )
        logger.info("[%s] best calibration method (validation-selected): %s", model_name, best_method)

    # --- Table 3: main results (model x best calibration method, full coverage) ---
    main_rows = []
    for model_name, y_prob in calibrated_prob_per_model.items():
        logger.info("[table_3] evaluating %s (%s) on full test set...", model_name, best_method_per_model[model_name])
        metrics = bootstrap_metrics_with_ci(
            y_test, y_prob, threshold=threshold, ece_bins=ece_bins, n_bootstrap=n_bootstrap, seed=seed
        )
        main_rows.append({
            "model": model_name, "calibration_method": best_method_per_model[model_name],
            "n_test": len(y_test), **metrics,
        })
    table3_main_results = pd.DataFrame(main_rows)

    # --- Table 4: clinical thresholds (model x best calibration method x referral rate) ---
    threshold_rows = []
    for model_name, y_prob in calibrated_prob_per_model.items():
        for referral_rate in sp_config["referral_rates"]:
            logger.info("[table_4] evaluating %s at referral_rate=%.2f...", model_name, referral_rate)
            result = evaluate_at_referral_rate(
                y_test, y_prob, referral_rate, threshold=threshold, ece_bins=ece_bins,
                n_bootstrap=n_bootstrap, seed=seed,
            )
            threshold_rows.append({
                "model": model_name, "calibration_method": best_method_per_model[model_name], **result,
            })
    table4_clinical_thresholds = pd.DataFrame(threshold_rows)

    # --- Table 5: ablation study (model x ALL calibration methods, full coverage) ---
    ablation_rows = []
    for model_name in ["logistic_regression", "lightgbm", "mlp_torch"]:
        for method in available_calibration_methods(model_name):
            logger.info("[table_5] evaluating %s (%s, ablation) on full test set...", model_name, method)
            y_prob = apply_calibration_method(config, model_name, method, raw_prob[model_name], mlp_logits)
            metrics = bootstrap_metrics_with_ci(
                y_test, y_prob, threshold=threshold, ece_bins=ece_bins, n_bootstrap=n_bootstrap, seed=seed
            )
            ablation_rows.append({
                "model": model_name, "calibration_method": method,
                "is_selected_method": method == best_method_per_model[model_name],
                "n_test": len(y_test), **metrics,
            })
    table5_ablation_study = pd.DataFrame(ablation_rows)

    # --- Table 6: subgroup results (model x best calibration method x subgroup levels) ---
    subgroup_source_df = test_features_df.reset_index(drop=True).copy()
    if "age_at_admission" in subgroup_source_df.columns:
        subgroup_source_df["age_group"] = derive_age_group(
            subgroup_source_df["age_at_admission"], cutoffs=fe_config["age_group_cutoffs"]
        )

    subgroup_frames = []
    for model_name, y_prob in calibrated_prob_per_model.items():
        logger.info("[table_6] evaluating %s (%s) across subgroups...", model_name, best_method_per_model[model_name])
        sub_df = evaluate_subgroups(
            y_test, y_prob, subgroup_source_df, fe_config["subgroup_columns"],
            threshold=threshold, ece_bins=ece_bins, n_bootstrap=n_bootstrap, seed=seed,
            min_size=fe_config["min_subgroup_size"], min_events=fe_config["min_subgroup_events"],
        )
        sub_df.insert(0, "model", model_name)
        sub_df.insert(1, "calibration_method", best_method_per_model[model_name])
        subgroup_frames.append(sub_df)
    table6_subgroup_results = pd.concat(subgroup_frames, ignore_index=True)

    # --- Raw per-case test predictions (for downstream figure generation) ---
    save_test_predictions(
        config, test_df, y_test, raw_prob, calibrated_prob_per_model,
        best_method_per_model, subgroup_source_df, logger,
    )

    # --- Save ---
    tables = {
        fe_config["main_results_table_path"]: table3_main_results,
        fe_config["clinical_thresholds_table_path"]: table4_clinical_thresholds,
        fe_config["ablation_study_table_path"]: table5_ablation_study,
        fe_config["subgroup_results_table_path"]: table6_subgroup_results,
    }
    for path_str, df in tables.items():
        out_path = Path(path_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("Saved %s (%d rows)", out_path, len(df))

    log_run_metadata(
        output_path=fe_config["run_metadata_path"],
        seed=seed,
        extra={
            "step": "final_evaluation",
            "n_test": len(y_test),
            "n_bootstrap": n_bootstrap,
            "ci_alpha": fe_config["ci_alpha"],
            "decision_threshold": threshold,
            "best_calibration_method_per_model": best_method_per_model,
            "referral_rates": sp_config["referral_rates"],
            "test_set_used": True,
            "test_set_usage_note": "single authorized final evaluation stage",
        },
    )
    logger.info("=== evaluate_final.py finished ===")

    return {
        "table_3_main_results": table3_main_results,
        "table_4_clinical_thresholds": table4_clinical_thresholds,
        "table_5_ablation_study": table5_ablation_study,
        "table_6_subgroup_results": table6_subgroup_results,
    }


if __name__ == "__main__":
    run_final_evaluation()
