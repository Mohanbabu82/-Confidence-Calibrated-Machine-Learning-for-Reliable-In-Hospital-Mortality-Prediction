"""Final held-out test-set evaluation for the paper pipeline
(notebooks/04_final_evaluation.ipynb).

TEST-SET POLICY: this module is the ONLY place in the paper pipeline
(notebooks 01-05) where test-set LABELS are used. Model hyperparameters
(02) and calibration method (03) are already frozen using train/validation
data before this module runs — it only computes metrics against those
frozen choices; it never searches over alternatives using test
performance. `run_final_evaluation()` writes a sentinel marker file on
first run and logs a (non-blocking) warning on any later run.

Every value in every output table is computed directly from the real test
predictions produced by the models/calibrators already fit in notebooks
02-03 — nothing here is simulated, guessed, or fabricated.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.calibration import brier_score, expected_calibration_error

METRIC_NAMES = ["auroc", "auprc", "brier_score", "ece"]  # the three that get bootstrap CIs, plus auprc for completeness
CI_METRIC_NAMES = ["auroc", "brier_score", "ece"]  # per task spec: CIs only for these three


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


def compute_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    """Sensitivity, specificity, PPV, NPV, F1, balanced accuracy at one
    decision threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tp, tn, fp, fn = _confusion_counts(y_true, y_pred)

    sensitivity = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    ppv = _safe_divide(tp, tp + fp)
    npv = _safe_divide(tn, tn + fn)
    f1 = (
        float("nan") if (np.isnan(ppv) or np.isnan(sensitivity) or (ppv + sensitivity) == 0)
        else 2 * ppv * sensitivity / (ppv + sensitivity)
    )
    balanced_accuracy = (
        float("nan") if (np.isnan(sensitivity) or np.isnan(specificity)) else (sensitivity + specificity) / 2
    )

    return {
        "threshold": threshold, "sensitivity": sensitivity, "specificity": specificity,
        "ppv": ppv, "npv": npv, "f1_score": f1, "balanced_accuracy": balanced_accuracy,
    }


def compute_discrimination_calibration_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, ece_bins: int = 10
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier_score": float("nan"), "ece": float("nan")}
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier_score": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=ece_bins),
    }


def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample WITH replacement separately within each class, preserving
    the exact original class counts in every resample.
    """
    y_true = np.asarray(y_true)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    resampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True) if len(pos_idx) else np.array([], dtype=int)
    resampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True) if len(neg_idx) else np.array([], dtype=int)
    return np.concatenate([resampled_pos, resampled_neg])


def bootstrap_ci(
    y_true: np.ndarray, y_prob: np.ndarray, ece_bins: int = 10,
    n_bootstrap: int = 1000, alpha: float = 0.05, seed: int = 42,
) -> dict[str, float]:
    """95% (or 1-alpha) stratified-bootstrap CI for AUROC, Brier score, and
    ECE only (per task spec — not every metric).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    point = compute_discrimination_calibration_metrics(y_true, y_prob, ece_bins=ece_bins)

    result: dict[str, float] = {}
    for name in CI_METRIC_NAMES:
        result[name] = point[name]

    if len(np.unique(y_true)) < 2 or len(y_true) < 2:
        for name in CI_METRIC_NAMES:
            result[f"{name}_ci_lower"] = float("nan")
            result[f"{name}_ci_upper"] = float("nan")
        result["auprc"] = point["auprc"]
        return result

    rng = np.random.default_rng(seed)
    boot_values = {name: np.full(n_bootstrap, np.nan) for name in CI_METRIC_NAMES}
    for b in range(n_bootstrap):
        idx = stratified_bootstrap_indices(y_true, rng)
        resample_metrics = compute_discrimination_calibration_metrics(y_true[idx], y_prob[idx], ece_bins=ece_bins)
        for name in CI_METRIC_NAMES:
            boot_values[name][b] = resample_metrics[name]

    lower_q, upper_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    for name in CI_METRIC_NAMES:
        values = boot_values[name]
        values = values[~np.isnan(values)]
        if len(values) == 0:
            result[f"{name}_ci_lower"] = float("nan")
            result[f"{name}_ci_upper"] = float("nan")
        else:
            result[f"{name}_ci_lower"] = float(np.percentile(values, lower_q))
            result[f"{name}_ci_upper"] = float(np.percentile(values, upper_q))

    result["auprc"] = point["auprc"]  # reported, but no CI requested for it
    return result


def load_calibrators(models_dir: Path, model_names: list[str]) -> dict[str, dict[str, Any]]:
    """Load already-fit (validation-only) Platt/isotonic calibrators for
    each model. Never refits.
    """
    calibrators: dict[str, dict[str, Any]] = {}
    for model_name in model_names:
        calibrators[model_name] = {}
        for method in ["platt", "isotonic"]:
            path = models_dir / f"calibration_{model_name}_{method}.joblib"
            if path.exists():
                calibrators[model_name][method] = joblib.load(path)
    return calibrators


def apply_all_methods(
    model_name: str, y_prob_uncal: np.ndarray, calibrators: dict[str, dict[str, Any]]
) -> dict[str, np.ndarray]:
    methods = {"uncalibrated": y_prob_uncal}
    for method, calibrator in calibrators.get(model_name, {}).items():
        methods[method] = calibrator.transform(y_prob_uncal)
    return methods


def _check_and_record_test_usage(marker_path: str | Path, logger) -> None:
    marker_path = Path(marker_path)
    if marker_path.exists():
        prior = json.loads(marker_path.read_text(encoding="utf-8"))
        logger.warning(
            "TEST-SET USAGE MARKER ALREADY EXISTS (first evaluated at %s, %d prior run(s)). "
            "Re-running to fix a bug is fine; re-running to search over choices based on "
            "test results is a held-out-set protocol violation.",
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
