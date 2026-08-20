"""Baseline model training for first-24h ICU in-hospital mortality.

Trains three baselines:
  1. Regularized (L2) logistic regression
  2. LightGBM gradient-boosted trees
  3. A small PyTorch MLP

Rules enforced throughout this module (do not relax without explicit
sign-off):
  - TRAIN split is used for model fitting only.
  - VALIDATION split is used for hyperparameter selection (grid search /
    learning-rate search) and early stopping only.
  - The TEST split is loaded (via load_and_split_cohort, for reproducible
    accounting) but is NEVER passed to any model's .fit(), never scored,
    never used to pick a hyperparameter, calibrate a probability, or
    choose a decision threshold. It is deliberately discarded immediately
    after loading (see `del test_df` below) so it cannot be touched later
    in this script by accident.

Every run saves: the three fitted models, validation prediction
probabilities per model, per-model training logs, exact hyperparameters
(outputs/tables/table_2_hyperparameters.csv), software versions, runtime,
and the global random seed (all folded into
logs/train_models_run_metadata.json via src.reproducibility).
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.config import load_config
from src.models import MLPClassifierTorch, build_lightgbm, build_logistic_regression
from src.preprocess import (
    IDENTIFIER_COLUMNS,
    ClinicalPreprocessor,
    drop_identifier_columns,
    identify_feature_types,
    load_and_split_cohort,
)
from src.reproducibility import get_logger, log_run_metadata, set_global_seed

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_training_data(
    config: dict[str, Any], logger
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, ClinicalPreprocessor]:
    """Load the cohort, reproduce the patient-level 70/10/20 split, and fit
    preprocessing on the TRAIN split only. Returns train/val feature
    matrices + labels, the raw train/val frames (for id tracking in output
    tables), and the fitted preprocessor. The test split is loaded (for
    reproducible cohort accounting) and then explicitly discarded.
    """
    train_df, val_df, test_df = load_and_split_cohort(config)
    logger.info(
        "Reproduced patient-level split: train=%d, val=%d, test=%d (test set not used below)",
        len(train_df), len(val_df), len(test_df),
    )

    # Deliberately not used anywhere in this script. Deleting rather than
    # merely ignoring it makes "test set untouched during training" a
    # structural property of this script, not just a convention.
    del test_df

    target_col = config["preprocessing"]["target_column"]
    train_features_df = drop_identifier_columns(train_df, IDENTIFIER_COLUMNS)
    val_features_df = drop_identifier_columns(val_df, IDENTIFIER_COLUMNS)

    model_exclude_cols = [target_col, "icu_intime"]
    numeric_cols, categorical_cols = identify_feature_types(train_features_df, exclude_cols=model_exclude_cols)
    logger.info("Numeric features (%d): %s", len(numeric_cols), numeric_cols)
    logger.info("Categorical features (%d): %s", len(categorical_cols), categorical_cols)

    # Fit exclusively on train; this deterministically reproduces the
    # artifact notebooks/01_cohort_audit.ipynb saves, so re-running this
    # script does not require that notebook to have been run first.
    preprocessor = ClinicalPreprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    preprocessor.fit(train_features_df)

    X_train = preprocessor.transform(train_features_df)
    X_val = preprocessor.transform(val_features_df)
    y_train = train_features_df[target_col].reset_index(drop=True)
    y_val = val_features_df[target_col].reset_index(drop=True)

    preprocessor_path = config["preprocessing"]["preprocessor_output_path"]
    Path(preprocessor_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    logger.info("Fitted preprocessor saved to %s", preprocessor_path)

    return X_train, y_train, X_val, y_val, train_df, val_df, preprocessor


def _save_val_predictions(
    val_df: pd.DataFrame, y_val: pd.Series, y_val_prob: np.ndarray, model_name: str, config: dict[str, Any]
) -> Path:
    id_col = config["preprocessing"]["id_column"]
    out_df = pd.DataFrame({
        id_col: val_df[id_col].reset_index(drop=True),
        "y_true": y_val.reset_index(drop=True),
        "y_prob": y_val_prob,
    })
    out_path = Path(config["training"]["tables_output_dir"]) / f"val_predictions_{model_name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# 1. Logistic regression (L2), C selected on validation AUROC
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
    config: dict[str, Any], logger,
) -> dict[str, Any]:
    start = time.perf_counter()
    C_grid = config["models"]["logistic_regression"]["C_grid"]

    search_rows = []
    best_model = None
    best_auroc = -np.inf
    best_C = None

    for C in C_grid:
        model = build_logistic_regression(config, C=C)
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        val_auroc = roc_auc_score(y_val, val_prob)
        search_rows.append({"C": C, "val_auroc": val_auroc})
        logger.info("[logistic_regression] C=%s -> val AUROC=%.4f", C, val_auroc)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_C = C
            best_model = model

    runtime_seconds = time.perf_counter() - start

    search_df = pd.DataFrame(search_rows)
    search_df["selected"] = search_df["C"] == best_C
    log_path = Path(config["training"]["tables_output_dir"]) / "training_log_logistic_regression.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    search_df.to_csv(log_path, index=False)

    val_prob = best_model.predict_proba(X_val)[:, 1]
    logger.info(
        "[logistic_regression] selected C=%s (val AUROC=%.4f) in %.2fs", best_C, best_auroc, runtime_seconds
    )

    return {
        "model_name": "logistic_regression",
        "model": best_model,
        "val_prob": val_prob,
        "runtime_seconds": runtime_seconds,
        "hyperparameters": {
            "penalty": config["models"]["logistic_regression"]["penalty"],
            "solver": config["models"]["logistic_regression"]["solver"],
            "C": best_C,
            "max_iter": config["models"]["logistic_regression"]["max_iter"],
            "random_state": config["models"]["logistic_regression"]["random_state"],
        },
        "val_auroc": best_auroc,
        "training_log_path": str(log_path),
    }


# ---------------------------------------------------------------------------
# 2. LightGBM, learning_rate selected on validation AUROC, early stopping
# ---------------------------------------------------------------------------

def train_lightgbm(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
    config: dict[str, Any], logger,
) -> dict[str, Any]:
    if lgb is None:
        raise ImportError("lightgbm is not installed. `pip install -r requirements.txt` first.")

    start = time.perf_counter()
    lgbm_config = config["models"]["lightgbm"]
    learning_rate_grid = lgbm_config["learning_rate_grid"]
    early_stopping_rounds = lgbm_config["early_stopping_rounds"]

    search_rows = []
    best_model = None
    best_auroc = -np.inf
    best_learning_rate = None
    best_evals_result = None

    for lr in learning_rate_grid:
        model = build_lightgbm(config, learning_rate=lr)
        evals_result: dict[str, Any] = {}
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.record_evaluation(evals_result),
                lgb.log_evaluation(period=0),
            ],
        )
        val_auroc = float(model.best_score_["valid_0"]["auc"])
        best_iteration = int(model.best_iteration_)
        search_rows.append({
            "learning_rate": lr,
            "best_iteration": best_iteration,
            "val_auroc": val_auroc,
        })
        logger.info(
            "[lightgbm] learning_rate=%s -> best_iteration=%d, val AUROC=%.4f",
            lr, best_iteration, val_auroc,
        )

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_learning_rate = lr
            best_model = model
            best_evals_result = evals_result

    runtime_seconds = time.perf_counter() - start

    search_df = pd.DataFrame(search_rows)
    search_df["selected"] = search_df["learning_rate"] == best_learning_rate
    search_log_path = Path(config["training"]["tables_output_dir"]) / "training_log_lightgbm.csv"
    search_log_path.parent.mkdir(parents=True, exist_ok=True)
    search_df.to_csv(search_log_path, index=False)

    iter_curve = best_evals_result["valid_0"]["auc"]
    iter_log_path = Path(config["training"]["tables_output_dir"]) / "training_log_lightgbm_iterations.csv"
    pd.DataFrame({"iteration": range(1, len(iter_curve) + 1), "valid_auc": iter_curve}).to_csv(
        iter_log_path, index=False
    )

    # best_model.predict_proba automatically uses best_iteration_ from
    # early stopping (LightGBM sklearn API default behavior).
    val_prob = best_model.predict_proba(X_val)[:, 1]
    logger.info(
        "[lightgbm] selected learning_rate=%s, best_iteration=%d (val AUROC=%.4f) in %.2fs",
        best_learning_rate, best_model.best_iteration_, best_auroc, runtime_seconds,
    )

    return {
        "model_name": "lightgbm",
        "model": best_model,
        "val_prob": val_prob,
        "runtime_seconds": runtime_seconds,
        "hyperparameters": {
            "learning_rate": best_learning_rate,
            "n_estimators_upper_bound": lgbm_config["n_estimators"],
            "best_iteration": int(best_model.best_iteration_),
            "num_leaves": lgbm_config["num_leaves"],
            "max_depth": lgbm_config["max_depth"],
            "early_stopping_rounds": early_stopping_rounds,
            "random_state": lgbm_config["random_state"],
        },
        "val_auroc": best_auroc,
        "training_log_path": str(search_log_path),
    }


# ---------------------------------------------------------------------------
# 3. PyTorch MLP, early stopping on validation loss
# ---------------------------------------------------------------------------

def train_mlp(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
    config: dict[str, Any], logger,
) -> dict[str, Any]:
    start = time.perf_counter()
    mlp_config = config["models"]["mlp_torch"]
    seed = mlp_config["random_state"]

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    X_train_t = torch.tensor(X_train.to_numpy(dtype=np.float32))
    y_train_t = torch.tensor(y_train.to_numpy(dtype=np.float32)).unsqueeze(1)
    X_val_t = torch.tensor(X_val.to_numpy(dtype=np.float32))
    y_val_t = torch.tensor(y_val.to_numpy(dtype=np.float32)).unsqueeze(1)

    model = MLPClassifierTorch(input_dim=X_train.shape[1], hidden_dims=mlp_config["hidden_dims"], seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=mlp_config["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=mlp_config["batch_size"], shuffle=True, generator=generator
    )

    best_val_loss = np.inf
    best_state_dict = None
    best_epoch = -1
    epochs_without_improvement = 0
    patience = mlp_config["early_stopping_patience"]
    max_epochs = mlp_config["max_epochs"]

    epoch_rows = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            n_batches += 1
        epoch_train_loss /= max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_prob_epoch = torch.sigmoid(val_logits).numpy().ravel()
            val_auroc_epoch = roc_auc_score(y_val, val_prob_epoch)

        is_best = val_loss < best_val_loss
        epoch_rows.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": val_loss,
            "val_auroc": val_auroc_epoch,
            "is_best": is_best,
        })

        if is_best:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or is_best:
            logger.info(
                "[mlp] epoch=%d train_loss=%.4f val_loss=%.4f val_auroc=%.4f%s",
                epoch, epoch_train_loss, val_loss, val_auroc_epoch, " (best)" if is_best else "",
            )

        if epochs_without_improvement >= patience:
            logger.info(
                "[mlp] early stopping at epoch=%d (no val_loss improvement for %d epochs; best epoch=%d)",
                epoch, patience, best_epoch,
            )
            break

    runtime_seconds = time.perf_counter() - start

    # Restore the best-on-validation weights (early stopping selection),
    # not the final epoch's weights.
    model.load_state_dict(best_state_dict)

    epoch_log_df = pd.DataFrame(epoch_rows)
    epoch_log_path = Path(config["training"]["tables_output_dir"]) / "training_log_mlp.csv"
    epoch_log_path.parent.mkdir(parents=True, exist_ok=True)
    epoch_log_df.to_csv(epoch_log_path, index=False)

    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(X_val_t)).numpy().ravel()
    best_val_auroc = roc_auc_score(y_val, val_prob)

    logger.info(
        "[mlp] restored best epoch=%d (val_loss=%.4f, val AUROC=%.4f) in %.2fs",
        best_epoch, best_val_loss, best_val_auroc, runtime_seconds,
    )

    return {
        "model_name": "mlp_torch",
        "model": model,
        "val_prob": val_prob,
        "runtime_seconds": runtime_seconds,
        "hyperparameters": {
            "hidden_dims": mlp_config["hidden_dims"],
            "max_epochs_upper_bound": max_epochs,
            "epochs_run": len(epoch_rows),
            "best_epoch": best_epoch,
            "learning_rate": mlp_config["learning_rate"],
            "batch_size": mlp_config["batch_size"],
            "early_stopping_patience": patience,
            "random_state": seed,
            "input_dim": X_train.shape[1],
        },
        "val_auroc": best_val_auroc,
        "training_log_path": str(epoch_log_path),
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_model_artifact(result: dict[str, Any], config: dict[str, Any]) -> Path:
    models_dir = Path(config["training"]["models_output_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    name = result["model_name"]

    if name == "mlp_torch":
        out_path = models_dir / "mlp_torch.pt"
        torch.save({
            "state_dict": result["model"].state_dict(),
            "input_dim": result["hyperparameters"]["input_dim"],
            "hidden_dims": result["hyperparameters"]["hidden_dims"],
        }, out_path)
    else:
        out_path = models_dir / f"{name}.joblib"
        joblib.dump(result["model"], out_path)

    return out_path


def build_hyperparameters_table(results: list[dict[str, Any]], seed: int) -> pd.DataFrame:
    rows = []
    for result in results:
        for hp_name, hp_value in result["hyperparameters"].items():
            rows.append({
                "model": result["model_name"],
                "hyperparameter": hp_name,
                "value": hp_value,
            })
        rows.append({
            "model": result["model_name"],
            "hyperparameter": "global_random_seed",
            "value": seed,
        })
        rows.append({
            "model": result["model_name"],
            "hyperparameter": "val_auroc",
            "value": round(result["val_auroc"], 4),
        })
        rows.append({
            "model": result["model_name"],
            "hyperparameter": "runtime_seconds",
            "value": round(result["runtime_seconds"], 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_training(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    config = config or load_config()
    seed = config["project"]["random_seed"]
    set_global_seed(seed)
    logger = get_logger(log_file=config["logging"]["log_file"])
    logger.info("=== train_models.py started (seed=%d) ===", seed)

    overall_start = time.perf_counter()

    X_train, y_train, X_val, y_val, train_df, val_df, _preprocessor = prepare_training_data(config, logger)
    logger.info("X_train: %s | X_val: %s", X_train.shape, X_val.shape)

    results = [
        train_logistic_regression(X_train, y_train, X_val, y_val, config, logger),
        train_lightgbm(X_train, y_train, X_val, y_val, config, logger),
        train_mlp(X_train, y_train, X_val, y_val, config, logger),
    ]

    for result in results:
        model_path = save_model_artifact(result, config)
        logger.info("[%s] model saved to %s", result["model_name"], model_path)

        pred_path = _save_val_predictions(val_df, y_val, result["val_prob"], result["model_name"], config)
        logger.info("[%s] validation predictions saved to %s", result["model_name"], pred_path)

    hyperparams_df = build_hyperparameters_table(results, seed)
    hyperparams_path = Path(config["training"]["hyperparameters_table_path"])
    hyperparams_path.parent.mkdir(parents=True, exist_ok=True)
    hyperparams_df.to_csv(hyperparams_path, index=False)
    logger.info("Hyperparameters table saved to %s", hyperparams_path)

    total_runtime_seconds = time.perf_counter() - overall_start

    log_run_metadata(
        output_path=config["training"]["run_metadata_path"],
        seed=seed,
        extra={
            "step": "train_models",
            "total_runtime_seconds": round(total_runtime_seconds, 2),
            "n_train": len(train_df),
            "n_val": len(val_df),
            "test_set_used": False,
            "models": {
                r["model_name"]: {
                    "val_auroc": round(r["val_auroc"], 4),
                    "runtime_seconds": round(r["runtime_seconds"], 2),
                    "hyperparameters": r["hyperparameters"],
                }
                for r in results
            },
        },
    )
    logger.info("=== train_models.py finished in %.2fs ===", total_runtime_seconds)

    return results


if __name__ == "__main__":
    run_training()
