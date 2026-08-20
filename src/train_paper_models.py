"""Model training for the paper pipeline (notebooks/02_train_models.ipynb):
regularized logistic regression + LightGBM, with class-imbalance handling,
validation-only hyperparameter selection/early stopping, and long-format
validation predictions for notebooks/03_calibration.ipynb.

TRAIN/VAL/TEST discipline: this module only ever touches train_df/val_df
(built by notebooks/01_cohort_audit.ipynb's frozen preprocessor). It never
loads or references the test split.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models import build_lightgbm, build_logistic_regression

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """neg/pos ratio in the TRAINING labels only — the standard LightGBM
    class-imbalance handle. Computed dynamically, never hardcoded.
    """
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos == 0:
        raise ValueError("No positive (mortality=1) cases in training data — cannot compute scale_pos_weight.")
    return n_neg / n_pos


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
    config: dict[str, Any], logger,
) -> dict[str, Any]:
    start = time.perf_counter()
    C_grid = config["models"]["logistic_regression"]["C_grid"]

    search_rows = []
    best_model, best_auroc, best_C = None, -np.inf, None

    for C in C_grid:
        model = build_logistic_regression(config, C=C)
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        val_auroc = roc_auc_score(y_val, val_prob)
        search_rows.append({"C": C, "val_auroc": val_auroc})
        logger.info("[logistic_regression] C=%s -> val AUROC=%.4f", C, val_auroc)
        if val_auroc > best_auroc:
            best_auroc, best_C, best_model = val_auroc, C, model

    runtime_seconds = time.perf_counter() - start
    val_prob = best_model.predict_proba(X_val)[:, 1]
    logger.info("[logistic_regression] selected C=%s (val AUROC=%.4f) in %.2fs", best_C, best_auroc, runtime_seconds)

    lr_config = config["models"]["logistic_regression"]
    return {
        "model_name": "logistic_regression",
        "model": best_model,
        "val_prob": val_prob,
        "runtime_seconds": runtime_seconds,
        "hyperparameters": {
            "penalty": lr_config["penalty"],
            "solver": lr_config["solver"],
            "C": best_C,
            "max_iter": lr_config["max_iter"],
            "class_weight": lr_config["class_weight"],
            "random_state": lr_config["random_state"],
        },
        "val_auroc": best_auroc,
        "search_log": pd.DataFrame(search_rows).assign(selected=lambda d: d["C"] == best_C),
    }


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
    scale_pos_weight = compute_scale_pos_weight(y_train) if lgbm_config.get("auto_scale_pos_weight") else None
    logger.info("[lightgbm] scale_pos_weight (neg/pos in train) = %s", scale_pos_weight)

    search_rows = []
    best_model, best_auroc, best_lr = None, -np.inf, None

    for lr in learning_rate_grid:
        model = build_lightgbm(config, learning_rate=lr)
        if scale_pos_weight is not None:
            model.set_params(scale_pos_weight=scale_pos_weight)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        val_auroc = float(model.best_score_["valid_0"]["auc"])
        search_rows.append({
            "learning_rate": lr, "best_iteration": int(model.best_iteration_), "val_auroc": val_auroc,
        })
        logger.info("[lightgbm] learning_rate=%s -> best_iteration=%d, val AUROC=%.4f",
                    lr, model.best_iteration_, val_auroc)
        if val_auroc > best_auroc:
            best_auroc, best_lr, best_model = val_auroc, lr, model

    runtime_seconds = time.perf_counter() - start
    val_prob = best_model.predict_proba(X_val)[:, 1]
    logger.info("[lightgbm] selected learning_rate=%s, best_iteration=%d (val AUROC=%.4f) in %.2fs",
                best_lr, best_model.best_iteration_, best_auroc, runtime_seconds)

    return {
        "model_name": "lightgbm",
        "model": best_model,
        "val_prob": val_prob,
        "runtime_seconds": runtime_seconds,
        "hyperparameters": {
            "learning_rate": best_lr,
            "n_estimators_upper_bound": lgbm_config["n_estimators"],
            "best_iteration": int(best_model.best_iteration_),
            "num_leaves": lgbm_config["num_leaves"],
            "max_depth": lgbm_config["max_depth"],
            "early_stopping_rounds": early_stopping_rounds,
            "scale_pos_weight": scale_pos_weight,
            "random_state": lgbm_config["random_state"],
        },
        "val_auroc": best_auroc,
        "search_log": pd.DataFrame(search_rows).assign(selected=lambda d: d["learning_rate"] == best_lr),
    }


def build_hyperparameters_table(results: list[dict[str, Any]], seed: int) -> pd.DataFrame:
    rows = []
    for result in results:
        for hp_name, hp_value in result["hyperparameters"].items():
            rows.append({"model": result["model_name"], "hyperparameter": hp_name, "value": hp_value})
        rows.append({"model": result["model_name"], "hyperparameter": "global_random_seed", "value": seed})
        rows.append({"model": result["model_name"], "hyperparameter": "val_auroc", "value": round(result["val_auroc"], 4)})
        rows.append({"model": result["model_name"], "hyperparameter": "runtime_seconds", "value": round(result["runtime_seconds"], 2)})
    return pd.DataFrame(rows)


def build_long_format_predictions(
    results: list[dict[str, Any]], df: pd.DataFrame, y: pd.Series, prob_key: str = "val_prob",
) -> pd.DataFrame:
    """outputs/predictions/{val,test}_predictions.parquet schema: one row
    per (stay_id, model), columns stay_id, model_name, predicted_prob,
    true_label. Uses stay_id (one row per ICU stay) per task spec, falling
    back to subject_id if stay_id is not present in df.
    """
    id_col = "stay_id" if "stay_id" in df.columns else "subject_id"
    frames = []
    for result in results:
        frames.append(pd.DataFrame({
            "stay_id": df[id_col].reset_index(drop=True),
            "model_name": result["model_name"],
            "predicted_prob": result[prob_key],
            "true_label": y.reset_index(drop=True),
        }))
    return pd.concat(frames, ignore_index=True)
