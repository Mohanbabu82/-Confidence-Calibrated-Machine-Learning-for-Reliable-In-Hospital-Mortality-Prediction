"""Model factory: baseline and gradient-boosted classifiers, plus a small
PyTorch MLP, all seeded for reproducibility.
"""
from __future__ import annotations

from typing import Any

import lightgbm as lgb
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.reproducibility import DEFAULT_SEED


def build_logistic_regression(config: dict[str, Any], C: float | None = None) -> LogisticRegression:
    """Build a regularized logistic regression.

    `C_grid` in config is a hyperparameter-search candidate list, not a
    constructor argument, so it is excluded here; pass the selected `C`
    explicitly (e.g. chosen via validation-set AUROC in src/train_models.py).
    """
    params = {k: v for k, v in config["models"]["logistic_regression"].items() if k != "C_grid"}
    if C is not None:
        params["C"] = C
    return LogisticRegression(**params)


def build_random_forest(config: dict[str, Any]) -> RandomForestClassifier:
    params = config["models"]["random_forest"]
    return RandomForestClassifier(**params)


def build_xgboost(config: dict[str, Any]) -> xgb.XGBClassifier:
    params = config["models"]["xgboost"]
    return xgb.XGBClassifier(**params, eval_metric="logloss")


def build_lightgbm(config: dict[str, Any], learning_rate: float | None = None) -> lgb.LGBMClassifier:
    """Build a LightGBM classifier.

    `learning_rate_grid` and `early_stopping_rounds` in config are
    hyperparameter-search / training-loop controls, not constructor
    arguments, so they are excluded here; pass the selected
    `learning_rate` explicitly (e.g. chosen via validation-set AUROC in
    src/train_models.py). early_stopping_rounds is applied via a callback
    at .fit() time, not the constructor. auto_scale_pos_weight is a flag
    read directly by the caller (e.g. src/train_paper_models.py, which
    then sets scale_pos_weight via .set_params() after computing it from
    the training labels) — not a constructor argument itself.
    """
    exclude = {"learning_rate_grid", "early_stopping_rounds", "auto_scale_pos_weight"}
    params = {k: v for k, v in config["models"]["lightgbm"].items() if k not in exclude}
    if learning_rate is not None:
        params["learning_rate"] = learning_rate
    return lgb.LGBMClassifier(**params)


class MLPClassifierTorch(nn.Module):
    """Small feed-forward network for binary mortality classification."""

    def __init__(self, input_dim: int, hidden_dims: list[int], seed: int = DEFAULT_SEED):
        super().__init__()
        torch.manual_seed(seed)

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
