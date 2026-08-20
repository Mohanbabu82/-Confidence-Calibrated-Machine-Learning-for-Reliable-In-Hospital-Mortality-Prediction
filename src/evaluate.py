"""Evaluation metrics for discrimination and calibration quality."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.calibration import expected_calibration_error, maximum_calibration_error


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """Compute the full discrimination + calibration metric suite for a model's predictions."""
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "mce": maximum_calibration_error(y_true, y_prob, n_bins=n_bins),
    }
