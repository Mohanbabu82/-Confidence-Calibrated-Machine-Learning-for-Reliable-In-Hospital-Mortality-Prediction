"""Selective prediction / clinician-referral safety layer.

================================================================================
CLINICAL FRAMING — READ BEFORE USE
================================================================================
This module implements a CLINICAL DECISION SUPPORT referral layer. It is
NOT, and must never be deployed as, an autonomous diagnostic or
mortality-prediction system. Concretely:

  - "Accepted" (non-referred) cases are those where the model's calibrated
    confidence is high enough to be offered as a decision-support input
    ALONGSIDE clinician judgment — the model's output is advisory, not a
    diagnosis, and does not replace clinician review of the underlying
    case.
  - "Referred" cases are those where model confidence is too low to
    support even an advisory output; these REQUIRE full, standard-of-care
    clinician review, unassisted by this model's prediction for that case.
  - No coverage level (including 100% / 0% referral) makes this system
    suitable for autonomous, unsupervised operation. Coverage/referral-rate
    is a tool for triaging clinician attention, not a substitute for it.
================================================================================

Implements: predictive-entropy uncertainty ranking, referral-threshold
selection at fixed referral rates, and the standard selective-prediction
metric suite (coverage, selective risk, sensitivity/specificity/PPV/NPV,
Brier score, ECE) evaluated on the ACCEPTED (non-referred) subset only.

Terminology note: "selective risk" (the selective-classification-literature
term for the empirical 0/1 loss on the accepted subset) and
"accepted-case error rate" are the same quantity, reported under both names
because the project brief requests both — one is the research-literature
term, the other the plain-language name used in the clinical policy table.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.calibration import brier_score, expected_calibration_error

CLINICAL_DISCLAIMER = (
    "CLINICAL DECISION SUPPORT ONLY — NOT AN AUTONOMOUS DIAGNOSTIC SYSTEM. "
    "This selective-prediction / referral layer is designed to route the "
    "model's least-confident predictions to mandatory clinician review. "
    "Predictions for 'accepted' (non-referred) cases are decision-support "
    "inputs for a clinician to weigh alongside the full clinical picture, "
    "not a diagnosis and not a recommendation to withhold clinician "
    "review. 'Referred' cases require full, standard-of-care clinician "
    "review because model confidence was assessed as too low to support "
    "even an advisory output. No referral-rate setting makes this system "
    "suitable for unsupervised or autonomous operation."
)


# ---------------------------------------------------------------------------
# Uncertainty ranking
# ---------------------------------------------------------------------------

def predictive_entropy(y_prob: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Binary predictive entropy: H(p) = -p*log2(p) - (1-p)*log2(1-p).

    Maximized at p=0.5 (maximally uncertain) and 0 at p=0 or p=1 (maximally
    confident). Used as the uncertainty score for ranking cases for
    referral: higher entropy -> more uncertain -> referred first.
    """
    p = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def rank_by_uncertainty(y_prob: np.ndarray) -> np.ndarray:
    """Return row indices ordered from MOST to LEAST uncertain (by predictive entropy)."""
    entropy = predictive_entropy(y_prob)
    # Stable sort so ties break by original (row) order, keeping the
    # referral set deterministic across reruns given identical input.
    return np.argsort(-entropy, kind="stable")


def compute_accepted_mask(y_prob: np.ndarray, referral_rate: float) -> np.ndarray:
    """Boolean mask, True = accepted (automated/decision-support-eligible),
    False = referred (mandatory clinician review). The `referral_rate`
    fraction of MOST uncertain cases (by predictive entropy) are referred.
    """
    if not (0.0 <= referral_rate <= 1.0):
        raise ValueError(f"referral_rate must be in [0, 1], got {referral_rate}")

    n = len(y_prob)
    n_referred = int(round(referral_rate * n))
    order = rank_by_uncertainty(y_prob)  # most uncertain first

    accepted_mask = np.ones(n, dtype=bool)
    accepted_mask[order[:n_referred]] = False
    return accepted_mask


# ---------------------------------------------------------------------------
# Metrics
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


def compute_selective_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    referral_rate: float,
    threshold: float = 0.5,
    ece_bins: int = 10,
) -> dict[str, Any]:
    """Compute the full selective-prediction metric suite at one referral
    rate. All metrics except coverage/n_accepted/n_referred are computed on
    the ACCEPTED (non-referred) subset only.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    accepted_mask = compute_accepted_mask(y_prob, referral_rate)
    n_accepted = int(accepted_mask.sum())
    n_referred = n - n_accepted
    coverage = n_accepted / n

    y_true_acc = y_true[accepted_mask]
    y_prob_acc = y_prob[accepted_mask]
    y_pred_acc = (y_prob_acc >= threshold).astype(int)

    if n_accepted == 0:
        warnings.warn(
            f"referral_rate={referral_rate:.2f} referred ALL cases (n_accepted=0); "
            "all accepted-subset metrics are undefined (NaN).",
            stacklevel=2,
        )
        return {
            "referral_rate": referral_rate,
            "coverage": 0.0,
            "n_accepted": 0,
            "n_referred": n_referred,
            "selective_risk": float("nan"),
            "accepted_case_error_rate": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "ppv": float("nan"),
            "npv": float("nan"),
            "brier_score": float("nan"),
            "ece": float("nan"),
            "decision_threshold": threshold,
        }

    tp, tn, fp, fn = _confusion_counts(y_true_acc, y_pred_acc)

    # selective_risk == accepted_case_error_rate: both are the empirical
    # 0/1 loss on the accepted subset (see module docstring).
    error_rate = _safe_divide(fp + fn, n_accepted)

    sensitivity = _safe_divide(tp, tp + fn)   # recall for the positive (death) class
    specificity = _safe_divide(tn, tn + fp)
    ppv = _safe_divide(tp, tp + fp)           # precision
    npv = _safe_divide(tn, tn + fn)

    metrics = {
        "referral_rate": referral_rate,
        "coverage": coverage,
        "n_accepted": n_accepted,
        "n_referred": n_referred,
        "selective_risk": error_rate,
        "accepted_case_error_rate": error_rate,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "brier_score": brier_score(y_true_acc, y_prob_acc),
        "ece": expected_calibration_error(y_true_acc, y_prob_acc, n_bins=ece_bins),
        "decision_threshold": threshold,
    }
    return metrics


def evaluate_referral_policy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    referral_rates: list[float] = (0.0, 0.05, 0.10, 0.20, 0.30),
    threshold: float = 0.5,
    ece_bins: int = 10,
) -> pd.DataFrame:
    """Evaluate the full selective-prediction metric suite across a grid of
    referral rates. Returns one row per referral rate.
    """
    rows = [
        compute_selective_metrics(y_true, y_prob, rr, threshold=threshold, ece_bins=ece_bins)
        for rr in referral_rates
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibration-method selection (data-driven, not hardcoded)
# ---------------------------------------------------------------------------

def select_best_calibration_method(
    calibration_metrics_df: pd.DataFrame,
    model_name: str,
    candidate_methods: list[str],
    metric_col: str = "ece_10bins",
) -> str:
    """Pick the calibration method with the lowest validation ECE (10 bins)
    among the given candidates for one model, using
    outputs/tables/table_3_calibration_metrics.csv (produced by
    notebooks/03_calibration.ipynb). Data-driven selection — never hardcode
    which calibration method is "best".
    """
    subset = calibration_metrics_df[
        (calibration_metrics_df["model"] == model_name)
        & (calibration_metrics_df["calibration_method"].isin(candidate_methods))
    ]
    if subset.empty:
        raise ValueError(
            f"No rows found in calibration metrics table for model='{model_name}' "
            f"with calibration_method in {candidate_methods}. Run "
            "notebooks/03_calibration.ipynb first."
        )
    best_row = subset.loc[subset[metric_col].idxmin()]
    return str(best_row["calibration_method"])
