"""DeLong's test for comparing two correlated AUROCs on the same paired
test set. Standard algorithm (Sun & Xu, 2014 fast DeLong; equivalent to
DeLong, DeLong & Clarke-Pearson, 1988). No external stats-test dependency
beyond scipy.stats.norm for the final p-value.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2


def _fast_delong(predictions: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """predictions: shape (n_models, n_samples). Returns (auc, covariance matrix)."""
    pos = predictions[:, y_true == 1]
    neg = predictions[:, y_true == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = predictions.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(predictions[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def delong_roc_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict:
    """Two-sided DeLong test for AUC(prob_a) == AUC(prob_b) on the SAME
    paired samples (y_true shared). Returns aucs, their difference, its
    standard error, z-statistic, and two-sided p-value.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true, kind="stable")  # positives first, matches _fast_delong's pos/neg split assumption
    y_sorted = y_true[order]
    preds = np.vstack([np.asarray(prob_a)[order], np.asarray(prob_b)[order]])

    aucs, cov = _fast_delong(preds, y_sorted)
    auc_diff = aucs[0] - aucs[1]
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = np.sqrt(max(var_diff, 0.0))
    z = auc_diff / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "auc_diff": float(auc_diff),
        "se_diff": float(se), "z": float(z), "p_value": float(p),
    }
