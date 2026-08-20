"""Post-hoc probability calibration: Platt scaling, isotonic regression,
temperature scaling (MLP only), and calibration-quality metrics.

DESIGN NOTE: these calibrators operate on already-computed prediction
SCORES (1-D probability or logit arrays), not on raw features + a
classifier. This is the standard approach when you have a trained model's
validation-set predictions and want to learn a probability -> probability
(or logit -> probability) recalibration mapping, and it is what
notebooks/03_calibration.ipynb uses: each of the three baselines from
src/train_models.py is calibrated using ONLY its own validation
predictions (src/train_models.py already enforces the model itself never
saw the validation labels for gradient updates — only for hyperparameter
selection / early stopping).

FITTING CAVEAT (by design, per project instructions): Platt scaling and
isotonic regression here are fit and evaluated on the SAME validation set.
This is the explicitly requested workflow, but note it is optimistic
in-sample calibration quality — isotonic regression in particular can
overfit a modestly sized validation set. A proper held-out assessment of
calibration quality belongs to a later, separate test-set evaluation step
(not performed here — test labels are never touched by this module).
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6  # clip bound to keep logit()/log() finite
SLOPE_WARNING_THRESHOLD = 10.0  # |slope| beyond this signals an unstable calibration-regression fit


def _clip_probs(y_prob: np.ndarray, eps: float = EPS) -> np.ndarray:
    return np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1 - eps)


def logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = _clip_probs(p, eps)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------

class PlattScaler:
    """Classic Platt scaling: fit a 1-D logistic regression of the true
    label on logit(uncalibrated probability), i.e. y ~ sigmoid(a * logit(p) + b).

    `a` (slope) and `b` (intercept) are exactly the "calibration slope" and
    "calibration intercept" diagnostic statistics computed elsewhere in
    this module for the uncalibrated predictions — Platt scaling and the
    calibration-regression diagnostic are the same fit.
    """

    def __init__(self):
        self.slope_: float | None = None
        self.intercept_: float | None = None
        self._is_fitted = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "PlattScaler":
        logit_p = logit(y_prob).reshape(-1, 1)
        # Effectively unregularized MLE logistic fit (very large C), matching
        # the classic Platt/calibration-regression formulation rather than a
        # regularized sklearn default.
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(logit_p, y_true)
        self.slope_ = float(lr.coef_[0][0])
        self.intercept_ = float(lr.intercept_[0])
        self._is_fitted = True

        if abs(self.slope_) > SLOPE_WARNING_THRESHOLD:
            warnings.warn(
                f"PlattScaler fit an unstable calibration slope ({self.slope_:.2f}). "
                "This typically means the input probabilities have very low variance "
                "(e.g. a heavily regularized base model and/or a small validation set) "
                "so the near-unregularized calibration-regression fit is poorly "
                "conditioned. Treat this slope/intercept as unreliable; consider a "
                "larger validation set or less aggressive base-model regularization.",
                stacklevel=2,
            )

        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("PlattScaler.fit() must be called before transform().")
        logit_p = logit(y_prob)
        z = self.slope_ * logit_p + self.intercept_
        return 1.0 / (1.0 + np.exp(-z))

    def fit_transform(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.fit(y_prob, y_true).transform(y_prob)


class IsotonicCalibrator:
    """Non-parametric monotonic calibration mapping p -> calibrated p."""

    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._is_fitted = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self._iso.fit(y_prob, y_true)
        self._is_fitted = True
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("IsotonicCalibrator.fit() must be called before transform().")
        return self._iso.transform(y_prob)

    def fit_transform(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.fit(y_prob, y_true).transform(y_prob)


class DensityWeightedAdaptiveCalibrator:
    """Density-Weighted Adaptive Calibration (DWAC).

    NOVELTY NOTE: this is the one genuinely new algorithmic component in
    this project (see docs/*.md "Novelty verdict" — Platt scaling,
    isotonic regression, and entropy-based referral are all established,
    unmodified methods; DWAC is not).

    Motivation: isotonic regression is flexible but overfits in regions of
    predicted-probability space with few supporting validation samples;
    Platt scaling is smooth and low-variance but too rigid to capture
    non-monotonic local miscalibration. DWAC blends the two PER PREDICTION,
    weighted by the local density of validation samples around that
    prediction's uncalibrated probability:

        p_dwac(p) = w(p) * p_isotonic(p) + (1 - w(p)) * p_platt(p)

    where w(p) in [0, 1] is a Gaussian-kernel density estimate of how many
    validation predictions fall near p (on the logit scale, so density is
    comparable across the full [0, 1] range), min-max normalized against
    the maximum density observed anywhere in the validation set and capped
    at 1. High local density -> trust the flexible isotonic fit more; low
    local density -> fall back toward the smoother, lower-variance Platt
    fit. Bandwidth is chosen by Silverman's rule of thumb on the
    logit-transformed validation probabilities (no hyperparameter tuning
    on held-out data).
    """

    def __init__(self, density_floor: float = 0.05):
        # Predictions in the lowest `density_floor` fraction of the observed
        # density range are treated as maximally sparse (w=0, pure Platt).
        self.density_floor = density_floor
        self._platt = PlattScaler()
        self._iso = IsotonicCalibrator()
        self._val_logits: np.ndarray | None = None
        self._bandwidth: float | None = None
        self._max_density: float | None = None
        self._is_fitted = False

    @staticmethod
    def _silverman_bandwidth(x: np.ndarray) -> float:
        n = len(x)
        std = np.std(x, ddof=1)
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        sigma = min(std, iqr / 1.349) if iqr > 0 else std
        sigma = max(sigma, 1e-6)
        return 0.9 * sigma * n ** (-1 / 5)

    def _density(self, query_logits: np.ndarray) -> np.ndarray:
        # Gaussian KDE of query points against stored validation logits,
        # vectorized: shape (n_query, n_val) -> mean over n_val.
        diff = (query_logits[:, None] - self._val_logits[None, :]) / self._bandwidth
        kernel = np.exp(-0.5 * diff ** 2)
        return kernel.mean(axis=1)

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "DensityWeightedAdaptiveCalibrator":
        y_prob = np.asarray(y_prob, dtype=np.float64)
        y_true = np.asarray(y_true)
        self._platt.fit(y_prob, y_true)
        self._iso.fit(y_prob, y_true)
        self._val_logits = logit(y_prob)
        self._bandwidth = self._silverman_bandwidth(self._val_logits)
        self._max_density = float(self._density(self._val_logits).max())
        self._is_fitted = True
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("DensityWeightedAdaptiveCalibrator.fit() must be called before transform().")
        y_prob = np.asarray(y_prob, dtype=np.float64)
        query_logits = logit(y_prob)
        density = self._density(query_logits) / self._max_density
        weight = np.clip((density - self.density_floor) / (1 - self.density_floor), 0.0, 1.0)

        p_platt = self._platt.transform(y_prob)
        p_iso = self._iso.transform(y_prob)
        return weight * p_iso + (1 - weight) * p_platt

    def fit_transform(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.fit(y_prob, y_true).transform(y_prob)


class TemperatureScaler(torch.nn.Module):
    """Single-parameter temperature scaling for neural network logits.
    MLP-only, per project instructions (Platt/isotonic apply to all three
    baselines; temperature scaling requires access to pre-sigmoid logits,
    which only the PyTorch MLP naturally exposes here).
    """

    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50) -> "TemperatureScaler":
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = torch.nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits).squeeze(), labels.float())
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits_t = torch.as_tensor(np.asarray(logits, dtype=np.float32))
            calibrated = torch.sigmoid(self.forward(logits_t).squeeze())
        return calibrated.numpy()


# ---------------------------------------------------------------------------
# Calibration-quality metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) over equal-WIDTH bins."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


def adaptive_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Adaptive ECE (a.k.a. ACE): same weighted-bin-gap formula as ECE, but
    bin edges are equal-FREQUENCY (quantiles of y_prob) rather than
    equal-width, so each bin holds roughly the same number of samples. This
    is more robust than fixed-width ECE when predicted probabilities are
    concentrated in a narrow range (typical for imbalanced outcomes like
    in-hospital mortality).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.unique(np.quantile(y_prob, quantiles))
    if len(bin_edges) < 2:
        # All predictions identical (degenerate); no calibration gap is computable.
        return 0.0

    ace = 0.0
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ace += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ace)


def maximum_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Maximum Calibration Error (MCE) over equal-width bins."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    max_error = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        max_error = max(max_error, abs(bin_acc - bin_conf))

    return float(max_error)


def negative_log_likelihood(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean negative log-likelihood (binary cross-entropy / log loss)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    p = _clip_probs(y_prob)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((y_prob - y_true) ** 2))


def calibration_intercept_slope(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Calibration-in-the-large (intercept) and calibration slope: fit
    y ~ sigmoid(slope * logit(p) + intercept) via (near-)unregularized
    logistic regression. Intercept ~ 0 and slope ~ 1 indicate good
    calibration; intercept > 0 indicates systematic underprediction,
    slope < 1 indicates predictions that are too extreme (overconfident).

    This is the same fit PlattScaler performs; exposed standalone here so
    it can be reported as a diagnostic for uncalibrated AND calibrated
    predictions without needing to construct a PlattScaler object.
    """
    scaler = PlattScaler().fit(y_prob, y_true)
    return scaler.intercept_, scaler.slope_


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ece_bins: tuple[int, ...] = (10, 15),
    adaptive_bins: int = 10,
    mce_bins: int = 10,
) -> dict[str, Any]:
    """One-stop calibration report for a single set of predictions."""
    intercept, slope = calibration_intercept_slope(y_true, y_prob)

    metrics: dict[str, Any] = {
        "brier_score": brier_score(y_true, y_prob),
        "adaptive_ece": adaptive_calibration_error(y_true, y_prob, n_bins=adaptive_bins),
        "mce": maximum_calibration_error(y_true, y_prob, n_bins=mce_bins),
        "nll": negative_log_likelihood(y_true, y_prob),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }
    for n_bins in ece_bins:
        metrics[f"ece_{n_bins}bins"] = expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    return metrics
