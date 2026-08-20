import numpy as np

from src.calibration import expected_calibration_error, maximum_calibration_error


def test_ece_perfect_calibration():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece == 0.0


def test_mce_non_negative():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.2, 0.6, 0.9, 0.4])
    mce = maximum_calibration_error(y_true, y_prob, n_bins=5)
    assert mce >= 0.0
