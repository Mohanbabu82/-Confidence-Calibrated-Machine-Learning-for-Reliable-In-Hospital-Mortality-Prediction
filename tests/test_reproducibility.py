import numpy as np

from src.reproducibility import DEFAULT_SEED, set_global_seed


def test_seed_reproducibility():
    set_global_seed(DEFAULT_SEED)
    a = np.random.rand(5)
    set_global_seed(DEFAULT_SEED)
    b = np.random.rand(5)
    assert np.array_equal(a, b)
