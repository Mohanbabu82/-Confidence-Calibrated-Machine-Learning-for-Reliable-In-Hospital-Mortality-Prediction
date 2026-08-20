"""Reproducibility utilities: global seeding and run logging.

Every script/notebook entry point should call `set_global_seed()` and
`get_logger()` at the start of execution so results are deterministic
and every run is traceable.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed all relevant RNGs (python, numpy, torch if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_logger(name: str = "calibrated_clinical_ai", log_file: str | Path = "logs/run.log") -> logging.Logger:
    """Return a logger that writes to console and to a run log file."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    return logger


def log_run_metadata(
    output_path: str | Path = "logs/run_metadata.json",
    seed: int = DEFAULT_SEED,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record environment/run metadata (timestamp, seed, package versions) to JSON for provenance."""
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "python_version": sys.version,
        "platform": sys.platform,
    }

    for pkg in ["numpy", "pandas", "sklearn", "xgboost", "lightgbm", "torch", "scipy", "statsmodels", "shap"]:
        try:
            module = __import__(pkg)
            metadata[f"{pkg}_version"] = getattr(module, "__version__", "unknown")
        except ImportError:
            metadata[f"{pkg}_version"] = "not installed"

    if extra:
        metadata.update(extra)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    set_global_seed(DEFAULT_SEED)
    logger = get_logger()
    logger.info("Reproducibility check: seed=%d", DEFAULT_SEED)
    meta = log_run_metadata(seed=DEFAULT_SEED)
    logger.info("Run metadata logged: %s", json.dumps(meta, indent=2))
