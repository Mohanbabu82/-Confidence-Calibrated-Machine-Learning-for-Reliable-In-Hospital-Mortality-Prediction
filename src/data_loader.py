"""Data loading utilities.

NOTE: This project does not ship, download, or generate any real medical
patient data. Paths below are placeholders — point them at your own local,
properly governed dataset before running the pipeline.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config


def load_raw_data(config: dict | None = None) -> pd.DataFrame:
    """Load the raw dataset from the configured placeholder path.

    Raises FileNotFoundError with a clear message if the placeholder path
    has not been replaced with a real local file.
    """
    config = config or load_config()
    raw_path = Path(config["paths"]["data_raw"])

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at placeholder path '{raw_path}'. "
            "This project ships no patient data. Update "
            "configs/config.yaml -> paths.data_raw to point at your own "
            "local, governance-approved dataset."
        )

    return pd.read_csv(raw_path)


def load_processed_data(config: dict | None = None) -> pd.DataFrame:
    """Load the processed dataset (Parquet) from the configured placeholder path."""
    config = config or load_config()
    processed_path = Path(config["paths"]["data_processed"])

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data file not found at placeholder path '{processed_path}'. "
            "Run the preprocessing pipeline (src/preprocess.py) on your own "
            "local dataset first."
        )

    return pd.read_parquet(processed_path)
