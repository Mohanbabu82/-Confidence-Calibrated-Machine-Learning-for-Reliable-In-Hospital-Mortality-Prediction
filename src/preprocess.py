"""Preprocessing pipeline for the first-24h ICU mortality cohort.

CHECK_SCHEMA / ASSUMPTION: This module expects a local parquet file
(default: data/processed/cohort_features.parquet) containing the JOIN of
sql/cohort.sql's output and sql/features.sql's output — i.e. features PLUS
identifier columns (subject_id, hadm_id, stay_id) PLUS the target column
(in_hospital_mortality). Identifiers are required at this stage only for
patient-level splitting and overlap/duplicate auditing; they are dropped
before anything is handed to a model (see drop_identifier_columns() and
ClinicalPreprocessor, neither of which ever treats an identifier as a
feature). This differs from src/data_extract.py's `output_path`, which
writes an identifier-free, model-ready table for a later modeling stage —
verify which file you actually have before running this module, and adjust
`config["preprocessing"]["cohort_features_path"]` if your extraction step
uses a different filename.

All preprocessing objects (imputers, scaler, one-hot encoder) are fit on
the TRAINING split only and reused (transform-only) on validation/test, to
avoid any information leakage from val/test into the fitted statistics.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_config
from src.reproducibility import DEFAULT_SEED

# Columns that identify a patient/admission/stay. Required for splitting
# and auditing in this module, but MUST be excluded from any feature
# matrix passed to a model (see drop_identifier_columns()).
IDENTIFIER_COLUMNS = ["subject_id", "hadm_id", "stay_id"]

DEFAULT_TARGET_COLUMN = "in_hospital_mortality"
DEFAULT_SPLIT_ID_COLUMN = "subject_id"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_cohort_features(path: str | Path | None = None, config: dict | None = None) -> pd.DataFrame:
    """Load the joined cohort+features table (identifiers + features + target)."""
    config = config or load_config()
    if path is None:
        path = config.get("preprocessing", {}).get(
            "cohort_features_path", "data/processed/cohort_features.parquet"
        )
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Cohort/features file not found at '{path}'. This project ships "
            "no MIMIC-IV data. Run sql/cohort.sql + sql/features.sql (see "
            "src/data_extract.py) against your own local, authorized extract "
            "first, and ensure the joined output is written to this path."
        )

    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Duplicate / integrity checks
# ---------------------------------------------------------------------------

@dataclass
class DuplicateCheckResult:
    n_rows: int
    n_fully_duplicate_rows: int
    n_duplicate_ids: int
    duplicate_id_values: list = field(default_factory=list)


def check_duplicates(df: pd.DataFrame, id_col: str = DEFAULT_SPLIT_ID_COLUMN) -> DuplicateCheckResult:
    """Check for fully duplicated rows and duplicate patient identifiers.

    A duplicate id_col value indicates more than one row per patient, which
    would violate the "one row per patient" cohort design (see
    sql/cohort.sql's first-ICU-stay-per-patient logic) and must be resolved
    before splitting, or patient-level splitting cannot guarantee no
    cross-split leakage.
    """
    n_fully_duplicate = int(df.duplicated(keep=False).sum())

    id_counts = df[id_col].value_counts()
    duplicate_ids = id_counts[id_counts > 1].index.tolist()

    return DuplicateCheckResult(
        n_rows=len(df),
        n_fully_duplicate_rows=n_fully_duplicate,
        n_duplicate_ids=len(duplicate_ids),
        duplicate_id_values=duplicate_ids,
    )


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully duplicated rows (exact row duplicates only)."""
    return df.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Patient-level split
# ---------------------------------------------------------------------------

def patient_level_split(
    df: pd.DataFrame,
    id_col: str = DEFAULT_SPLIT_ID_COLUMN,
    target_col: str = DEFAULT_TARGET_COLUMN,
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/validation/test by unique patient id, stratified by
    outcome, so no patient's rows can appear in more than one split.

    Even when the cohort is already one-row-per-patient (as produced by
    sql/cohort.sql's first-ICU-stay-per-patient logic), splitting on the id
    column rather than the row index is the correct, robust approach: it
    keeps working unchanged if the cohort definition ever changes to allow
    multiple rows per patient.
    """
    if abs((train_size + val_size + test_size) - 1.0) > 1e-9:
        raise ValueError(
            f"train_size + val_size + test_size must sum to 1.0, got "
            f"{train_size} + {val_size} + {test_size} = {train_size + val_size + test_size}"
        )

    unique_patients = df[[id_col, target_col]].drop_duplicates(subset=id_col)

    train_ids, temp_ids = train_test_split(
        unique_patients[id_col],
        train_size=train_size,
        stratify=unique_patients[target_col],
        random_state=seed,
    )

    temp_df = unique_patients[unique_patients[id_col].isin(temp_ids)]
    relative_val_size = val_size / (val_size + test_size)
    val_ids, test_ids = train_test_split(
        temp_df[id_col],
        train_size=relative_val_size,
        stratify=temp_df[target_col],
        random_state=seed,
    )

    train_df = df[df[id_col].isin(train_ids)].reset_index(drop=True)
    val_df = df[df[id_col].isin(val_ids)].reset_index(drop=True)
    test_df = df[df[id_col].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df


def check_patient_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str = DEFAULT_SPLIT_ID_COLUMN,
) -> None:
    """Raise AssertionError if any patient id appears in more than one split."""
    train_ids = set(train_df[id_col])
    val_ids = set(val_df[id_col])
    test_ids = set(test_df[id_col])

    train_val = train_ids & val_ids
    train_test = train_ids & test_ids
    val_test = val_ids & test_ids

    assert not train_val, f"{len(train_val)} patient id(s) leak between train and val: {list(train_val)[:10]}"
    assert not train_test, f"{len(train_test)} patient id(s) leak between train and test: {list(train_test)[:10]}"
    assert not val_test, f"{len(val_test)} patient id(s) leak between val and test: {list(val_test)[:10]}"


# ---------------------------------------------------------------------------
# Cohort flow / class distribution
# ---------------------------------------------------------------------------

def load_and_split_cohort(
    config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Single source of truth for reproducing the exact patient-level
    70/10/20 split: load -> drop exact-duplicate rows -> split -> verify no
    patient overlap. Given the same cohort_features.parquet and the same
    configs/config.yaml -> preprocessing.random_seed, this is deterministic,
    so any script/notebook calling this reproduces the identical split
    without needing to persist row indices anywhere.
    """
    config = config or load_config()
    df = load_cohort_features(config=config)
    dedup_df = drop_duplicate_rows(df)

    id_col = config["preprocessing"]["id_column"]
    target_col = config["preprocessing"]["target_column"]

    train_df, val_df, test_df = patient_level_split(
        dedup_df,
        id_col=id_col,
        target_col=target_col,
        train_size=config["preprocessing"]["train_size"],
        val_size=config["preprocessing"]["val_size"],
        test_size=config["preprocessing"]["test_size"],
        seed=config["preprocessing"]["random_seed"],
    )
    check_patient_overlap(train_df, val_df, test_df, id_col=id_col)

    return train_df, val_df, test_df


def compute_cohort_flow(
    raw_df: pd.DataFrame,
    dedup_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str = DEFAULT_SPLIT_ID_COLUMN,
) -> pd.DataFrame:
    """Build a CONSORT-style cohort flow table: row/patient counts at each
    preprocessing stage, for reporting and sanity-checking.
    """
    stages = [
        ("loaded (raw cohort_features.parquet)", raw_df),
        ("after duplicate-row removal", dedup_df),
        ("train split", train_df),
        ("validation split", val_df),
        ("test split", test_df),
    ]
    rows = [
        {"stage": name, "n_rows": len(d), "n_unique_patients": d[id_col].nunique()}
        for name, d in stages
    ]
    return pd.DataFrame(rows)


def class_distribution(df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COLUMN) -> pd.DataFrame:
    """Return outcome counts and proportions."""
    counts = df[target_col].value_counts().sort_index()
    props = df[target_col].value_counts(normalize=True).sort_index()
    return pd.DataFrame({"count": counts, "proportion": props})


# ---------------------------------------------------------------------------
# Identifier handling
# ---------------------------------------------------------------------------

def drop_identifier_columns(df: pd.DataFrame, id_cols: list[str] = IDENTIFIER_COLUMNS) -> pd.DataFrame:
    """Drop identifier columns before building a feature matrix. Identifiers
    are used only for splitting/auditing (above) and must never reach the
    model as features.
    """
    present = [c for c in id_cols if c in df.columns]
    return df.drop(columns=present)


# ---------------------------------------------------------------------------
# Column typing
# ---------------------------------------------------------------------------

def identify_feature_types(
    df: pd.DataFrame,
    exclude_cols: list[str],
) -> tuple[list[str], list[str]]:
    """Split feature columns into numeric and categorical, excluding
    identifiers/target/timestamps.
    """
    candidate_cols = [c for c in df.columns if c not in exclude_cols]

    numeric_cols = [
        c for c in candidate_cols
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    categorical_cols = [
        c for c in candidate_cols
        if c not in numeric_cols
    ]

    return numeric_cols, categorical_cols


# ---------------------------------------------------------------------------
# Fitted preprocessing pipeline
# ---------------------------------------------------------------------------

class ClinicalPreprocessor:
    """Fit-once-on-train, transform-everywhere preprocessing pipeline:
    missingness indicators -> median imputation (numeric) -> standard
    scaling (numeric) -> most-frequent imputation + one-hot encoding
    (categorical).

    All statistics (medians, scale/mean, category vocabulary) are learned
    exclusively from the data passed to `fit()`. Call `fit()` on the
    training split only, then `transform()` on train/val/test.
    """

    def __init__(self, numeric_cols: list[str], categorical_cols: list[str]):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.missing_indicator_cols_: list[str] = []
        self.numeric_imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None
        self.categorical_imputer_: SimpleImputer | None = None
        self.onehot_encoder_: OneHotEncoder | None = None
        self._is_fitted = False

    @staticmethod
    def _normalize_categorical_missing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Return a copy of df[cols] with every missing representation
        (Python None, pandas.NA, np.nan) normalized to np.nan.

        Bug this guards against: sklearn's SimpleImputer(missing_values=np.nan)
        (the default) does not reliably catch a literal Python `None` stored
        in an object-dtype column — it only replaces np.nan — so a raw
        `None` category can silently survive imputation and one-hot
        encoding as its own spurious level (e.g. "gender_None"). Using
        pandas' `.isna()` mask (which does catch None/NA/NaN uniformly)
        to force everything to np.nan before handing off to SimpleImputer
        closes that gap.
        """
        out = df[cols].copy()
        for c in cols:
            out[c] = out[c].where(out[c].notna(), np.nan)
        return out

    def fit(self, df_train: pd.DataFrame) -> "ClinicalPreprocessor":
        # Missingness indicators: only for columns that actually have
        # missing values in the training data (avoids all-zero constant
        # columns for fields that happen to be complete in train).
        all_cols = self.numeric_cols + self.categorical_cols
        self.missing_indicator_cols_ = [c for c in all_cols if df_train[c].isna().any()]

        if self.numeric_cols:
            self.numeric_imputer_ = SimpleImputer(strategy="median")
            numeric_imputed_train = pd.DataFrame(
                self.numeric_imputer_.fit_transform(df_train[self.numeric_cols]),
                columns=self.numeric_cols,
                index=df_train.index,
            )
            self.scaler_ = StandardScaler()
            self.scaler_.fit(numeric_imputed_train)

        if self.categorical_cols:
            categorical_train = self._normalize_categorical_missing(df_train, self.categorical_cols)
            self.categorical_imputer_ = SimpleImputer(strategy="most_frequent")
            categorical_imputed_train = pd.DataFrame(
                self.categorical_imputer_.fit_transform(categorical_train),
                columns=self.categorical_cols,
                index=df_train.index,
            )
            self.onehot_encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.onehot_encoder_.fit(categorical_imputed_train)

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("ClinicalPreprocessor.fit() must be called before transform().")

        parts: list[pd.DataFrame] = []

        if self.missing_indicator_cols_:
            missing_df = df[self.missing_indicator_cols_].isna().astype(int)
            missing_df.columns = [f"{c}_missing" for c in self.missing_indicator_cols_]
            parts.append(missing_df.reset_index(drop=True))

        if self.numeric_cols:
            numeric_imputed = pd.DataFrame(
                self.numeric_imputer_.transform(df[self.numeric_cols]),
                columns=self.numeric_cols,
                index=df.index,
            )
            numeric_scaled = pd.DataFrame(
                self.scaler_.transform(numeric_imputed),
                columns=self.numeric_cols,
                index=df.index,
            )
            parts.append(numeric_scaled.reset_index(drop=True))

        if self.categorical_cols:
            categorical_normalized = self._normalize_categorical_missing(df, self.categorical_cols)
            categorical_imputed = pd.DataFrame(
                self.categorical_imputer_.transform(categorical_normalized),
                columns=self.categorical_cols,
                index=df.index,
            )
            encoded = self.onehot_encoder_.transform(categorical_imputed)
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.onehot_encoder_.get_feature_names_out(self.categorical_cols),
                index=df.index,
            )
            parts.append(encoded_df.reset_index(drop=True))

        if not parts:
            warnings.warn("ClinicalPreprocessor: no numeric or categorical columns configured.")
            return pd.DataFrame(index=df.index)

        return pd.concat(parts, axis=1)

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df_train).transform(df_train)


# ---------------------------------------------------------------------------
# Table 1 (cohort characteristics)
# ---------------------------------------------------------------------------

def _summarize_numeric(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) == 0:
        return "N/A"
    median = s.median()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return f"{median:.1f} [{q1:.1f}, {q3:.1f}]"


def _summarize_categorical(series: pd.Series) -> dict[str, str]:
    s = series.dropna()
    n = len(s)
    counts = s.value_counts()
    return {
        str(level): f"{count} ({100 * count / n:.1f}%)" if n > 0 else "N/A"
        for level, count in counts.items()
    }


def generate_table1(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    group_col: str | None = None,
) -> pd.DataFrame:
    """Generate a standard "Table 1" cohort-characteristics table.

    Numeric variables are summarized as median [IQR]; categorical variables
    as n (%) per level. If `group_col` is given (e.g. a 'split' column with
    values train/val/test), an "Overall" column plus one column per group
    value are produced; otherwise a single "Overall" column.
    """
    group_values = ["Overall"]
    if group_col is not None:
        group_values += sorted(df[group_col].dropna().unique().tolist())

    rows: list[dict[str, Any]] = []

    for col in numeric_cols:
        row: dict[str, Any] = {"variable": col}
        for gv in group_values:
            subset = df if gv == "Overall" else df[df[group_col] == gv]
            row[gv] = _summarize_numeric(subset[col])
        rows.append(row)

    for col in categorical_cols:
        overall_levels = _summarize_categorical(df[col])
        for level in overall_levels:
            row = {"variable": f"{col} = {level}"}
            for gv in group_values:
                subset = df if gv == "Overall" else df[df[group_col] == gv]
                level_summary = _summarize_categorical(subset[col])
                row[gv] = level_summary.get(level, "0 (0.0%)")
            rows.append(row)

    return pd.DataFrame(rows).set_index("variable")


if __name__ == "__main__":
    print(
        "This module expects a locally supplied cohort/features parquet "
        "file — see load_cohort_features() and configs/config.yaml "
        "(preprocessing.cohort_features_path). Run notebooks/"
        "01_cohort_audit.ipynb for the full audit + preprocessing workflow."
    )
