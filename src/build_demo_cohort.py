"""Build a first-24h ICU mortality cohort from the MIMIC-IV Clinical
Database DEMO (v2.2) — a small (~100 patient), publicly available,
de-identified CSV extract. This is the ONLY script in this project that
reads directly from raw CSVs on disk rather than from a database/catalog
(src/data_extract.py targets the full MIMIC-IV database instead).

CHECK_SCHEMA: this module assumes the standard MIMIC-IV Demo v2.2
distribution layout — files named patients/admissions/icustays/
chartevents/labevents/d_items/d_labitems, either flat or under hosp/ and
icu/ subfolders, as .csv or .csv.gz. `_find_csv()` searches recursively
under `demo_cohort.demo_data_dir` and is tolerant of both layouts and
compression, but you MUST verify the discovered files are the ones you
expect (it prints/logs every path it resolves) and that column names
inside them match what this script expects (see each `CHECK_SCHEMA`
comment below) — MIMIC-IV Demo versions can differ in column names/casing.

OUTCOME COLUMN NAME NOTE: per this task's explicit instruction, the binary
outcome column here is named "mortality" (1 = died in-hospital, 0 =
survived), sourced from admissions.hospital_expire_flag. This differs from
`configs/config.yaml -> preprocessing.target_column`
("in_hospital_mortality"), which the rest of the pipeline
(src/preprocess.py onward) expects. Before running notebooks 01+ against
this demo-built cohort, either:
  (a) update preprocessing.target_column to "mortality" in
      configs/config.yaml, or
  (b) rename the column after loading (df.rename(columns={"mortality":
      "in_hospital_mortality"})).
This script does not silently rename it for you.

LEAKAGE PREVENTION (same rules as sql/cohort.sql + sql/features.sql):
  - One row per patient (first ICU stay only).
  - ICU stays shorter than 24h are excluded, not truncated.
  - All vitals/labs are filtered to charttime < intime + 24h.
  - Only admissions.hospital_expire_flag (known at discharge) is used for
    the label; it is never referenced when building features.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.reproducibility import get_logger, log_run_metadata, set_global_seed

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

# CHECK_SCHEMA: candidate filename stems per logical table. Extend this
# list if your extracted demo uses different names/casing.
_FILENAME_CANDIDATES = {
    "patients": ["patients"],
    "admissions": ["admissions"],
    "icustays": ["icustays", "icu_stays"],
    "chartevents": ["chartevents"],
    "d_items": ["d_items"],
    "labevents": ["labevents"],
    "d_labitems": ["d_labitems"],
}


def _find_csv(data_dir: Path, table_name: str, logger) -> Path:
    """Recursively search data_dir for a .csv or .csv.gz file matching one
    of the known filename candidates for `table_name`. Raises a clear
    FileNotFoundError (listing what WAS found, if anything) rather than
    guessing.
    """
    candidates = _FILENAME_CANDIDATES[table_name]
    found: list[Path] = []
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        stem = path.name.lower()
        for ext in (".csv.gz", ".csv"):
            if stem.endswith(ext) and stem[: -len(ext)] in candidates:
                found.append(path)

    if not found:
        raise FileNotFoundError(
            f"Could not find a CSV for table '{table_name}' (looked for "
            f"{[c + '.csv[.gz]' for c in candidates]}) anywhere under '{data_dir}'. "
            "This project ships no MIMIC-IV data — extract the MIMIC-IV Clinical "
            f"Database Demo (v2.2) into {data_dir} first. If your files use a "
            "different name, add it to _FILENAME_CANDIDATES in src/build_demo_cohort.py."
        )
    if len(found) > 1:
        logger.warning(
            "Multiple candidate files found for table '%s': %s — using the first one (%s).",
            table_name, found, found[0],
        )

    logger.info("Resolved table '%s' -> %s", table_name, found[0])
    return found[0]


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix == ".gz" or path.name.endswith(".csv.gz"):
        with gzip.open(path, "rt") as f:
            return pd.read_csv(f, **kwargs)
    return pd.read_csv(path, **kwargs)


def load_demo_tables(config: dict[str, Any], logger) -> dict[str, pd.DataFrame]:
    """Load the seven required demo tables into memory. The demo is small
    (~100 patients), so no chunking/streaming is needed.
    """
    data_dir = Path(config["demo_cohort"]["demo_data_dir"])
    if not data_dir.exists():
        raise FileNotFoundError(
            f"'{data_dir}' does not exist. Extract the MIMIC-IV Clinical Database "
            f"Demo (v2.2) into this path first."
        )

    tables = {}
    for table_name in _FILENAME_CANDIDATES:
        path = _find_csv(data_dir, table_name, logger)
        # CHECK_SCHEMA: parse_dates columns assume standard MIMIC-IV column
        # names; adjust if your demo's columns are named differently.
        parse_dates = {
            "admissions": ["admittime", "dischtime", "deathtime"],
            "icustays": ["intime", "outtime"],
            "chartevents": ["charttime"],
            "labevents": ["charttime"],
        }.get(table_name)
        tables[table_name] = _read_csv(path, parse_dates=parse_dates)
        logger.info("Loaded %s: %d rows, %d columns", table_name, *tables[table_name].shape)

    return tables


# ---------------------------------------------------------------------------
# Cohort construction (mirrors sql/cohort.sql's logic, in pandas)
# ---------------------------------------------------------------------------

def build_cohort(tables: dict[str, pd.DataFrame], config: dict[str, Any], logger) -> pd.DataFrame:
    """One row per patient (first ICU stay only), demographics + target.
    CHECK_SCHEMA: assumes patients.subject_id/gender/anchor_age,
    admissions.subject_id/hadm_id/admittime/admission_type/
    hospital_expire_flag, icustays.subject_id/hadm_id/stay_id/intime/
    outtime/los/first_careunit.
    """
    dc_config = config["demo_cohort"]

    icustays = tables["icustays"].sort_values(["subject_id", "intime"])
    first_stay = icustays.groupby("subject_id", as_index=False).first()

    eligible = first_stay[first_stay["los"] >= dc_config["min_los_days"]].copy()
    logger.info(
        "ICU stays: %d total -> %d first-stay -> %d after >=%.0fh LOS filter",
        len(icustays), len(first_stay), len(eligible), dc_config["min_los_days"] * 24,
    )

    admissions_cols = ["subject_id", "hadm_id", "admission_type", "hospital_expire_flag"]
    admissions_cols = [c for c in admissions_cols if c in tables["admissions"].columns]
    cohort = eligible.merge(tables["admissions"][admissions_cols], on=["subject_id", "hadm_id"], how="left")

    patients_cols = ["subject_id", "gender", "anchor_age"]
    patients_cols = [c for c in patients_cols if c in tables["patients"].columns]
    cohort = cohort.merge(tables["patients"][patients_cols], on="subject_id", how="left")

    cohort = cohort[cohort["anchor_age"] >= dc_config["min_age"]].reset_index(drop=True)

    cohort = cohort.rename(columns={"anchor_age": "age_at_admission"})
    cohort[dc_config["target_column"]] = cohort["hospital_expire_flag"].astype("Int64")

    keep_cols = [
        "subject_id", "hadm_id", "stay_id", "intime", "outtime", "los",
        "first_careunit", "gender", "age_at_admission", "admission_type",
        dc_config["target_column"],
    ]
    keep_cols = [c for c in keep_cols if c in cohort.columns]
    cohort = cohort[keep_cols]

    logger.info(
        "Final cohort: %d patients, %d deaths (%.1f%%)",
        len(cohort), int(cohort[dc_config["target_column"]].sum()),
        100 * cohort[dc_config["target_column"]].mean(),
    )
    return cohort


# ---------------------------------------------------------------------------
# First-24h feature extraction (mirrors sql/features.sql's logic, in pandas)
# ---------------------------------------------------------------------------

# CHECK_SCHEMA: label-matching patterns for chartevents.itemid via
# d_items.label. Verify against your demo's actual d_items table — the
# demo's item catalog is a subset of the full MIMIC-IV catalog and may not
# contain every label pattern below.
_VITAL_LABEL_PATTERNS = {
    "heart_rate": ["heart rate"],
    "sbp": ["non invasive blood pressure systolic", "arterial blood pressure systolic"],
    "dbp": ["non invasive blood pressure diastolic", "arterial blood pressure diastolic"],
    "resp_rate": ["respiratory rate"],
    "spo2": ["o2 saturation", "spo2"],
    "temp": ["temperature f"],
    "gcs_eye": ["gcs - eye opening", "gcs eye"],
    "gcs_verbal": ["gcs - verbal response", "gcs verbal"],
    "gcs_motor": ["gcs - motor response", "gcs motor"],
}

# CHECK_SCHEMA: label-matching patterns for labevents.itemid via d_labitems.label.
_LAB_LABEL_PATTERNS = {
    "creatinine": ["creatinine"],
    "sodium": ["sodium"],
    "potassium": ["potassium"],
    "bicarbonate": ["bicarbonate"],
    "wbc": ["white blood cell", "wbc"],
    "glucose": ["glucose"],
    "bun": ["urea nitrogen", "bun"],
    "lactate": ["lactate"],
}


def _match_itemids(d_items: pd.DataFrame, label_col: str, patterns: dict[str, list[str]]) -> dict[str, list[int]]:
    labels_lower = d_items[label_col].str.lower()
    matched: dict[str, list[int]] = {}
    for concept, keywords in patterns.items():
        mask = pd.Series(False, index=d_items.index)
        for kw in keywords:
            mask |= labels_lower.str.contains(kw, na=False, regex=False)
        matched[concept] = d_items.loc[mask, "itemid"].tolist()
    return matched


def build_first24h_features(
    cohort: pd.DataFrame, tables: dict[str, pd.DataFrame], logger
) -> pd.DataFrame:
    """First-24h vital and lab summary features, one row per patient,
    strictly filtered to charttime < intime + 24h (see module docstring's
    leakage-prevention section).
    """
    window = cohort[["subject_id", "hadm_id", "stay_id", "intime"]].copy()
    window["window_end"] = window["intime"] + pd.Timedelta(hours=24)

    # --- Vitals from chartevents ---
    vital_itemids = _match_itemids(tables["d_items"], "label", _VITAL_LABEL_PATTERNS)
    for concept, ids in vital_itemids.items():
        logger.info("Vital concept '%s' matched %d itemid(s)", concept, len(ids))

    chartevents = tables["chartevents"].merge(window, on=["subject_id", "hadm_id", "stay_id"], how="inner")
    chartevents = chartevents[
        (chartevents["charttime"] >= chartevents["intime"]) & (chartevents["charttime"] < chartevents["window_end"])
    ]
    if "valuenum" in chartevents.columns:
        chartevents = chartevents[chartevents["valuenum"].notna()]

    vitals_summary = window[["subject_id"]].copy()
    for concept, ids in vital_itemids.items():
        concept_rows = chartevents[chartevents["itemid"].isin(ids)]
        agg = concept_rows.groupby("subject_id")["valuenum"].mean()
        vitals_summary[f"{concept}_mean"] = vitals_summary["subject_id"].map(agg)

    # Worst (lowest) GCS total in the first 24h = sum of component minima.
    gcs_components = {}
    for comp in ["gcs_eye", "gcs_verbal", "gcs_motor"]:
        ids = vital_itemids.get(comp, [])
        concept_rows = chartevents[chartevents["itemid"].isin(ids)]
        gcs_components[comp] = concept_rows.groupby("subject_id")["valuenum"].min()
    gcs_total_min = sum(gcs_components.values()) if gcs_components else pd.Series(dtype=float)
    vitals_summary["gcs_total_min"] = vitals_summary["subject_id"].map(gcs_total_min)
    vitals_summary = vitals_summary.drop(
        columns=[c for c in ["gcs_eye_mean", "gcs_verbal_mean", "gcs_motor_mean"] if c in vitals_summary.columns]
    )

    # --- Labs from labevents ---
    lab_itemids = _match_itemids(tables["d_labitems"], "label", _LAB_LABEL_PATTERNS)
    for concept, ids in lab_itemids.items():
        logger.info("Lab concept '%s' matched %d itemid(s)", concept, len(ids))

    labevents = tables["labevents"].merge(
        window[["subject_id", "hadm_id", "intime", "window_end"]], on=["subject_id", "hadm_id"], how="inner"
    )
    labevents = labevents[
        (labevents["charttime"] >= labevents["intime"]) & (labevents["charttime"] < labevents["window_end"])
    ]
    if "valuenum" in labevents.columns:
        labevents = labevents[labevents["valuenum"].notna()]

    labs_summary = window[["subject_id"]].copy()
    _lab_agg = {
        "creatinine": "max", "sodium": "min", "potassium": "max",
        "bicarbonate": "min", "wbc": "max", "glucose": "max",
        "bun": "max", "lactate": "max",
    }
    for concept, ids in lab_itemids.items():
        concept_rows = labevents[labevents["itemid"].isin(ids)]
        agg_fn = _lab_agg.get(concept, "max")
        agg = concept_rows.groupby("subject_id")["valuenum"].agg(agg_fn)
        labs_summary[f"{concept}_{agg_fn}"] = labs_summary["subject_id"].map(agg)

    features = vitals_summary.merge(labs_summary, on="subject_id", how="left")
    return features


# ---------------------------------------------------------------------------
# Feature selection (10-15 numeric features per task instructions)
# ---------------------------------------------------------------------------

SELECTED_NUMERIC_FEATURES = [
    "age_at_admission",
    "heart_rate_mean",
    "sbp_mean",
    "dbp_mean",
    "resp_rate_mean",
    "spo2_mean",
    "temp_mean",
    "gcs_total_min",
    "creatinine_max",
    "sodium_min",
    "potassium_max",
    "bicarbonate_min",
    "wbc_max",
    "glucose_max",
    "lactate_max",
]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_demo_cohort(config: dict[str, Any] | None = None) -> pd.DataFrame:
    config = config or load_config()
    seed = config["project"]["random_seed"]
    set_global_seed(seed)
    logger = get_logger(log_file=config["logging"]["log_file"])
    logger.info("=== build_demo_cohort.py started (seed=%d) ===", seed)

    tables = load_demo_tables(config, logger)
    cohort = build_cohort(tables, config, logger)
    features = build_first24h_features(cohort, tables, logger)

    dc_config = config["demo_cohort"]
    final_df = cohort.merge(features, on="subject_id", how="left")

    available_features = [c for c in SELECTED_NUMERIC_FEATURES if c in final_df.columns]
    missing_features = [c for c in SELECTED_NUMERIC_FEATURES if c not in final_df.columns]
    if missing_features:
        logger.warning(
            "%d of %d requested features could not be built from this demo extract "
            "(likely absent from the demo's d_items/d_labitems catalog): %s",
            len(missing_features), len(SELECTED_NUMERIC_FEATURES), missing_features,
        )

    id_cols = ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los", "first_careunit"]
    id_cols = [c for c in id_cols if c in final_df.columns]
    demographic_cols = [c for c in ["gender", "admission_type"] if c in final_df.columns]
    output_cols = id_cols + demographic_cols + available_features + [dc_config["target_column"]]
    output_df = final_df[output_cols]

    output_path = Path(dc_config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    logger.info(
        "Cohort written to %s: %d rows, %d columns (%d numeric features)",
        output_path, output_df.shape[0], output_df.shape[1], len(available_features),
    )

    log_run_metadata(
        seed=seed,
        extra={
            "step": "build_demo_cohort",
            "source": "MIMIC-IV Clinical Database Demo v2.2",
            "n_patients": len(output_df),
            "n_deaths": int(output_df[dc_config["target_column"]].sum()),
            "features_built": available_features,
            "features_missing": missing_features,
            "target_column": dc_config["target_column"],
        },
    )
    logger.info("=== build_demo_cohort.py finished ===")

    return output_df


if __name__ == "__main__":
    build_demo_cohort()
