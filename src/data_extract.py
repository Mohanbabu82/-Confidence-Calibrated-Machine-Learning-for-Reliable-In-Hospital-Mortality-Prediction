"""MIMIC-IV local extraction: run sql/cohort.sql and sql/features.sql against
a local database/catalog, join them, enforce the identifier/feature split,
and write a model-ready table to data/processed/.

Scope and safety constraints (do not relax without explicit sign-off):
  - Reads ONLY from the local path configured under mimic_iv.data_dir /
    mimic_iv.duckdb_path / mimic_iv.postgres (all local; no network calls,
    no external services, no uploads).
  - Never writes raw patient-level data outside data/processed/ (which is
    git-ignored — see .gitignore).
  - Identifier columns (subject_id, hadm_id, stay_id) are kept in a
    separate split-assignment table and explicitly dropped from the
    model-ready feature matrix — they must never reach a model as features.

CHECK_SCHEMA: This script assumes sql/cohort.sql and sql/features.sql
execute successfully against your local MIMIC-IV catalog under the
schema/table names documented there. Before running:
  1. Verify data/raw/mimiciv actually contains your authorized MIMIC-IV
     extract (this repository ships none).
  2. Verify configs/config.yaml -> mimic_iv.db_backend matches how your
     local data is stored (duckdb file/catalog of CSV+Parquet vs. a
     running Postgres server with mimiciv_hosp/mimiciv_icu schemas).
  3. Resolve every CHECK_SCHEMA comment in sql/cohort.sql and
     sql/features.sql against your actual table/column names first.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config
from src.reproducibility import get_logger, log_run_metadata, set_global_seed

# Columns that must NEVER be passed to a model as a feature. Enforced here
# as a hard allowlist-exclusion, independent of whatever the SQL happens to
# return, so a future change to features.sql cannot silently reintroduce an
# identifier as a feature.
IDENTIFIER_COLUMNS = ["subject_id", "hadm_id", "stay_id"]

# Columns that are valid to carry through the join (e.g. for window
# calculations or auditing) but must never reach the model-ready feature
# matrix, because they encode information not available during the first
# 24 hours (discharge/outcome-adjacent timing) or are not model inputs.
# icu_outtime / icu_los_days are deliberately excluded from cohort.sql's
# SELECT list already (see leakage-fix comment there); this list is a
# second, independent enforcement layer in case that ever regresses.
NON_FEATURE_TIMING_COLUMNS = ["icu_outtime", "icu_los_days"]

TARGET_COLUMN = "in_hospital_mortality"


def _read_sql_file(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _connect_duckdb(config: dict[str, Any]):
    """Open a local DuckDB connection over the configured local catalog/files.

    CHECK_SCHEMA: duckdb_path must point at a DuckDB database file (or
    directory of CSV/Parquet files registered as views) that already
    contains mimiciv_hosp.* / mimiciv_icu.* tables under those exact
    names, OR you must register them yourself before calling this
    function (e.g. via duckdb's read_csv_auto / read_parquet + CREATE VIEW).
    """
    import duckdb

    duckdb_path = Path(config["mimic_iv"]["duckdb_path"])
    if not duckdb_path.exists():
        raise FileNotFoundError(
            f"DuckDB catalog not found at '{duckdb_path}'. This project ships "
            "no MIMIC-IV data. Point configs/config.yaml -> mimic_iv.duckdb_path "
            "at your own local, authorized MIMIC-IV catalog (see CHECK_SCHEMA "
            "notes in sql/cohort.sql and sql/features.sql)."
        )
    return duckdb.connect(str(duckdb_path), read_only=True)


def _connect_postgres(config: dict[str, Any]):
    """Open a local Postgres connection using config values only (no network
    calls to anything other than the configured local/authorized host).
    """
    import psycopg2

    pg_cfg = config["mimic_iv"]["postgres"]
    return psycopg2.connect(
        host=pg_cfg["host"],
        port=pg_cfg["port"],
        dbname=pg_cfg["dbname"],
        user=pg_cfg["user"],
    )


def get_connection(config: dict[str, Any]):
    """Return a DB-API-compatible connection per configs/config.yaml -> mimic_iv.db_backend."""
    backend = config["mimic_iv"]["db_backend"]
    if backend == "duckdb":
        return _connect_duckdb(config)
    elif backend == "postgres":
        return _connect_postgres(config)
    else:
        raise ValueError(
            f"Unsupported mimic_iv.db_backend '{backend}'. Expected 'duckdb' or 'postgres'."
        )


def extract_cohort(conn, config: dict[str, Any]) -> pd.DataFrame:
    """Execute sql/cohort.sql and return the cohort as a DataFrame."""
    sql_text = _read_sql_file(config["mimic_iv"]["cohort_sql"])
    return pd.read_sql(sql_text, conn)


def extract_features(conn, config: dict[str, Any]) -> pd.DataFrame:
    """Execute sql/features.sql and return the feature table as a DataFrame.

    CHECK_SCHEMA: sql/features.sql references a placeholder `cohort_table`
    name for its cohort CTE. Depending on your backend, either:
      (a) materialize sql/cohort.sql's output as a real table/view first
          named per configs/config.yaml -> mimic_iv.cohort_table_name, or
      (b) edit sql/features.sql's `cohort` CTE to inline cohort.sql's
          logic directly.
    This function assumes (a): it materializes the cohort into a temp view
    before running features.sql.
    """
    cohort_df = extract_cohort(conn, config)
    table_name = config["mimic_iv"]["cohort_table_name"]

    try:
        import duckdb  # noqa: F401
        conn.register(table_name, cohort_df)
    except ImportError:
        raise RuntimeError(
            "Materializing the cohort as a temp view is implemented for the "
            "duckdb backend. For postgres, first CREATE TABLE/VIEW "
            f"{table_name} from sql/cohort.sql's output in your database, "
            "then rerun this function."
        )

    features_sql_text = _read_sql_file(config["mimic_iv"]["features_sql"])
    features_df = pd.read_sql(features_sql_text, conn)
    return cohort_df, features_df


def build_model_ready_table(
    cohort_df: pd.DataFrame, features_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join cohort (with label) and features on identifier columns, then
    split into (a) a split-assignment table (identifiers + target, for
    patient-level train/val/test splitting) and (b) a model-ready feature
    matrix with all identifier columns dropped.
    """
    joined = cohort_df.merge(features_df, on=IDENTIFIER_COLUMNS, how="left")

    missing_target = TARGET_COLUMN not in joined.columns
    if missing_target:
        raise ValueError(
            f"Expected target column '{TARGET_COLUMN}' from sql/cohort.sql "
            "was not found after the join. Check sql/cohort.sql's SELECT list."
        )

    split_assignment_df = joined[IDENTIFIER_COLUMNS + [TARGET_COLUMN]].copy()

    # Hard enforcement: identifiers and outcome-adjacent timing columns are
    # never part of the feature matrix. Uses errors='ignore' since these
    # columns are not expected to be present (cohort.sql no longer selects
    # them) — this is a defense-in-depth check, not the primary control.
    columns_to_drop = IDENTIFIER_COLUMNS + [
        c for c in NON_FEATURE_TIMING_COLUMNS if c in joined.columns
    ]
    model_ready_df = joined.drop(columns=columns_to_drop)

    return split_assignment_df, model_ready_df


def run_extraction(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """End-to-end extraction: connect, extract cohort+features, join, enforce
    identifier/feature separation, write to configured output path, and log
    run metadata for reproducibility.
    """
    config = config or load_config()
    set_global_seed(config["project"]["random_seed"])
    logger = get_logger(log_file=config["logging"]["log_file"])

    logger.info("Connecting to local MIMIC-IV catalog (backend=%s)", config["mimic_iv"]["db_backend"])
    conn = get_connection(config)

    try:
        logger.info("Extracting cohort via %s", config["mimic_iv"]["cohort_sql"])
        cohort_df, features_df = extract_features(conn, config)
        logger.info("Cohort rows: %d | Feature rows: %d", len(cohort_df), len(features_df))

        split_assignment_df, model_ready_df = build_model_ready_table(cohort_df, features_df)

        assert not any(col in model_ready_df.columns for col in IDENTIFIER_COLUMNS), (
            "Identifier column leaked into model-ready feature matrix — aborting."
        )
        assert not any(col in model_ready_df.columns for col in NON_FEATURE_TIMING_COLUMNS), (
            "Outcome-adjacent timing column (icu_outtime/icu_los_days) leaked "
            "into model-ready feature matrix — aborting."
        )

        output_path = Path(config["mimic_iv"]["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model_ready_df.to_parquet(output_path, index=False)
        logger.info("Model-ready feature table written to %s (%d rows, %d columns)",
                     output_path, model_ready_df.shape[0], model_ready_df.shape[1])

        split_path = output_path.with_name(output_path.stem + "_split_ids.parquet")
        split_assignment_df.to_parquet(split_path, index=False)
        logger.info("Split-assignment table (identifiers + target) written to %s", split_path)

        log_run_metadata(
            output_path="logs/run_metadata.json",
            seed=config["project"]["random_seed"],
            extra={
                "step": "data_extract",
                "n_rows": int(model_ready_df.shape[0]),
                "n_features": int(model_ready_df.shape[1]),
                "target_column": TARGET_COLUMN,
                "identifier_columns_excluded": IDENTIFIER_COLUMNS,
            },
        )

        return model_ready_df
    finally:
        conn.close()


if __name__ == "__main__":
    run_extraction()
