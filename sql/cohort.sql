-- =============================================================================
-- cohort.sql
-- First-24-hours ICU in-hospital mortality cohort (MIMIC-IV)
--
-- Purpose: define the study cohort (one row per qualifying ICU stay) and the
-- prediction target (in-hospital mortality), using ONLY information that is
-- knowable at the end of the first 24 hours of an ICU stay, plus static
-- demographics/administrative fields fixed at admission time.
--
-- Data source: local MIMIC-IV extract under data/raw/mimiciv (per project
-- config; not committed to this repository).
--
-- CHECK_SCHEMA: This query assumes the standard MIMIC-IV v2.x table/column
-- names below (mimiciv_hosp.*, mimiciv_icu.*). Before running, confirm:
--   - Your local database/catalog actually exposes these schema names
--     (some local builds use "mimic_hosp"/"mimic_icu", "hosp"/"icu", or
--     a flat schema with no prefix — adjust the FROM/JOIN clauses below).
--   - Column names and types match your loaded MIMIC-IV version
--     (v2.0 vs v2.2 have minor differences, e.g. "race" vs "ethnicity").
-- =============================================================================

WITH first_icu_stay AS (
    -- CHECK_SCHEMA: verify table mimiciv_icu.icustays exists with columns
    -- subject_id, hadm_id, stay_id, intime, outtime, los, first_careunit.
    SELECT
        icu.subject_id,
        icu.hadm_id,
        icu.stay_id,
        icu.intime,
        icu.outtime,
        icu.los,
        icu.first_careunit,
        ROW_NUMBER() OVER (
            PARTITION BY icu.subject_id
            ORDER BY icu.intime ASC
        ) AS icu_stay_rank
    FROM mimiciv_icu.icustays AS icu
),

eligible_stays AS (
    -- Study inclusion: first ICU stay per patient only, to avoid
    -- within-patient correlation / leakage across repeat admissions.
    -- Require ICU LOS >= 24 hours so a full first-24h feature window exists
    -- (stays shorter than 24h are excluded rather than truncated, so we
    -- never use post-window information).
    SELECT *
    FROM first_icu_stay
    WHERE icu_stay_rank = 1
      AND los >= 1.0  -- los is in days in mimiciv_icu.icustays; 1.0 day = 24h
),

admission_info AS (
    -- CHECK_SCHEMA: verify table mimiciv_hosp.admissions exists with columns
    -- subject_id, hadm_id, admittime, dischtime, deathtime, admission_type,
    -- admission_location, insurance, marital_status, race (or ethnicity,
    -- depending on version), hospital_expire_flag.
    SELECT
        adm.subject_id,
        adm.hadm_id,
        adm.admittime,
        adm.dischtime,
        adm.deathtime,
        adm.admission_type,
        adm.admission_location,
        adm.insurance,
        adm.marital_status,
        adm.race,                 -- CHECK_SCHEMA: some versions call this "ethnicity"
        adm.hospital_expire_flag  -- 1 = died during this hospitalization, 0 = survived
    FROM mimiciv_hosp.admissions AS adm
),

patient_info AS (
    -- CHECK_SCHEMA: verify table mimiciv_hosp.patients exists with columns
    -- subject_id, gender, anchor_age, anchor_year, dod.
    -- anchor_age is the patient's age in anchor_year, used as an
    -- approximate age at admission (MIMIC-IV de-identifies exact DOB).
    SELECT
        pat.subject_id,
        pat.gender,
        pat.anchor_age,
        pat.anchor_year,
        pat.dod  -- date of death (any time, not just in-hospital); NOT a feature, audit-only
    FROM mimiciv_hosp.patients AS pat
)

SELECT
    -- Identifiers: retain for cohort construction, joins, and patient-level
    -- train/val/test splitting ONLY. Do not pass subject_id / hadm_id /
    -- stay_id to the model as features (see src/data_extract.py and
    -- docs/data_dictionary.md for the enforced exclusion list).
    es.subject_id,
    es.hadm_id,
    es.stay_id,

    -- Timing anchor: used to define the first-24h feature window in
    -- features.sql. Known at t=0 (ICU admission), so this alone is safe.
    --
    -- LEAKAGE FIX: icu_outtime and icu_los_days were previously selected
    -- here "for downstream use", but features.sql never actually needs
    -- them (it only uses icu_intime), and both directly encode ICU
    -- discharge timing / length of stay — neither is knowable during the
    -- first 24 hours, and LOS is a well-known mortality-correlated
    -- leakage feature in ICU outcome studies. They are deliberately NOT
    -- selected here. If a future change needs them for cohort auditing,
    -- add them to a separate audit-only query — never to this SELECT.
    es.intime  AS icu_intime,
    es.first_careunit,

    -- Demographics (static, known at/near admission — safe, no leakage)
    pi.gender,
    -- Approximate age at ICU admission. anchor_age is age in anchor_year;
    -- MIMIC-IV shifts admittime into the same de-identified anchor window,
    -- so anchor_age is used directly as age-at-admission proxy.
    -- CHECK_SCHEMA: confirm this approximation matches your cohort's needs;
    -- refine using admittime - anchor_year offset if higher precision is required.
    pi.anchor_age AS age_at_admission,

    ai.admission_type,
    ai.admission_location,
    ai.insurance,
    ai.marital_status,
    ai.race,

    -- ---------------------------------------------------------------
    -- Target variable: in-hospital mortality.
    -- Defined ONLY from admissions.hospital_expire_flag, which reflects
    -- the outcome of the ENTIRE hospitalization (known only at discharge).
    -- This is the label, never a feature — it must not leak into any
    -- first-24h feature computed in features.sql.
    -- ---------------------------------------------------------------
    ai.hospital_expire_flag AS in_hospital_mortality

FROM eligible_stays AS es
INNER JOIN admission_info AS ai
    ON es.subject_id = ai.subject_id
   AND es.hadm_id    = ai.hadm_id
INNER JOIN patient_info AS pi
    ON es.subject_id = pi.subject_id

-- Basic adult-cohort inclusion criterion. Adjust threshold per your
-- study's inclusion/exclusion criteria (documented in docs/data_dictionary.md).
WHERE pi.anchor_age >= 18

ORDER BY es.subject_id, es.hadm_id, es.stay_id;
