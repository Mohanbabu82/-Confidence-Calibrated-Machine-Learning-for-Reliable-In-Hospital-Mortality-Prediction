-- =============================================================================
-- features.sql
-- First-24-hours feature extraction for ICU in-hospital mortality prediction
-- (MIMIC-IV): vital-sign summaries and laboratory summaries.
--
-- LEAKAGE PREVENTION RULES ENFORCED IN THIS FILE:
--   1. Every event is filtered to charttime BETWEEN icu_intime AND
--      icu_intime + INTERVAL '24' HOUR. Nothing after the 24h mark is used.
--   2. Only aggregate summary statistics (min/max/mean/first/last count) are
--      computed per stay — no raw timestamped values are exposed as features,
--      which prevents indirectly encoding length-of-stay or discharge timing.
--   3. This file does NOT reference admissions.hospital_expire_flag,
--      dischtime, deathtime, or dod anywhere. The label lives only in
--      cohort.sql / the final joined table, never in feature computation.
--   4. Identifiers (subject_id, hadm_id, stay_id) are carried through only
--      as join keys, not as model features (enforced again in
--      src/data_extract.py's feature-column allowlist).
--
-- CHECK_SCHEMA: This file assumes tables mimiciv_icu.chartevents,
-- mimiciv_icu.d_items, mimiciv_hosp.labevents, mimiciv_hosp.d_labitems with
-- the standard MIMIC-IV v2.x column names used below. Also verify:
--   - Vital-sign itemids: this query resolves itemids dynamically via
--     d_items.label pattern matching (ILIKE) rather than hardcoding numeric
--     itemids, because itemid values differ between MIMIC-III and MIMIC-IV
--     and across chartevents source systems (e.g. multiple itemids can map
--     to "Heart Rate" from different monitors). YOU MUST manually inspect
--     mimiciv_icu.d_items for your loaded dataset and confirm the label
--     patterns below actually match the intended concept, or replace them
--     with an explicit itemid allowlist (e.g. from the mimic-code
--     concepts/measurement/vitalsign.sql reference query) for precision.
--   - Lab itemids: same caveat applies to mimiciv_hosp.d_labitems.label.
--   - ILIKE / regex syntax is PostgreSQL-flavored; adjust for your engine
--     (e.g. DuckDB supports ILIKE natively; BigQuery needs LOWER()+LIKE
--     or REGEXP_CONTAINS).
-- =============================================================================

WITH cohort AS (
    -- Reuses the cohort definition; in production, materialize cohort.sql's
    -- output as a table/view first and reference it here instead of a CTE
    -- duplicate, to guarantee the exact same cohort is used everywhere.
    -- CHECK_SCHEMA: replace this stub with a reference to your materialized
    -- cohort table, e.g. FROM mimiciv_derived.study_cohort
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        icu_intime
    FROM cohort_table  -- CHECK_SCHEMA: placeholder name; point at materialized cohort.sql output
),

first_24h_window AS (
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        icu_intime,
        icu_intime + INTERVAL '24' HOUR AS window_end
    FROM cohort
),

-- -----------------------------------------------------------------------
-- Vital signs (mimiciv_icu.chartevents), first 24h only
-- -----------------------------------------------------------------------
vitals_raw AS (
    SELECT
        w.subject_id,
        w.hadm_id,
        w.stay_id,
        ce.charttime,
        ce.valuenum,
        d.label AS item_label,
        CASE
            WHEN d.label ILIKE '%heart rate%'                      THEN 'heart_rate'
            WHEN d.label ILIKE '%non invasive blood pressure sys%'
              OR d.label ILIKE '%arterial blood pressure sys%'     THEN 'sbp'
            WHEN d.label ILIKE '%non invasive blood pressure dia%'
              OR d.label ILIKE '%arterial blood pressure dia%'     THEN 'dbp'
            WHEN d.label ILIKE '%non invasive blood pressure mean%'
              OR d.label ILIKE '%arterial blood pressure mean%'    THEN 'map'
            WHEN d.label ILIKE '%respiratory rate%'                THEN 'resp_rate'
            WHEN d.label ILIKE '%temperature%f%'                   THEN 'temp_f'
            WHEN d.label ILIKE '%o2 saturation%'
              OR d.label ILIKE '%spo2%'                             THEN 'spo2'
            WHEN d.label ILIKE '%gcs - total%'
              OR d.label ILIKE '%gcs total%'                        THEN 'gcs_total'
            -- CHECK_SCHEMA: standard MIMIC-IV d_items does NOT include a
            -- single "GCS Total" chart item in most builds — GCS is
            -- recorded as three separate components (eye/verbal/motor),
            -- and total GCS is conventionally derived by summing them
            -- (see mimic-code's gcs.sql reference concept). The direct
            -- 'gcs_total' match above is kept as a fallback in case your
            -- build differs, but VERIFY against your actual d_items
            -- table which of these two paths (direct item vs. derived
            -- components) actually returns data.
            WHEN d.label ILIKE '%gcs%eye%'                           THEN 'gcs_eye'
            WHEN d.label ILIKE '%gcs%verbal%'                        THEN 'gcs_verbal'
            WHEN d.label ILIKE '%gcs%motor%'                         THEN 'gcs_motor'
            ELSE NULL
        END AS vital_name
    FROM first_24h_window AS w
    -- CHECK_SCHEMA: verify mimiciv_icu.chartevents has columns subject_id,
    -- hadm_id, stay_id, charttime, itemid, valuenum, and that joining on
    -- stay_id is appropriate (chartevents in MIMIC-IV is ICU-stay scoped).
    INNER JOIN mimiciv_icu.chartevents AS ce
        ON w.stay_id = ce.stay_id
       AND ce.charttime >= w.icu_intime
       AND ce.charttime <  w.window_end   -- strict "<" enforces the 24h cutoff; no leakage past window
    -- CHECK_SCHEMA: verify mimiciv_icu.d_items has columns itemid, label.
    INNER JOIN mimiciv_icu.d_items AS d
        ON ce.itemid = d.itemid
    WHERE ce.valuenum IS NOT NULL
      AND ce.error IS DISTINCT FROM 1  -- CHECK_SCHEMA: confirm "error" column exists in your build; drops flagged bad measurements
),

vitals_summary AS (
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        MIN(CASE WHEN vital_name = 'heart_rate' THEN valuenum END) AS heart_rate_min,
        MAX(CASE WHEN vital_name = 'heart_rate' THEN valuenum END) AS heart_rate_max,
        AVG(CASE WHEN vital_name = 'heart_rate' THEN valuenum END) AS heart_rate_mean,

        MIN(CASE WHEN vital_name = 'sbp' THEN valuenum END) AS sbp_min,
        MAX(CASE WHEN vital_name = 'sbp' THEN valuenum END) AS sbp_max,
        AVG(CASE WHEN vital_name = 'sbp' THEN valuenum END) AS sbp_mean,

        MIN(CASE WHEN vital_name = 'dbp' THEN valuenum END) AS dbp_min,
        MAX(CASE WHEN vital_name = 'dbp' THEN valuenum END) AS dbp_max,
        AVG(CASE WHEN vital_name = 'dbp' THEN valuenum END) AS dbp_mean,

        MIN(CASE WHEN vital_name = 'map' THEN valuenum END) AS map_min,
        MAX(CASE WHEN vital_name = 'map' THEN valuenum END) AS map_max,
        AVG(CASE WHEN vital_name = 'map' THEN valuenum END) AS map_mean,

        MIN(CASE WHEN vital_name = 'resp_rate' THEN valuenum END) AS resp_rate_min,
        MAX(CASE WHEN vital_name = 'resp_rate' THEN valuenum END) AS resp_rate_max,
        AVG(CASE WHEN vital_name = 'resp_rate' THEN valuenum END) AS resp_rate_mean,

        MIN(CASE WHEN vital_name = 'temp_f' THEN valuenum END) AS temp_f_min,
        MAX(CASE WHEN vital_name = 'temp_f' THEN valuenum END) AS temp_f_max,
        AVG(CASE WHEN vital_name = 'temp_f' THEN valuenum END) AS temp_f_mean,

        MIN(CASE WHEN vital_name = 'spo2' THEN valuenum END) AS spo2_min,
        MAX(CASE WHEN vital_name = 'spo2' THEN valuenum END) AS spo2_max,
        AVG(CASE WHEN vital_name = 'spo2' THEN valuenum END) AS spo2_mean,

        -- Worst (lowest) GCS total in the window: prefer a direct "GCS
        -- Total" item if present in this build, else fall back to the sum
        -- of each component's worst (lowest) value observed in the window
        -- — the standard clinical convention for "worst GCS in period".
        -- CHECK_SCHEMA: confirm which path is populated in your data.
        COALESCE(
            MIN(CASE WHEN vital_name = 'gcs_total' THEN valuenum END),
            MIN(CASE WHEN vital_name = 'gcs_eye'    THEN valuenum END)
              + MIN(CASE WHEN vital_name = 'gcs_verbal' THEN valuenum END)
              + MIN(CASE WHEN vital_name = 'gcs_motor'  THEN valuenum END)
        ) AS gcs_total_min,
        MAX(CASE WHEN vital_name = 'gcs_total' THEN valuenum END) AS gcs_total_max

    FROM vitals_raw
    WHERE vital_name IS NOT NULL
    GROUP BY subject_id, hadm_id, stay_id
),

-- -----------------------------------------------------------------------
-- Laboratory results (mimiciv_hosp.labevents), first 24h only
-- -----------------------------------------------------------------------
labs_raw AS (
    SELECT
        w.subject_id,
        w.hadm_id,
        w.stay_id,
        le.charttime,
        le.valuenum,
        d.label AS lab_label,
        CASE
            WHEN d.label ILIKE '%creatinine%'  AND d.label NOT ILIKE '%urine%' THEN 'creatinine'
            WHEN d.label ILIKE '%sodium%'      AND d.label NOT ILIKE '%urine%' THEN 'sodium'
            WHEN d.label ILIKE '%potassium%'   AND d.label NOT ILIKE '%urine%' THEN 'potassium'
            WHEN d.label ILIKE '%bicarbonate%'                                  THEN 'bicarbonate'
            WHEN d.label ILIKE '%hematocrit%'                                   THEN 'hematocrit'
            WHEN d.label ILIKE '%white blood cell%' OR d.label ILIKE '%wbc%'    THEN 'wbc'
            WHEN d.label ILIKE '%glucose%'     AND d.label NOT ILIKE '%urine%' THEN 'glucose'
            WHEN d.label ILIKE '%urea nitrogen%' OR d.label ILIKE '%bun%'       THEN 'bun'
            WHEN d.label ILIKE '%platelet%'                                     THEN 'platelet'
            WHEN d.label ILIKE '%lactate%'                                      THEN 'lactate'
            ELSE NULL
        END AS lab_name
    FROM first_24h_window AS w
    -- CHECK_SCHEMA: verify mimiciv_hosp.labevents has columns subject_id,
    -- hadm_id, charttime, itemid, valuenum. labevents is hadm_id-scoped,
    -- NOT stay_id-scoped, so we join on hadm_id (a patient may have labs
    -- from before/around ICU admission under the same hadm_id — the
    -- charttime window filter below is what actually enforces "first 24h
    -- of the ICU stay", not the join key).
    INNER JOIN mimiciv_hosp.labevents AS le
        ON w.subject_id = le.subject_id
       AND w.hadm_id    = le.hadm_id
       AND le.charttime >= w.icu_intime
       AND le.charttime <  w.window_end
    -- CHECK_SCHEMA: verify mimiciv_hosp.d_labitems has columns itemid, label.
    INNER JOIN mimiciv_hosp.d_labitems AS d
        ON le.itemid = d.itemid
    WHERE le.valuenum IS NOT NULL
),

labs_summary AS (
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        MIN(CASE WHEN lab_name = 'creatinine' THEN valuenum END) AS creatinine_min,
        MAX(CASE WHEN lab_name = 'creatinine' THEN valuenum END) AS creatinine_max,
        AVG(CASE WHEN lab_name = 'creatinine' THEN valuenum END) AS creatinine_mean,

        MIN(CASE WHEN lab_name = 'sodium' THEN valuenum END) AS sodium_min,
        MAX(CASE WHEN lab_name = 'sodium' THEN valuenum END) AS sodium_max,

        MIN(CASE WHEN lab_name = 'potassium' THEN valuenum END) AS potassium_min,
        MAX(CASE WHEN lab_name = 'potassium' THEN valuenum END) AS potassium_max,

        MIN(CASE WHEN lab_name = 'bicarbonate' THEN valuenum END) AS bicarbonate_min,
        MAX(CASE WHEN lab_name = 'bicarbonate' THEN valuenum END) AS bicarbonate_max,

        MIN(CASE WHEN lab_name = 'hematocrit' THEN valuenum END) AS hematocrit_min,
        MAX(CASE WHEN lab_name = 'hematocrit' THEN valuenum END) AS hematocrit_max,

        MIN(CASE WHEN lab_name = 'wbc' THEN valuenum END) AS wbc_min,
        MAX(CASE WHEN lab_name = 'wbc' THEN valuenum END) AS wbc_max,

        MIN(CASE WHEN lab_name = 'glucose' THEN valuenum END) AS glucose_min,
        MAX(CASE WHEN lab_name = 'glucose' THEN valuenum END) AS glucose_max,

        MIN(CASE WHEN lab_name = 'bun' THEN valuenum END) AS bun_min,
        MAX(CASE WHEN lab_name = 'bun' THEN valuenum END) AS bun_max,

        MIN(CASE WHEN lab_name = 'platelet' THEN valuenum END) AS platelet_min,
        MAX(CASE WHEN lab_name = 'platelet' THEN valuenum END) AS platelet_max,

        MAX(CASE WHEN lab_name = 'lactate' THEN valuenum END) AS lactate_max

    FROM labs_raw
    WHERE lab_name IS NOT NULL
    GROUP BY subject_id, hadm_id, stay_id
)

-- -----------------------------------------------------------------------
-- Final feature table: one row per ICU stay, join keys + feature columns.
-- Join this output to cohort.sql's output (on subject_id/hadm_id/stay_id)
-- to attach the in_hospital_mortality label for modeling.
-- -----------------------------------------------------------------------
SELECT
    c.subject_id,
    c.hadm_id,
    c.stay_id,

    v.heart_rate_min,  v.heart_rate_max,  v.heart_rate_mean,
    v.sbp_min,         v.sbp_max,         v.sbp_mean,
    v.dbp_min,         v.dbp_max,         v.dbp_mean,
    v.map_min,         v.map_max,         v.map_mean,
    v.resp_rate_min,   v.resp_rate_max,   v.resp_rate_mean,
    v.temp_f_min,      v.temp_f_max,      v.temp_f_mean,
    v.spo2_min,        v.spo2_max,        v.spo2_mean,
    v.gcs_total_min,   v.gcs_total_max,

    l.creatinine_min,   l.creatinine_max,   l.creatinine_mean,
    l.sodium_min,       l.sodium_max,
    l.potassium_min,    l.potassium_max,
    l.bicarbonate_min,  l.bicarbonate_max,
    l.hematocrit_min,   l.hematocrit_max,
    l.wbc_min,          l.wbc_max,
    l.glucose_min,      l.glucose_max,
    l.bun_min,          l.bun_max,
    l.platelet_min,     l.platelet_max,
    l.lactate_max

FROM cohort AS c
LEFT JOIN vitals_summary AS v
    ON c.subject_id = v.subject_id AND c.hadm_id = v.hadm_id AND c.stay_id = v.stay_id
LEFT JOIN labs_summary AS l
    ON c.subject_id = l.subject_id AND c.hadm_id = l.hadm_id AND c.stay_id = l.stay_id

ORDER BY c.subject_id, c.hadm_id, c.stay_id;
