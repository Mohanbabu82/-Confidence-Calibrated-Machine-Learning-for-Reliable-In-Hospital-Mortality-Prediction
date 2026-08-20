# Data Dictionary — First-24h ICU In-Hospital Mortality (MIMIC-IV)

> **CHECK_SCHEMA**: This dictionary documents the *intended* fields based on
> standard MIMIC-IV v2.x table structure. Before running any extraction,
> verify every table/column name against your actual local
> `data/raw/mimiciv` load (schema names, column names, and value encodings
> can differ by version/export). Fields marked `CHECK_SCHEMA` below require
> explicit manual confirmation.

## Cohort Definition (`sql/cohort.sql`)

| Rule | Value |
|---|---|
| Population | Adult (age ≥ 18) ICU patients |
| Stay selection | First ICU stay per patient (`subject_id`) only |
| Minimum stay length | ≥ 24 hours (`icustays.los >= 1.0`) — stays shorter than 24h are **excluded**, never truncated |
| Unit of analysis | One row per ICU stay (`stay_id`) |

**Rationale for "first stay only":** using every ICU stay per patient would
violate the i.i.d. assumption used by standard train/test splitting and
risks leaking patient-specific signal across splits. Restricting to the
first stay keeps one independent observation per patient.

**Rationale for "≥ 24h minimum":** the study defines features over a fixed
first-24-hour window. Stays shorter than 24h cannot have that window fully
observed without incorporating information from close to (or after)
discharge/death, which would leak outcome-adjacent signal into features.

## Identifiers (join keys only — NEVER model features)

| Field | Table | Purpose |
|---|---|---|
| `subject_id` | `mimiciv_hosp.patients` | Patient-level identifier. **Use only** for train/val/test split assignment (ensures no patient appears in more than one split) and for joins. |
| `hadm_id` | `mimiciv_hosp.admissions` | Hospital-admission identifier. Join key only. |
| `stay_id` | `mimiciv_icu.icustays` | ICU-stay identifier. Join key only. |

`src/data_extract.py` enforces this via an explicit `IDENTIFIER_COLUMNS`
constant that is dropped from the model-ready feature matrix and only
retained in a separate split-assignment table.

## Demographics (static, admission-time — no leakage)

| Feature | Source | Notes |
|---|---|---|
| `gender` | `mimiciv_hosp.patients.gender` | CHECK_SCHEMA: confirm encoding (`'M'`/`'F'`). |
| `age_at_admission` | `mimiciv_hosp.patients.anchor_age` | Approximate age at ICU admission via MIMIC-IV's de-identified anchor-year mechanism. CHECK_SCHEMA: confirm this approximation is acceptable for your study; MIMIC-IV does not expose exact DOB. |
| `admission_type` | `mimiciv_hosp.admissions.admission_type` | e.g. EMERGENCY, ELECTIVE, URGENT. CHECK_SCHEMA: confirm categories present in your load. |
| `admission_location` | `mimiciv_hosp.admissions.admission_location` | Where the patient was admitted from. |
| `insurance` | `mimiciv_hosp.admissions.insurance` | Administrative field; consider fairness/bias implications before use as a feature. |
| `marital_status` | `mimiciv_hosp.admissions.marital_status` | May contain nulls. |
| `race` | `mimiciv_hosp.admissions.race` | CHECK_SCHEMA: some MIMIC-IV versions name this column `ethnicity`. Sensitive attribute — see fairness note below. |
| `first_careunit` | `mimiciv_icu.icustays.first_careunit` | ICU type at admission (e.g. MICU, SICU). Known at t=0, no leakage. |

**Fairness note:** `race`/`ethnicity` and `insurance` are administrative/
demographic fields, not physiological signal. Including them as raw model
features can encode and amplify healthcare disparities. Decide deliberately
whether to include them as features, use them only for post-hoc fairness
auditing (recommended default), or exclude entirely — document the choice
in `docs/methodology.md` before modeling.

## First-24h Vital Signs (`sql/features.sql`, from `mimiciv_icu.chartevents`)

All aggregated as `min` / `max` / `mean` over `[icu_intime, icu_intime + 24h)`
unless noted. Values are matched from `mimiciv_icu.d_items.label` via
pattern matching — **CHECK_SCHEMA: manually verify these patterns against
your actual `d_items` table**, ideally cross-checked against the
`mimic-code` reference concept queries (`vitalsign.sql`), since itemid sets
differ across monitor source systems within MIMIC-IV itself.

| Feature | Concept | Unit (typical) |
|---|---|---|
| `heart_rate_{min,max,mean}` | Heart rate | bpm |
| `sbp_{min,max,mean}` | Systolic blood pressure (non-invasive or arterial line) | mmHg |
| `dbp_{min,max,mean}` | Diastolic blood pressure | mmHg |
| `map_{min,max,mean}` | Mean arterial pressure | mmHg |
| `resp_rate_{min,max,mean}` | Respiratory rate | breaths/min |
| `temp_f_{min,max,mean}` | Temperature | °F — CHECK_SCHEMA: confirm unit; some rows may be °C under a different label |
| `spo2_{min,max,mean}` | Peripheral oxygen saturation | % |
| `gcs_total_min` | Glasgow Coma Scale, worst (lowest) total score in window | points (3–15). CHECK_SCHEMA: standard MIMIC-IV `d_items` typically has no single "GCS Total" chart item — GCS is recorded as eye/verbal/motor components. `features.sql` uses a direct-item match if present, else derives the total as the sum of each component's worst value in the window (standard clinical convention). Verify which path returns data in your build. |
| `gcs_total_max` | Glasgow Coma Scale, total score (direct item match only) | points (3–15). Only populated if your build has a direct "GCS Total" chart item; not derived from components (component-max summing is not a standard clinical convention). |

## First-24h Laboratory Summaries (`sql/features.sql`, from `mimiciv_hosp.labevents`)

Aggregated the same way, matched via `mimiciv_hosp.d_labitems.label`.
**CHECK_SCHEMA: verify label patterns**, especially that "creatinine"/
"glucose" exclusions of urine-based panels correctly separate serum from
urine tests in your load.

| Feature | Concept | Unit (typical) |
|---|---|---|
| `creatinine_{min,max,mean}` | Serum creatinine | mg/dL |
| `sodium_{min,max}` | Serum sodium | mEq/L |
| `potassium_{min,max}` | Serum potassium | mEq/L |
| `bicarbonate_{min,max}` | Serum bicarbonate | mEq/L |
| `hematocrit_{min,max}` | Hematocrit | % |
| `wbc_{min,max}` | White blood cell count | K/uL |
| `glucose_{min,max}` | Serum glucose | mg/dL |
| `bun_{min,max}` | Blood urea nitrogen | mg/dL |
| `platelet_{min,max}` | Platelet count | K/uL |
| `lactate_max` | Lactate (peak only — most clinically relevant direction) | mmol/L |

## Target Variable

| Field | Source | Definition |
|---|---|---|
| `in_hospital_mortality` | `mimiciv_hosp.admissions.hospital_expire_flag` | 1 if the patient died during this hospitalization, 0 otherwise. Reflects the outcome of the **entire** hospital stay, known only at discharge — used exclusively as the label, never as or alongside a feature. |

## Leakage-Prevention Checklist

- [x] All vitals/labs filtered strictly to `charttime < icu_intime + 24h`.
- [x] Stays shorter than 24h excluded (not truncated) from the cohort.
- [x] `hospital_expire_flag`, `dischtime`, `deathtime`, `dod` never referenced
      inside `features.sql` — the label is computed only in `cohort.sql`.
- [x] Only per-stay aggregate statistics exposed as features — no raw
      timestamps, no row counts that could encode length-of-stay-to-death.
- [x] `subject_id` used for split assignment only, dropped from the
      feature matrix before model training (`src/data_extract.py`).
- [x] `icu_outtime` / `icu_los_days` (ICU discharge time / length of stay)
      are not selected in `cohort.sql` and are additionally stripped by
      `src/data_extract.py` (`NON_FEATURE_TIMING_COLUMNS`) if ever
      reintroduced — both are outcome-adjacent and unknowable in the first
      24 hours. Fixed 2026-08-18 after a leakage review found they were
      previously reaching the model-ready feature matrix.
- [ ] CHECK_SCHEMA items above manually verified against your local
      MIMIC-IV load (must be completed by you before trusting results).
