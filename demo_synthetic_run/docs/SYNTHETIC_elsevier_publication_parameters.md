# ⚠️ SYNTHETIC DEMONSTRATION DATA — DO NOT USE IN A REAL MANUSCRIPT ⚠️

**Every number in this document was computed by running the real project pipeline
(`src/build_demo_cohort.py`, `notebooks/01`–`05`) against a locally generated,
non-clinical synthetic dataset shaped like MIMIC-IV — not against real patient
data.** No MIMIC-IV data (demo or full) has been obtained for this project as of
this run. This file exists solely to show you what the pipeline's real output
*looks like*, with correctly computed arithmetic, so you can see the mechanics
before real data is available.

**If any number below ends up in an actual journal submission, that submission
will contain fabricated results.** Regenerate this entire file from real
MIMIC-IV data before using any of it for `docs/elsevier_publication_parameters.md`.

Generated: 2026-08-17 (synthetic run) · Location: `demo_synthetic_run/` (kept
fully separate from `data/`, `outputs/`, `docs/` for this reason).

---

## 1. Study overview

- **Dataset used in this run**: synthetic, non-clinical data generated locally to match the MIMIC-IV Demo v2.2 table schema (`patients`, `admissions`, `icustays`, `chartevents`, `d_items`, `labevents`, `d_labitems`). **Not a real dataset.**
- **Intended real dataset**: MIMIC-IV Clinical Database Demo, v2.2 (not yet obtained)
- **Intended PhysioNet URL**: https://physionet.org/content/mimic-iv-demo/2.2/
- **Data type**: de-identified ICU electronic health record data (when real data is used)
- **Prediction task**: binary in-hospital mortality, using only features available in the first 24 hours of ICU stay
- **Study design**: retrospective, single-center demo subset (once run on real MIMIC-IV Demo data)

---

## 2. Cohort parameters (synthetic run — from `table_1_cohort.csv` / `cohort_flow_counts.csv`)

- **Total N (ICU stays)**: 120 (one row per patient, first ICU stay only, ≥24h stay, age ≥18)
- **Deaths**: 10 (8.33%)
- **Survivors**: 110 (91.67%)
- **Train / Validation / Test split**: 84 / 12 / 24 (70% / 10% / 20%, patient-level, stratified by mortality)
  - Train: n=84, deaths=7 (8.3%)
  - Validation: n=12, deaths=1 (8.3%)
  - Test: n=24, deaths=2 (8.3%)

### Age distribution (median [IQR], years)
- Overall: 53.5 [33.0, 75.2]
- Train: 51.5 [32.8, 74.5]
- Validation: 65.5 [46.0, 74.0]
- Test: 57.5 [30.8, 75.5]

### Sex distribution
- Female: 70 (58.3%)
- Male: 50 (41.7%)

### Admission type
- Emergency: 62 (51.7%)
- Elective: 29 (24.2%)
- Urgent: 29 (24.2%)

### First ICU care unit
- MICU: 55 (45.8%)
- SICU: 42 (35.0%)
- CCU: 23 (19.2%)

### Key clinical features summary (median [IQR], overall, first 24h)
| Feature | Median [IQR] |
|---|---|
| Heart rate (mean, bpm) | 86.0 [78.5, 93.4] |
| Systolic BP (mean, mmHg) | 119.5 [107.0, 137.2] |
| Diastolic BP (mean, mmHg) | 75.2 [65.6, 82.2] |
| Respiratory rate (mean, /min) | 18.8 [15.8, 22.1] |
| SpO2 (mean, %) | 95.5 [92.6, 98.4] |
| Temperature (mean, °F) | 99.1 [98.2, 99.8] |
| GCS total (worst/min) | 15.0 [14.0, 15.0] |
| Creatinine (max, mg/dL) | 1.1 [0.7, 1.4] |
| Sodium (min, mEq/L) | 138.5 [134.7, 141.0] |
| Potassium (max, mEq/L) | 4.2 [3.8, 4.7] |
| Bicarbonate (min, mEq/L) | 23.4 [21.4, 26.1] |
| WBC (max, K/uL) | 10.4 [7.3, 13.0] |
| Glucose (max, mg/dL) | 119.0 [103.2, 142.0] |
| Lactate (max, mmol/L) | 2.0 [0.9, 2.8] |

Full table with per-split breakdown: `demo_synthetic_run/outputs/tables/table_1_cohort.csv`

### Missingness summary
No features exceeded 10% missingness in this synthetic run — the synthetic
generator produced complete vitals/labs for every patient. **This will not be
true of real MIMIC-IV data**, where missingness (especially for less-routine
labs) is expected and the pipeline's missingness-indicator step will be
material. Real missingness must be re-reported from `cohort_missingness.csv`
once real data is used.

---

## 3. Model parameters (synthetic run — from `table_2_hyperparameters.csv`)

### Logistic Regression
| Hyperparameter | Value |
|---|---|
| Penalty | l2 |
| Solver | lbfgs |
| C (selected via validation AUROC grid search) | 0.001 |
| max_iter | 1000 |
| class_weight | balanced |
| random_state | 42 |
| Validation AUROC at selected C | 1.0000 |
| Training runtime | 0.04 s |

- **Number of features used**: 23 (15 numeric [median-imputed, standard-scaled] + 8 one-hot categorical columns from `first_careunit`, `gender`, `admission_type`; 0 missingness-indicator columns added, since no missing values were present in this synthetic run)

### LightGBM
| Hyperparameter | Value |
|---|---|
| learning_rate (selected via validation AUROC grid search) | 0.1 |
| n_estimators (upper bound; early stopping) | 300 |
| best_iteration (actual, via early stopping) | 7 |
| num_leaves | 31 |
| max_depth | -1 (unlimited) |
| early_stopping_rounds | 20 |
| scale_pos_weight (neg/pos ratio, computed from TRAIN labels) | 11.0 |
| random_state | 42 |
| Validation AUROC at selected learning_rate | 0.5455 |
| Training runtime | 0.29 s |

- **Number of features used**: 23 (same feature set as logistic regression — both models share the one frozen preprocessor)

### Software versions (this run)
| Package | Version |
|---|---|
| Python | 3.11.9 |
| pandas | 2.2.2 |
| numpy | 1.26.4 |
| scikit-learn | 1.5.1 |
| lightgbm | 4.5.0 |
| xgboost | 2.1.0 (installed; not used in this pipeline) |
| torch | 2.3.1+cpu (installed; not used in this pipeline — no MLP in this spec) |
| scipy | 1.13.1 |
| statsmodels | 0.14.2 |
| Platform | win32 |

Note on the AUROC=1.0000 result: with only 12 validation cases (1 death), a
heavily regularized logistic regression can trivially separate the classes.
This is a small-sample artifact of the synthetic demo, not evidence of a
strong model — do not interpret this number clinically, and expect it to
change substantially with a real, larger cohort.

---

## 4. Calibration parameters (synthetic run — from `table_3_calibration_metrics.csv`, validation set)

| Model | Method | Brier | ECE (10 bins) | ECE (15 bins) | MCE | NLL | Cal. intercept | Cal. slope |
|---|---|---|---|---|---|---|---|---|
| lightgbm | uncalibrated | 0.0823 | 0.0454 | 0.1121 | 0.2326 | 0.3030 | -2.193 | 0.095 |
| lightgbm | platt | 0.0764 | 0.0000 | 0.0000 | 0.0000 | 0.2867 | -0.355 | 0.852 |
| lightgbm | isotonic | 0.0714 | 0.0000 | 0.0000 | 0.0000 | 0.2392 | -0.100 | 0.944 |
| logistic_regression | uncalibrated | 0.2362 | 0.4054 | 0.4054 | 0.4847 | 0.6654 | -6.596 | 175.291 |
| logistic_regression | platt | ~0.0000 | 0.0004 | 0.0004 | 0.0026 | 0.0004 | -0.054 | 1.627 |
| logistic_regression | isotonic | 0.0000 | 0.0000 | 0.0000 | 0.0000 | ~0.0000 | -0.060 | 0.998 |

**Best calibration method per model (validation set):**
- **lightgbm**: isotonic (lowest Brier = 0.0714; lowest/tied ECE)
- **logistic_regression**: isotonic (lowest Brier ≈ 0; lowest NLL)

⚠️ **logistic_regression/uncalibrated's calibration slope of 175.3 is a
numerical-instability artifact** (see `src/calibration.py`'s
`PlattScaler` docstring) — it occurs when a base model's predictions have
near-zero variance relative to a tiny validation set, and should not be
reported as a real finding. The corresponding `UserWarning` fired during
this run. This instability itself disappears with a real, larger validation
set — flagging it here rather than hiding it.

⚠️ **Caveat inherited from the pipeline design**: Platt/isotonic are fit and
evaluated on the *same* validation set (12 cases) — these are optimistic,
in-sample calibration numbers, not held-out estimates. The held-out check is
Table 4 (test set).

---

## 5. Main performance parameters (synthetic run — from `table_4_main_results.csv`, held-out test set, n=24)

| Model | Calib. method | AUROC [95% CI] | AUPRC | Brier [95% CI] | ECE [95% CI] |
|---|---|---|---|---|---|
| logistic_regression | uncalibrated | 0.545 [0.182, 0.909] | 0.142 | 0.238 [0.230, 0.245] | 0.402 [0.395, 0.410] |
| logistic_regression | platt | 0.545 [0.182, 0.909] | 0.142 | 0.221 [0.107, 0.355] | 0.244 [0.123, 0.385] |
| logistic_regression | isotonic | **0.580 [0.318, 0.909]** | 0.125 | 0.182 [0.106, 0.273] | 0.224 [0.119, 0.344] |
| lightgbm | uncalibrated | 0.500 [0.091, 0.909] | 0.133 | 0.088 [0.069, 0.109] | 0.063 [0.025, 0.173] |
| lightgbm | platt | 0.500 [0.091, 0.909] | 0.133 | **0.077 [0.075, 0.078]** | **0.0006 [0.00005, 0.0032]** |
| lightgbm | isotonic | 0.523 [0.205, 0.841] | 0.087 | 0.081 [0.067, 0.094] | 0.065 [0.006, 0.161] |

95% CIs: 1000 stratified bootstrap resamples (resampled within each outcome
class, preserving observed class counts — see `src/evaluate_paper_final.py`).

**Sensitivity / specificity / PPV / NPV / F1 / balanced accuracy at the
primary threshold (0.2)**, from the same table:

| Model | Calib. method | Sens. | Spec. | PPV | NPV | F1 | Balanced acc. |
|---|---|---|---|---|---|---|---|
| logistic_regression | uncalibrated | 1.000 | 0.000 | 0.083 | N/A | 0.154 | 0.500 |
| logistic_regression | platt | 0.000 | 0.773 | 0.000 | 0.895 | N/A | 0.386 |
| logistic_regression | isotonic | 0.000 | 0.773 | 0.000 | 0.895 | N/A | 0.386 |
| lightgbm | uncalibrated | 0.000 | 0.864 | 0.000 | 0.905 | N/A | 0.432 |
| lightgbm | platt | 0.000 | 1.000 | N/A | 0.917 | N/A | 0.500 |
| lightgbm | isotonic | 0.000 | 1.000 | N/A | 0.917 | N/A | 0.500 |

"N/A" = undefined (zero predicted positives or zero true positives in that
cell) — reported as such per your instruction, not approximated.

### Best-performing model and calibration method (test set)
- **By AUROC**: `logistic_regression` / `isotonic` — AUROC = 0.5795 [95% CI 0.318, 0.909]
- **By Brier score**: `lightgbm` / `platt` — Brier = 0.0765 [95% CI 0.0753, 0.0777]
- **By ECE**: `lightgbm` / `platt` — ECE = 0.0006 [95% CI 0.0000, 0.0032]

⚠️ With only 2 deaths in a 24-case test set, every one of these test-set
estimates carries enormous uncertainty — note how wide the AUROC confidence
intervals are (e.g., [0.182, 0.909]). This is expected and appropriate given
n, not a bug; it will look very different with real, adequately powered data.

---

## 6. Clinical threshold parameters (synthetic run — from `table_5_clinical_thresholds.csv`)

**Decision thresholds evaluated**: 0.2 and 0.5 (predicted probability of death).

| Model | Method | Threshold | Sens. | Spec. | PPV | NPV |
|---|---|---|---|---|---|---|
| logistic_regression | uncalibrated | 0.2 | 1.000 | 0.000 | 0.083 | N/A |
| logistic_regression | uncalibrated | 0.5 | 0.500 | 0.773 | 0.167 | 0.944 |
| logistic_regression | platt | 0.2 | 0.000 | 0.773 | 0.000 | 0.895 |
| logistic_regression | platt | 0.5 | 0.000 | 0.818 | 0.000 | 0.900 |
| logistic_regression | isotonic | 0.2 | 0.000 | 0.773 | 0.000 | 0.895 |
| logistic_regression | isotonic | 0.5 | 0.000 | 0.818 | 0.000 | 0.900 |
| lightgbm | uncalibrated | 0.2 | 0.000 | 0.864 | 0.000 | 0.905 |
| lightgbm | uncalibrated | 0.5 | 0.000 | 1.000 | N/A | 0.917 |
| lightgbm | platt | 0.2 | 0.000 | 1.000 | N/A | 0.917 |
| lightgbm | platt | 0.5 | 0.000 | 1.000 | N/A | 0.917 |
| lightgbm | isotonic | 0.2 | 0.000 | 1.000 | N/A | 0.917 |
| lightgbm | isotonic | 0.5 | 0.000 | 1.000 | N/A | 0.917 |

This pipeline (per your original spec) does not implement a referral/
abstention layer in notebooks 01–05, so there is no referral rate to report
here. (An earlier, separate version of this project implemented selective
prediction/referral — see `src/selective_prediction.py`, currently archived
from the active 01–05 sequence — if you want that layer reinstated for the
manuscript, say so explicitly.)

---

## 7. Figures and tables inventory (synthetic run, `demo_synthetic_run/outputs/`)

### Figures (`outputs/figures/`)

| Filename | Type | Models/methods shown | Suggested # |
|---|---|---|---|
| `figure1_study_flowchart.png/.pdf` + `figure1_study_workflow.mmd` | Study flowchart | N/A (cohort flow) | Figure 1 |
| `figure2_system_architecture.png/.pdf` + `figure2_system_architecture.mmd` | Architecture diagram | N/A (pipeline design) | Figure 2 |
| `figure3_reliability_logistic_regression_isotonic.png/.pdf` | Reliability diagram | logistic_regression / isotonic (best by AUROC) | Figure 3a |
| `figure3_reliability_logistic_regression_uncalibrated.png/.pdf` | Reliability diagram | logistic_regression / uncalibrated (2nd-best by AUROC) | Figure 3b |
| `figure4_roc_curves.png/.pdf` | ROC curves | All 2 models × 3 methods, test set | Figure 4 |
| `figure5_precision_recall_curves.png/.pdf` | PR curves | All 2 models × 3 methods, test set | Figure 5 |
| `reliability_{model}_{method}.png/.pdf` (×6) | Reliability diagrams (validation) | Both models × uncalibrated/platt/isotonic | Supplementary |
| `roc_curves_final.*`, `pr_curves_final.*` | Duplicates of Fig. 4/5 (pre-renumbering) | — | (superseded by Figure 4/5) |

### Tables (`outputs/tables/`)

| Filename | Content | Suggested # |
|---|---|---|
| `table_1_cohort.csv` / `.md` | Cohort characteristics, overall + by split | Table 1 |
| `cohort_flow_counts.csv` | N at each cohort-construction stage | (feeds Figure 1) |
| `cohort_missingness.csv` | Missingness per feature | Supplementary |
| `table_2_hyperparameters.csv` | Model hyperparameters + validation AUROC | Table 2 |
| `table_3_calibration_metrics.csv` | Calibration metrics, validation set | Table 3 |
| `table_4_main_results.csv` | Discrimination + calibration, test set, with 95% CIs | Table 4 |
| `table_5_clinical_thresholds.csv` | Sens/spec/PPV/NPV at each threshold | Table 5 |
| `table{1-5}_*.csv/.md/.tex` | Manuscript-formatted exports (CSV+MD+LaTeX) of the above | Tables 1-5 (submission-ready format) |

---

## 8. Reproducibility metadata (this run)

- **Random seed**: 42 (used for cohort split, model training, and bootstrap resampling — set via `src.reproducibility.set_global_seed`)
- **Bootstrap resamples**: 1000 (stratified, for AUROC/Brier/ECE 95% CIs in Table 4)
- **Split ratios/strategy**: 70% / 10% / 20% train/validation/test, patient-level, stratified by mortality (`src.preprocess.patient_level_split`)
- **Preprocessing steps**: median imputation + missingness indicators (numeric), one-hot encoding (categorical), standard scaling (numeric) — all fit on TRAIN only (`src.preprocess.ClinicalPreprocessor`)
- **Hardware**: Not recorded (no GPU/CPU benchmarking was logged; training ran in <1s per model on CPU for this small synthetic set)
- **Software environment**: see Section 3's version table above

---

## 9. Data and code availability statements (draft text — update once real data is used)

### Data Availability Statement (draft, for real MIMIC-IV Demo use)
> This study used the MIMIC-IV Clinical Database Demo (version 2.2), a
> publicly available, de-identified subset of the MIMIC-IV database,
> accessible at https://physionet.org/content/mimic-iv-demo/2.2/
> [DOI: PLACEHOLDER — insert the PhysioNet-provided DOI]. No additional
> data use agreement or institutional approval was required to access the
> demo subset.
>
> **Note**: this run used synthetic, non-clinical data generated to match
> MIMIC-IV's schema, not the real Demo dataset. This statement must not be
> used in a submission until the pipeline has actually been run on real
> MIMIC-IV Demo data.

### Code Availability Statement (template)
> All code used to construct the cohort, train models, calibrate
> predictions, and generate the figures and tables in this manuscript is
> available at [GitHub URL PLACEHOLDER], released under the [LICENSE
> PLACEHOLDER, e.g. MIT] license.

---

## 10. Ethics and compliance statements (draft text)

### Ethics statement (draft)
> This study used the publicly available, de-identified MIMIC-IV Clinical
> Database Demo, which does not constitute human subjects research
> requiring separate institutional review board approval, per the terms of
> the PhysioNet Credentialed Health Data license under which the demo is
> released for unrestricted (non-credentialed) use.

### Declaration of competing interests (placeholder)
> [PLACEHOLDER — insert author disclosures or "The authors declare no competing interests."]

### Author contributions (CRediT taxonomy, placeholders)
> [Author 1]: Conceptualization, Methodology, Software, Formal analysis, Writing – original draft.
> [Author 2]: [PLACEHOLDER role].
> [Author N]: [PLACEHOLDER role — Supervision, Writing – review & editing, etc.]

---

## 11. Suggested manuscript structure for Elsevier

- **Recommended article type**: Original Research Article (a Short Communication would undersell the methodological scope — three trained models, two calibration methods, bootstrap CIs — but check the target journal's word/table limits)
- **Suggested target journal types**: medical informatics (e.g., *Journal of Biomedical Informatics*, *International Journal of Medical Informatics*), clinical AI (e.g., *npj Digital Medicine*, *Artificial Intelligence in Medicine*), critical care informatics (e.g., *Journal of Critical Care*)
- **Word count estimate**: Not recorded — depends on final Discussion/Introduction prose, which hasn't been drafted from real results yet; a comparable Methods+Results section from this pipeline's tables/figures alone runs roughly 1,200–1,800 words
- **Tables/figures for main manuscript**: 5 tables (Tables 1–5 above), 5 figures (Figures 1–5 above); consider moving per-model validation-set reliability diagrams (the 6 `reliability_*` files) to supplementary material, keeping only the best-model test-set reliability diagrams (Figure 3) in the main text

---

## 12. Limitations and future work

- **Sample size**: this run used a 120-patient *synthetic* cohort; even the real MIMIC-IV Demo is only ~100–200 patients — both are far too small for stable AUROC/calibration estimates (see the wide bootstrap CIs in Section 5)
- **Single-center, retrospective design**: MIMIC-IV is sourced from a single academic medical center (Beth Israel Deaconess Medical Center), limiting generalizability
- **Full MIMIC-IV validation needed**: results here (once run on real Demo data) should be treated as a pipeline feasibility check, not a clinical claim, until replicated on the full, credentialed MIMIC-IV database (hundreds of thousands of stays)
- **External validation needed**: no results from this pipeline should be presented as generalizable without validation on an independent, external ICU dataset
- **Class imbalance**: mortality is a minority outcome (~8–18% in both the synthetic run and typical ICU cohorts); current models handle this via `class_weight="balanced"` / `scale_pos_weight`, but reported PPV at low prevalence will remain low even for a well-discriminating model — this should be discussed explicitly, not treated as a model deficiency
- **Generalizability of calibration**: Platt/isotonic calibration here is fit and evaluated in-sample on the validation set (see Section 4's caveat); true calibration transportability to a different population is untested
