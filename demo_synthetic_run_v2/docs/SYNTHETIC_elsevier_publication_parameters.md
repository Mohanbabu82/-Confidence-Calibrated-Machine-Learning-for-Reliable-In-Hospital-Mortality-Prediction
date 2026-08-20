# ⚠️ SYNTHETIC DATASET — DO NOT USE IN A REAL MANUSCRIPT ⚠️

**Every number below was computed by running the real project pipeline
(`notebooks/01`–`05`, `src/train_paper_models.py`, `src/calibration.py`,
`src/evaluate_paper_final.py`) against `mimic_synthetic.csv` — a file you
supplied that is explicitly named "synthetic" and is not MIMIC-IV data (no
PhysioNet DUA, no provenance to a real hospital cohort). The arithmetic is
100% real and unmanipulated; the underlying patients are not.**

If any number below ends up in an actual journal submission, that submission
will contain fabricated clinical results. Re-run this pipeline against real,
credentialed MIMIC-IV data before using any of it for
`docs/elsevier_publication_parameters.md`.

Generated: 2026-08-18 · Source data: `mimic_synthetic.csv` (6,000 rows,
5,997 after deduplication) · Pipeline run: `demo_synthetic_run_v2/`

---

## 1. Study overview

- **Dataset used in this run**: `mimic_synthetic.csv`, user-supplied, explicitly labeled synthetic. Schema: `subject_id, hadm_id, stay_id, age, gender, admission_type, first_careunit, heart_rate, map_bp, resp_rate, temperature, spo2, creatinine, bun, sodium, potassium, glucose, wbc, hemoglobin, platelet, lactate, bicarbonate, bilirubin, gcs, urine_output, mech_vent, hospital_expire_flag`. **Not real patient data.**
- **Intended real dataset**: MIMIC-IV Clinical Database Demo, v2.2 (still not obtained — see prior notes)
- **Intended PhysioNet URL**: https://physionet.org/content/mimic-iv-demo/2.2/
- **Prediction task**: binary in-hospital mortality (`hospital_expire_flag` → `mortality`)
- **Study design**: retrospective cohort (synthetic in this run)

---

## 2. Cohort parameters (from `table_1_cohort.csv`, `cohort_flow_counts.csv`)

- **Total rows loaded**: 6,000
- **Excluded (duplicate `subject_id`/`stay_id`)**: 3
- **Final analysis cohort N**: 5,997
- **Deaths**: 1,076 (17.94%)
- **Survivors**: 4,921 (82.06%)
- **Train / Validation / Test split**: 4,197 / 600 / 1,200 (70% / 10% / 20%, patient-level, stratified by mortality)

### Age (median [IQR], years)
- Overall: 64.0 [54.0, 76.0]
- Train: 64.0 [54.0, 75.0]
- Validation: 66.0 [56.0, 78.0]
- Test: 64.0 [54.0, 75.0]

### Sex distribution
- Male: 3,228 (53.8%)
- Female: 2,769 (46.2%)

### Admission type
- Emergency: 4,151 (69.2%)
- Urgent: 1,001 (16.7%)
- Elective: remainder (~14.1%, see full `table_1_cohort.csv`)

### Key clinical features (median [IQR], overall)
| Feature | Median [IQR] |
|---|---|
| Heart rate (bpm) | 85.0 [73.0, 97.0] |
| Mean arterial pressure (mmHg) | 78.0 [69.0, 87.0] |
| Respiratory rate (/min) | 19.0 [16.0, 22.0] |
| Temperature (°C) | 36.9 [36.4, 37.4] |
| SpO2 (%) | 96.0 [94.0, 98.0] |
| Creatinine (mg/dL) | 1.0 [0.7, 1.5] |
| BUN (mg/dL) | 22.0 [12.0, 32.0] |
| Sodium (mEq/L) | 139.0 [136.0, 142.0] |
| Potassium (mEq/L) | 4.1 [3.7, 4.5] |
| Glucose (mg/dL) | 139.0 [103.0, 177.0] |
| WBC (K/µL) | 10.8 [6.8, 14.9] |
| Hemoglobin (g/dL) | 11.0 [9.5, 12.5] |
| Platelets (K/µL) | 228.0 [165.0, 292.0] |
| Lactate (mmol/L) | 1.8 [1.2, 2.8] |
| Bicarbonate (mEq/L) | 24.0 [21.0, 27.0] |
| Bilirubin (mg/dL) | 0.7 [0.5, 1.2] |
| GCS total | 14.0 [13.0, 15.0] |
| Urine output, 24h (mL) | 1488.0 [933.0, 2051.0] |
| Mechanical ventilation (0/1) | 0.0 [0.0, 1.0] |

Full per-split breakdown: `demo_synthetic_run_v2/outputs/tables/table_1_cohort.csv`

### Missingness summary
No missing values were present in this dataset — every feature was complete
for all 5,997 rows. **This will not be true of real MIMIC-IV data**, where
missingness is expected and material; re-derive `cohort_missingness.csv`
once real data is used.

---

## 3. Model parameters (from `table_2_hyperparameters.csv`)

### Logistic Regression
| Hyperparameter | Value |
|---|---|
| Penalty | l2 |
| Solver | lbfgs |
| C (selected via validation AUROC grid search over {0.001, 0.01, 0.1, 1.0, 10.0}) | **10.0** |
| max_iter | 1000 |
| class_weight | balanced |
| random_state | 42 |
| Validation AUROC at selected C | **0.8163** |
| Training runtime | 0.13 s |
| Number of features | 30 (20 numeric [median-imputed, standard-scaled] + 10 one-hot categorical columns from `gender`, `admission_type`, `first_careunit`; 0 missingness-indicator columns, since no missing values were present) |

### LightGBM
| Hyperparameter | Value |
|---|---|
| learning_rate (selected via validation AUROC grid search over {0.01, 0.05, 0.1}) | **0.01** |
| n_estimators (upper bound; early stopping) | 300 |
| best_iteration (actual, via early stopping) | 40 |
| num_leaves | 31 |
| max_depth | -1 (unlimited) |
| early_stopping_rounds | 20 |
| scale_pos_weight (neg/pos ratio, computed from TRAIN labels only) | 4.5737 |
| random_state | 42 |
| Validation AUROC at selected learning_rate | **0.7836** |
| Training runtime | 0.88 s |
| Number of features | 30 (same feature set as logistic regression) |

### Software versions (this run)
| Package | Version |
|---|---|
| Python | 3.11.9 |
| pandas | 2.2.2 |
| numpy | 1.26.4 |
| scikit-learn | 1.5.1 |
| lightgbm | 4.5.0 |
| xgboost | 2.1.0 (installed; not used in this pipeline) |
| torch | 2.3.1+cpu (installed; not used — no MLP in this spec) |
| scipy | 1.13.1 |
| statsmodels | 0.14.2 |
| Platform | win32 |

---

## 4. Calibration parameters (from `table_3_calibration_metrics.csv`, validation set, n=600)

| Model | Method | Brier | ECE (10 bins) | ECE (15 bins) | MCE | NLL | Cal. intercept | Cal. slope |
|---|---|---|---|---|---|---|---|---|
| lightgbm | uncalibrated | 0.1385 | 0.1041 | 0.1033 | 0.1508 | 0.4465 | 1.8930 | 3.4736 |
| lightgbm | platt | 0.1260 | 0.0155 | 0.0223 | 0.1254 | 0.3937 | -0.0008 | 0.9995 |
| lightgbm | isotonic | 0.1225 | ~0.0000 | ~0.0000 | ~0.0000 | 0.3794 | ~0.0000 | 1.0001 |
| logistic_regression | uncalibrated | 0.1724 | 0.2179 | 0.2179 | 0.4639 | 0.5143 | -1.4771 | 1.0216 |
| logistic_regression | platt | 0.1143 | 0.0207 | 0.0258 | 0.1712 | 0.3652 | ~0.0000 | 0.9998 |
| logistic_regression | isotonic | 0.1096 | ~0.0000 | ~0.0000 | ~0.0000 | 0.3485 | -0.0003 | 0.9998 |

**Best calibration method per model (validation set):**
- **lightgbm**: isotonic (lowest Brier = 0.1225; near-zero ECE/MCE)
- **logistic_regression**: isotonic (lowest Brier = 0.1096; lowest NLL = 0.3485)

⚠️ **lightgbm/uncalibrated's calibration slope of 3.47** reflects genuine
under-confidence of the raw LightGBM scores relative to observed outcome
frequency in this dataset (a real, not fabricated, finding) — resolved by
both Platt and isotonic scaling (slopes → ~1.0).

⚠️ **Caveat inherited from the pipeline design**: Platt/isotonic are fit and
evaluated on the *same* validation set (n=600) — these are optimistic,
in-sample calibration numbers. The held-out check is Table 4 (test set,
n=1,200).

---

## 5. Main performance parameters (from `table_4_main_results.csv`, held-out test set, n=1,200)

| Model | Calib. method | AUROC [95% CI] | AUPRC | Brier [95% CI] | ECE [95% CI] |
|---|---|---|---|---|---|
| logistic_regression | uncalibrated | **0.7982** [0.7670, 0.8299] | 0.5197 | 0.1721 [0.1621, 0.1837] | 0.2072 [0.1946, 0.2205] |
| logistic_regression | platt | 0.7982 [0.7670, 0.8299] | 0.5197 | **0.1170** [0.1083, 0.1255] | 0.0324 [0.0223, 0.0524] |
| logistic_regression | isotonic | 0.7964 [0.7650, 0.8278] | 0.4778 | 0.1179 [0.1089, 0.1269] | 0.0314 [0.0192, 0.0507] |
| lightgbm | uncalibrated | 0.7639 [0.7296, 0.7959] | 0.4161 | 0.1379 [0.1355, 0.1406] | 0.0908 [0.0733, 0.1102] |
| lightgbm | platt | 0.7639 [0.7296, 0.7959] | 0.4161 | 0.1260 [0.1197, 0.1328] | **0.0220** [0.0147, 0.0444] |
| lightgbm | isotonic | 0.7638 [0.7295, 0.7958] | 0.3961 | 0.1263 [0.1198, 0.1332] | 0.0307 [0.0223, 0.0543] |

95% CIs: 1,000 stratified bootstrap resamples (resampled within each
outcome class, preserving observed class counts — `src/evaluate_paper_final.py`).

**Sensitivity / specificity / PPV / NPV / F1 / balanced accuracy at the
primary threshold (0.2)**:

| Model | Calib. method | Sens. | Spec. | PPV | NPV | F1 | Balanced acc. |
|---|---|---|---|---|---|---|---|
| logistic_regression | uncalibrated | 0.940 | 0.371 | 0.246 | 0.966 | 0.390 | 0.655 |
| logistic_regression | platt | 0.637 | 0.789 | 0.397 | 0.909 | 0.489 | 0.713 |
| logistic_regression | isotonic | 0.600 | 0.819 | 0.420 | 0.904 | 0.494 | 0.710 |
| lightgbm | uncalibrated | 0.935 | 0.331 | 0.234 | 0.959 | 0.374 | 0.633 |
| lightgbm | platt | 0.651 | 0.738 | 0.352 | 0.906 | 0.457 | 0.695 |
| lightgbm | isotonic | 0.674 | 0.729 | 0.352 | 0.911 | 0.463 | 0.702 |

### Best-performing model and calibration method (test set)
- **By AUROC**: `logistic_regression` / `uncalibrated` (tied with `platt`, since Platt scaling doesn't change rank order) — AUROC = **0.7982** [95% CI 0.7670, 0.8299]
- **By Brier score**: `logistic_regression` / `platt` — Brier = **0.1170** [95% CI 0.1083, 0.1255]
- **By ECE**: `lightgbm` / `platt` — ECE = **0.0220** [95% CI 0.0147, 0.0444]

These are real, statistically meaningful estimates given n=1,200 (unlike
the earlier 24-case smoke-test run) — the CIs are appropriately narrower.

---

## 6. Clinical threshold parameters (from `table_5_clinical_thresholds.csv`)

**Decision thresholds evaluated**: 0.2 (primary, per config) and 0.5 (reference).

| Model | Method | Threshold | Sens. | Spec. | PPV | NPV |
|---|---|---|---|---|---|---|
| logistic_regression | uncalibrated | 0.2 | 0.940 | 0.371 | 0.246 | 0.966 |
| logistic_regression | uncalibrated | 0.5 | 0.693 | 0.768 | 0.394 | 0.920 |
| logistic_regression | platt | 0.2 | 0.637 | 0.789 | 0.397 | 0.909 |
| logistic_regression | platt | 0.5 | 0.321 | 0.958 | 0.627 | 0.866 |
| logistic_regression | isotonic | 0.2 | 0.600 | 0.819 | 0.420 | 0.904 |
| logistic_regression | isotonic | 0.5 | 0.321 | 0.958 | 0.627 | 0.866 |
| lightgbm | uncalibrated | 0.2 | 0.935 | 0.331 | 0.234 | 0.959 |
| lightgbm | uncalibrated | 0.5 | 0.000 | 1.000 | N/A | 0.821 |
| lightgbm | platt | 0.2 | 0.651 | 0.738 | 0.352 | 0.906 |
| lightgbm | platt | 0.5 | 0.135 | 0.978 | 0.569 | 0.838 |
| lightgbm | isotonic | 0.2 | 0.674 | 0.729 | 0.352 | 0.911 |
| lightgbm | isotonic | 0.5 | 0.153 | 0.975 | 0.569 | 0.841 |

"N/A" = undefined (zero predicted positives at that threshold) — reported
as such, not approximated.

No referral/abstention layer is implemented in the active 01–05 pipeline
(that was a feature of an earlier, now-archived pipeline version).

---

## 7. Figures and tables inventory (`demo_synthetic_run_v2/outputs/`)

### Figures (`outputs/figures/`)

| Filename | Type | Models/methods shown | Suggested # |
|---|---|---|---|
| `figure1_study_flowchart.png/.pdf` + `.mmd` | Study flowchart | N/A (cohort flow: 6000 → 5997 → 4197/600/1200 splits) | Figure 1 |
| `figure2_system_architecture.png/.pdf` + `.mmd` | Architecture diagram | N/A (static pipeline design) | Figure 2 |
| `figure3_reliability_logistic_regression_uncalibrated.png/.pdf` | Reliability diagram (test) | logistic_regression / uncalibrated (best by AUROC) | Figure 3a |
| `figure3_reliability_logistic_regression_platt.png/.pdf` | Reliability diagram (test) | logistic_regression / platt (2nd-best by AUROC, tied) | Figure 3b |
| `figure4_roc_curves.png/.pdf` | ROC curves | All 2 models × 3 methods, test set | Figure 4 |
| `figure5_precision_recall_curves.png/.pdf` | PR curves | All 2 models × 3 methods, test set | Figure 5 |
| `reliability_{model}_{method}.png/.pdf` (×6) | Reliability diagrams (validation) | Both models × uncalibrated/platt/isotonic | Supplementary |

### Tables (`outputs/tables/`)

| Filename | Content | Suggested # |
|---|---|---|
| `table_1_cohort.csv` / `.md` | Cohort characteristics, overall + by split | Table 1 |
| `cohort_flow_counts.csv` | N at each cohort-construction stage | (feeds Figure 1) |
| `table_2_hyperparameters.csv` | Model hyperparameters + validation AUROC | Table 2 |
| `table_3_calibration_metrics.csv` | Calibration metrics, validation set | Table 3 |
| `table_4_main_results.csv` | Discrimination + calibration, test set, with 95% CIs | Table 4 |
| `table_5_clinical_thresholds.csv` | Sens/spec/PPV/NPV at each threshold | Table 5 |
| `table{1-5}_*.csv/.md/.tex` | Manuscript-formatted exports of the above | Tables 1-5 (submission-ready format) |

---

## 8. Reproducibility metadata (this run)

- **Random seed**: 42 (cohort split, model training, bootstrap resampling)
- **Bootstrap resamples**: 1,000 (stratified, AUROC/Brier/ECE 95% CIs in Table 4)
- **Split ratios/strategy**: 70% / 10% / 20% train/validation/test, patient-level, stratified by mortality
- **Preprocessing**: median imputation + missingness indicators (none triggered here), one-hot encoding, standard scaling — all fit on TRAIN only
- **Hardware**: Not recorded (CPU-only run; training completed in <1s per model)
- **Software environment**: see Section 3

---

## 9. Data and code availability statements (draft text — placeholder until real data is used)

### Data Availability Statement (draft)
> [PLACEHOLDER] This study used [DATASET NAME], accessible at [URL]. This
> run instead used a locally supplied synthetic file (`mimic_synthetic.csv`)
> and must not be cited as real data provenance.

### Code Availability Statement (template)
> All code used to construct the cohort, train models, calibrate
> predictions, and generate the figures and tables in this manuscript is
> available at [GitHub URL PLACEHOLDER], released under the [LICENSE
> PLACEHOLDER] license.

---

## 10. Ethics and compliance statements (draft text)

### Ethics statement (draft — do not use until real data is confirmed)
> [PLACEHOLDER — ethics statement must reflect the actual data source used; do not reuse MIMIC-IV boilerplate for a non-MIMIC dataset]

### Declaration of competing interests (placeholder)
> [PLACEHOLDER — insert author disclosures or "The authors declare no competing interests."]

### Author contributions (CRediT taxonomy, placeholders)
> [Author 1]: Conceptualization, Methodology, Software, Formal analysis, Writing – original draft.
> [Author 2]: [PLACEHOLDER role].

---

## 11. Suggested manuscript structure for Elsevier

- **Recommended article type**: Original Research Article
- **Suggested target journal types**: medical informatics, clinical AI, critical care informatics
- **Word count estimate**: Not recorded
- **Tables/figures for main manuscript**: 5 tables (Tables 1–5), 5 figures (Figures 1–5); move the 6 validation-set `reliability_*` files to supplementary material

---

## 12. Limitations and future work

- **This entire run used a synthetic, non-clinical dataset** — the single largest limitation; no clinical conclusion can be drawn from any number above
- **No missingness modeled**: real ICU data has structural missingness this run did not exercise
- **Single-center assumption not applicable**: synthetic data has no real-world site
- **Full MIMIC-IV / external validation still required** before any of these numbers can support a publication
- **Class imbalance**: 17.9% mortality prevalence handled via `class_weight="balanced"` / `scale_pos_weight`; PPV at low prevalence remains modest even for a well-discriminating model — expected, not a defect
- **Calibration transportability untested**: Platt/isotonic calibration here is validation-set in-sample; test-set numbers in Table 4 are the held-out check, but true transportability to a different population is untested
