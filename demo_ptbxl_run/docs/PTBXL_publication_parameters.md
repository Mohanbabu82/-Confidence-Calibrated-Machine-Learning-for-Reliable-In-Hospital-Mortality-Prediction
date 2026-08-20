# Confidence-Calibrated ECG Abnormality Detection — Publication Parameters

**Dataset: PTB-XL v1.0.3 (real, PhysioNet-hosted, publicly available — no synthetic data).**
This is a **separate paper** from the in-hospital mortality project. PTB-XL has no
mortality outcome, so it cannot be used for that paper; this document is for a
distinct research question: *predicting ECG diagnostic abnormality from
first-principles per-lead signal summary statistics, with post-hoc calibration.*

**Scope disclosure (read before using any number below):**
- Source: `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip`, user-supplied, verified against the real PTB-XL file structure (`ptbxl_database.csv`, `scp_statements.csv`, WFDB `.dat`/`.hea` signals).
- **This run used a stratified, seed-42, N=4,000 subsample of the full 21,799-record dataset** (folds and label balance preserved), for tractable processing time — not the full dataset. All 4,000 records' raw signals were read and real per-lead statistics computed; nothing is simulated.
- Features are **simple per-lead summary statistics** (mean/std/min/max/RMS over the 10-second, 100 Hz recording per lead), not a deep-learning model over raw waveforms, and not the clinically validated feature sets (QRS duration, ST-segment deviation, etc.) used in the PTB-XL benchmark literature. Discrimination reported below reflects this simplified feature set, not a ceiling on what ECG signal can achieve.
- Task is **binary**: `abnormal` = 0 if the record's SCP-derived diagnostic superclass is exactly `{NORM}`, 1 otherwise (any of MI / STTC / CD / HYP present). This collapses the standard PTB-XL 5-class superclass task to binary for direct reuse of the existing calibration pipeline — a legitimate simplification, but not the standard multi-label PTB-XL benchmark task.

---

## 1. Study overview

- **Dataset**: PTB-XL, Physikalisch-Technische Bundesanstalt ECG database, v1.0.3
- **PhysioNet URL**: https://physionet.org/content/ptb-xl/1.0.3/
- **Data type**: de-identified 12-lead ECG recordings (10s, 100 Hz in this run) + clinical metadata
- **Prediction task**: binary ECG abnormality detection (NORM vs. MI/STTC/CD/HYP), from per-lead signal summary statistics + demographics
- **Study design**: retrospective, multi-site (PTB-XL spans multiple recording sites/devices), using the dataset's own patient-stratified fold assignment

---

## 2. Cohort parameters

- **Full dataset**: 21,799 records; 21,388 with a parseable diagnostic superclass
- **Analysis subsample (this run)**: N = **4,000** (stratified by fold × label, seed 42)
- **Abnormal**: 2,304 (57.60%) — **Normal**: 1,696 (42.40%)
- **Split** (PTB-XL official `strat_fold` protocol, patient-level, zero leakage verified — 0/3,857 patients span more than one fold): folds 1–8 → train (n=3,195), fold 9 → validation (n=401), fold 10 → test (n=404)

### Demographics
- Age: present for all 4,000 records (no missing)
- Sex: 2,075 male, 1,925 female
- **Missingness**: height missing in 2,632/4,000 (65.8%), weight missing in 2,223/4,000 (55.6%) — genuine, real missingness in PTB-XL metadata, handled by the pipeline's median-imputation + missingness-indicator step (both indicator columns were added by the preprocessor)

### Features (63 numeric + 1 categorical after preprocessing → 67 columns post-encoding)
- Per-lead (I, II, III, aVR, aVL, aVF, V1–V6) mean, std, min, max, RMS = 60 signal features
- Demographics: age, height, weight (3 features)
- Categorical: gender (one-hot)

Full table: `demo_ptbxl_run/outputs/tables/table_1_cohort.csv`

---

## 3. Model parameters (`table_2_hyperparameters.csv`)

### Logistic Regression
| Hyperparameter | Value |
|---|---|
| Penalty / solver | l2 / lbfgs |
| C (grid search over {0.001, 0.01, 0.1, 1.0, 10.0}) | **0.01** |
| class_weight | balanced |
| Validation AUROC | **0.8134** |
| Runtime | 0.23 s |

### LightGBM
| Hyperparameter | Value |
|---|---|
| learning_rate (grid search over {0.01, 0.05, 0.1}) | **0.05** |
| best_iteration (early stopping, patience=20) | 100 |
| num_leaves / max_depth | 31 / -1 |
| scale_pos_weight (train neg/pos) | 0.7355 |
| Validation AUROC | **0.8621** |
| Runtime | 2.12 s |

Software versions: same environment as the mortality pipeline (Python 3.11.9, scikit-learn 1.5.1, lightgbm 4.5.0) plus **wfdb** (installed this session for WFDB signal I/O).

---

## 4. Calibration parameters (`table_3_calibration_metrics.csv`, validation set, n=401)

| Model | Method | Brier | ECE (10 bins) | ECE (15 bins) | MCE | NLL | Cal. slope |
|---|---|---|---|---|---|---|---|
| lightgbm | uncalibrated | 0.1527 | 0.0610 | 0.0635 | 0.1613 | 0.4631 | 1.1028 |
| lightgbm | platt | 0.1497 | 0.0474 | 0.0516 | 0.1175 | 0.4540 | 1.0002 |
| lightgbm | isotonic | **0.1424** | ~0.0000 | ~0.0000 | ~0.0000 | **0.4280** | 1.0002 |
| logistic_regression | uncalibrated | 0.1805 | 0.0847 | 0.0896 | 0.1536 | 0.5323 | 1.3588 |
| logistic_regression | platt | 0.1737 | 0.0470 | 0.0496 | 0.0923 | 0.5114 | 1.0000 |
| logistic_regression | isotonic | 0.1663 | ~0.0000 | ~0.0000 | ~0.0000 | 0.4846 | 1.0000 |

**Best by validation Brier/NLL**: `lightgbm` / isotonic for both models.

---

## 5. Main results (`table_4_main_results.csv`, held-out test set, n=404, unlocked once)

| Model | Method | AUROC [95% CI] | AUPRC | Brier [95% CI] | ECE [95% CI] |
|---|---|---|---|---|---|
| logistic_regression | uncalibrated | 0.7497 [0.6992, 0.7975] | 0.8002 | 0.2049 [0.1853, 0.2266] | 0.0696 [0.0600, 0.1182] |
| logistic_regression | platt | 0.7497 [0.6992, 0.7975] | 0.8002 | 0.2048 [0.1823, 0.2282] | 0.0802 [0.0600, 0.1279] |
| logistic_regression | isotonic | 0.7441 [0.6924, 0.7931] | 0.7737 | 0.2082 [0.1850, 0.2325] | 0.0845 [0.0646, 0.1358] |
| lightgbm | uncalibrated | **0.8081** [0.7645, 0.8493] | **0.8621** | 0.1825 [0.1605, 0.2055] | 0.0770 [0.0629, 0.1213] |
| lightgbm | platt | 0.8081 [0.7645, 0.8493] | 0.8621 | 0.1806 [0.1577, 0.2040] | 0.0634 [0.0438, 0.1102] |
| lightgbm | isotonic | 0.8037 [0.7594, 0.8456] | 0.8406 | 0.1833 [0.1597, 0.2082] | **0.0618** [0.0461, 0.1113] |

95% CIs: 1,000 stratified bootstrap resamples on the test set.

**Best model+calibration:**
- **By AUROC**: `lightgbm` / uncalibrated (tied with platt, rank-preserving) — **0.8081** [0.7645, 0.8493]
- **By Brier**: `lightgbm` / platt — **0.1806** [0.1577, 0.2040]
- **By ECE**: `lightgbm` / isotonic — **0.0618** [0.0461, 0.1113]

At primary threshold 0.5 (chosen because this task's base rate is ~58%, unlike the low-prevalence mortality task): `lightgbm`/platt gives sensitivity 0.747, specificity 0.690, PPV 0.767, NPV 0.667, F1 0.757, balanced accuracy 0.718.

---

## 6. Clinical threshold table (`table_5_clinical_thresholds.csv`)

Thresholds evaluated: 0.3 (higher sensitivity) and 0.5 (primary, base-rate-matched). Full table in the CSV; `lightgbm`/uncalibrated at 0.3 gives sensitivity 0.828 / specificity 0.573 — a reasonable "rule-out" operating point if missed abnormalities are costlier than false alarms.

---

## 7. Figures and tables inventory

**Figures** (`demo_ptbxl_run/outputs/figures/`): `roc_curves_final.png/.pdf`, `pr_curves_final.png/.pdf`, `reliability_{model}_{method}.png/.pdf` (×6, validation set).

**Tables** (`demo_ptbxl_run/outputs/tables/`): `table_1_cohort.csv/.md`, `cohort_flow_counts.csv`, `table_2_hyperparameters.csv`, `table_3_calibration_metrics.csv`, `table_4_main_results.csv`, `table_5_clinical_thresholds.csv`.

---

## 8. Reproducibility

- Seed 42 throughout (subsampling, model training, bootstrap)
- Split: PTB-XL's own `strat_fold` (patient-stratified, verified zero cross-fold patient overlap in this subsample)
- Preprocessing: median imputation + missingness indicators (height, weight), one-hot encoding (gender), standard scaling
- Bootstrap: 1,000 stratified resamples

---

## 9–12. Data availability, ethics, manuscript structure, limitations

- **Data availability**: PTB-XL is public on PhysioNet under the ODC-BY license; no credentialing required. Cite: Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset," *Scientific Data*, 2020.
- **Ethics**: publicly released, de-identified data; no separate IRB required for secondary analysis of this public dataset (confirm against your institution's policy).
- **Limitations**: (1) this run used a 4,000-record subsample, not the full 21,799; (2) features are simple per-lead statistics, not clinically validated ECG features or a signal deep-learning model — treat AUROC values as a floor, not the achievable ceiling; (3) binary collapse of the 5-class superclass task is a simplification; (4) no external validation.
- **Recommended next step**: re-run against the full 21,799-record dataset and/or a proper ECG feature set (QRS duration, ST elevation/depression, T-wave morphology) or a 1D-CNN over raw waveforms, before submission.
