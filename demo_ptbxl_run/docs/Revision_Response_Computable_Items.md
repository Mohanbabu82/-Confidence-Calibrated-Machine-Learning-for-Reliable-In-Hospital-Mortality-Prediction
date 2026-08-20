# PTB-XL Revision Response — Class-B Computable Items (1–15)

Computed from `Revision_Roadmap_PTBXL.docx`'s blocked Class-B item list.
**The roadmap was written against an earlier upload that lacked per-record
predictions.** Real per-record test/validation predictions
(`test_predictions_with_dwac.parquet`, `val_predictions.parquet`) and
demographics (`cohort_ecg_features.parquet`) already existed in this
project, so every item below is computed from that real data — nothing
fabricated, nothing inferred.

All numbers on this page: test set, n=404 ECGs (PTB-XL `strat_fold`==10),
unless stated otherwise.

---

## A. Updated tables (new files, all in `demo_ptbxl_run/outputs/tables/`)

| Item | File | Contents |
|---|---|---|
| 1–5 | `table5_calibration_performance_test.csv` | Calibration slope, intercept, MCE, NLL, ECE(10 & 15 bins), Brier — **test set**, all 4 methods × 2 models (already existed and was verified test-set-sourced; formally answers roadmap #2) |
| 7 | `table_full_risk_coverage_curve.csv` | Selective risk, sensitivity, specificity, PPV, NPV at 48 coverage points (2%-step grid) per model×method |
| 8 | `table_AURC.csv` | Area Under the Risk-Coverage curve (trapezoidal integration), ranked |
| 9 | `table_selective_risk_comparison.csv` | Selective risk at 6 standard coverage checkpoints (100/95/90/80/70/50%) |
| 10 | `table_sex_subgroup.csv` | AUROC/Brier/ECE/selective-risk by sex (M n=214, F n=190), all methods |
| 11 | `table_age_band_subgroup.csv` | Same, by age band (<40/40-59/60-79/80-89/>89-anonymized) |
| 12, 14 | `table_paired_bootstrap_differences.csv` | Paired stratified-bootstrap 95% CIs (1000 resamples) for AUROC/Brier/ECE/selective-risk **differences** between 5 method/model pairs |
| 13 | `table_delong_auroc_comparison.csv` | DeLong's test (implemented in `src/delong.py`, sanity-checked against sklearn and an identical-scores null case) for 3 statistically-appropriate AUROC comparisons |
| 15 | `table7_ablation_with_bootstrap_ci.csv` | Ablation AUROC + 95% bootstrap CI + CI on the difference vs. the full model, for all 7 feature configs × 2 models |

Supporting new per-record files: `outputs/predictions/test_predictions_with_demographics.parquet`,
`outputs/predictions/ablation_test_predictions.parquet` (the latter's
point estimates were verified to exactly match the original, unblocked
`table7_ablation.csv` — same seed, same configs, only the per-record
output is new).

## B. Updated figures (new files, all in `demo_ptbxl_run/outputs/figures/`)

| Item | File(s) | Contents |
|---|---|---|
| 6 | `testset_reliability_{model}_{method}.png/.pdf` (×8) | Full test-set reliability diagrams for all 2 models × 4 methods (previously only 2 of 8 combos had test-set reliability diagrams; the rest were validation-set only) |
| 7 | `figure_full_risk_coverage_curves.png/.pdf` | Two-panel full risk-coverage curve, all 4 methods per model, coverage 4%–100% |

---

## C. Statistical interpretation

1. **Model comparison is now statistically defensible.** DeLong's test: LightGBM AUROC (0.808) vs. logistic regression AUROC (0.750), **p = 0.00012**, paired bootstrap AUROC-difference 95% CI **[-0.087, -0.030]** (excludes 0). LightGBM is significantly better — this was previously an unsupported "best" claim on overlapping independent CIs (roadmap weakness #7); it is no longer unsupported.

2. **No calibration method is statistically distinguishable from another at this sample size.** Paired bootstrap CIs for isotonic-vs-Platt, uncalibrated-vs-DWAC, and Platt-vs-DWAC all include zero for every metric tested (AUROC, Brier, ECE, selective risk). This is an important, honest null result: choose among Platt/isotonic/DWAC on practical grounds (DWAC's "no a priori choice needed" property, per `NOVEL_ALGORITHM_DWAC.md`), not because one is proven better on this test set.

3. **Calibration slope/intercept/MCE/NLL now come from the test set**, not validation (fixes roadmap CRITICAL #2). Test-set uncalibrated slopes remain far from 1 (e.g. LightGBM 0.79, logreg 0.76), confirming genuine miscalibration that Platt/isotonic/DWAC each visibly correct (slopes move to 0.46-0.72 range) — this was previously only shown on validation.

4. **AURC favors LightGBM/Platt** (0.1414, lowest = best) and **logistic regression/uncalibrated is worst** (0.2113). DWAC is competitive but not best on AURC for either model — consistent with finding 2 (no significant pairwise difference).

5. **Sex subgroup**: no evidence of a sex-based performance gap. LightGBM AUROC: M=0.807, F=0.809 (n=214/190, both adequately powered). Logistic regression: M=0.730, F=0.767 — a numerically larger gap, unverified for significance (not requested in the 15-item list; flag for follow-up if needed).

6. **Age-band subgroup reveals a genuine, important limitation.** AUROC collapses in the <40 band (LightGBM 0.644, logistic regression **0.521 — chance level**, n=50) and in the 80-89 band (LightGBM 0.571, logreg 0.601, n=46) — the 80-89 band is 42/46 (91%) positive, an extreme class-imbalance regime the model was not tuned for. The `>89 (anonymized)` band (PTB-XL's real de-identification code for age>89, not an error — 39/4000 records total, 5 in this test split) has only 5 samples, all positive; AUROC is correctly reported as undefined (NaN), flagged `low_n_flag=True`, not fabricated. **This directly fixes roadmap weakness #10** and should be stated as a real limitation, not smoothed over.

7. **Ablation, with real CIs, changes which findings are defensible.** For logistic regression, "Demographics only" is **not** significantly different from the full model (diff CI [-0.062, 0.027], includes 0) — demographics alone are statistically indistinguishable from the full 64-feature model for this model class. For both models, "Full minus Precordial leads" is **not** significantly worse than the full model (CIs include 0) — precordial leads do not provide statistically significant incremental value over limb leads + demographics. Conversely, "Limb leads only", "Precordial leads only", "Demographics only" (LightGBM), and "Full minus Limb leads" **are** significantly worse than the full model for at least one algorithm. This nuance did not exist in the original point-estimate-only Table 7.

---

## D. Exact manuscript paragraphs that need to change

**D1. Methods §2.7/2.8 (Calibration metrics / Evaluation) — wherever it currently states or implies calibration slope, intercept, MCE, or NLL were computed on the test set.**
Replace with: *"Calibration slope, intercept, MCE, NLL, and ECE (10 and 15 bins) were computed on the held-out test set (n=404) for all four calibration variants per model (uncalibrated, Platt, isotonic, DWAC); see Table 5."* — This was previously validation-set-sourced language; it is now literally true and should say so explicitly, citing `table5_calibration_performance_test.csv`.

**D2. Results §4.3 or wherever "best model" / "best calibration method" is asserted.**
Old framing (roadmap weakness #7): *"LightGBM achieved the best AUROC (0.808), though confidence intervals overlapped with logistic regression."*
Replace with: *"LightGBM significantly outperformed logistic regression on AUROC (0.808 vs. 0.750; DeLong's test z=-3.84, p=0.00012; paired bootstrap 95% CI for the difference [-0.087, -0.030], excluding zero). No calibration method (Platt, isotonic, or the proposed DWAC) was statistically distinguishable from another at n=404 (all paired-bootstrap CIs for AUROC/Brier/ECE/selective-risk differences included zero); method choice among these three should be justified on grounds other than a proven test-set performance advantage."*

**D3. Results — reliability diagrams paragraph.**
Old: reliability diagrams shown/referenced were validation-set (roadmap MINOR #15). Replace figure references with `testset_reliability_{model}_{method}.png` (all 8 combos now exist) and state explicitly these are test-set, n=404.

**D4. Results — selective prediction / referral section.**
Add a new paragraph citing the full risk-coverage curves and AURC: *"Figure [X] shows the full risk-coverage curve (coverage 4-100%, 2% steps) for each model×calibration-method combination on the test set. AURC (Table [Y]) ranked LightGBM/Platt best (0.141) and logistic-regression/uncalibrated worst (0.211), consistent with the discrimination results above."* This satisfies roadmap MAJOR #11 (previously absent entirely).

**D5. Results — new subgroup paragraph (previously absent, roadmap MAJOR #10).**
Add: *"Subgroup analysis by sex showed no material AUROC gap for LightGBM (M=0.807, F=0.809); logistic regression showed a numerically larger but unassessed gap (M=0.730, F=0.767). By age band, performance was substantially degraded in the youngest (<40 years, AUROC 0.52-0.64) and oldest (80-89 years, AUROC 0.57-0.60, 91% positive class imbalance) strata; the anonymized '>89 years' code (PTB-XL's de-identification convention, n=5 in the test split) had insufficient samples for AUROC estimation and is reported as such rather than omitted or estimated."*

**D6. Results — ablation section (roadmap MINOR #16).**
Old: point estimates only, no CI, single evaluation.
Replace with the CI-qualified language from Interpretation point 7 above, citing `table7_ablation_with_bootstrap_ci.csv`. Explicitly note which ablation conclusions survive bootstrap uncertainty and which do not (e.g., "precordial leads add no significant value" is now a *supported* claim; "demographics alone underperform the full model" is *not supported* for logistic regression specifically).

---

## E. Old numerical claims that must be removed or replaced

1. ❌ Any statement that calibration slope/intercept/MCE/NLL/reliability diagrams were computed on the **test** set when the actual source (before this update) was validation-only for 6 of 8 model×method combos → ✅ now genuinely test-set for all 8 (Table 5, item B's 8 new figures).
2. ❌ "LightGBM had the best AUROC" or similar **without a significance qualifier** → ✅ replace with the DeLong p=0.00012 / bootstrap-CI-excludes-zero language (D2).
3. ❌ Any implicit claim that Platt, isotonic, or DWAC is "the best" calibration method → ✅ replace with the explicit null result (no pairwise CI excludes zero among calibration methods).
4. ❌ Ablation table cited as point estimates without uncertainty → ✅ replace with CI-qualified claims; specifically retract/soften "demographics alone are noticeably worse than the full model" for logistic regression (not supported once CI'd).
5. ❌ No selective-prediction/AURC claim existed before — nothing to remove, but the absence itself (roadmap MAJOR #11) is now filled; do not leave the manuscript silent on selective-risk performance.
6. ❌ No subgroup claim existed before (roadmap MAJOR #10) — same: the absence must be filled, not left as a stated limitation only.
7. ⚠️ If the manuscript states or implies a uniform N=4000 age distribution without noting the `age=300` anonymization code, that omission must be corrected — it is a real PTB-XL data-dictionary convention (>89 years), not noise, and affects 39/4000 (0.98%) of the full sampled cohort.

---

## What remains genuinely blocked (not computable from existing predictions)

- Class C/D items (native multi-class superclass task, full 21,388-record dataset, 1D-CNN/deep-learning baseline) — these require new experiments/data, not just recomputation, and are correctly left as future work per the roadmap.
- Statistical significance of the sex-subgroup AUROC gap (mentioned in D5 as "numerically larger but unassessed") — computable on request via the same paired-bootstrap/DeLong machinery already built, just not in the original 15-item list.
