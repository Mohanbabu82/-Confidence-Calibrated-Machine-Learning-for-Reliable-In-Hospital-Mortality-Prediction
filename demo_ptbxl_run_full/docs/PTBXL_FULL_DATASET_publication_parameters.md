# PTB-XL Full-Dataset Run — Publication Parameters

## Status update on the previously stated limitation

`demo_ptbxl_run/docs/PTBXL_publication_parameters.md` (the earlier, 4,000-record
subsample run) stated: *"This run used a stratified, seed-42, N=4,000 subsample
of the full 21,799-record dataset … not the full dataset"* (limitation #1),
with recommended next step *"re-run against the full 21,799-record dataset …
before submission."*

**This directory (`demo_ptbxl_run_full/`) is that re-run.** N=21,388 (all
records with a parseable diagnostic superclass out of 21,799 total — the
same 411-record exclusion the subsample run also applied, for the same
reason: no usable label). No subsampling was performed. The earlier
4,000-record run is left untouched in `demo_ptbxl_run/` for provenance and
comparison, per "do not alter existing results."

---

## What changed vs. the 4,000-record subsample

| | Subsample (n=4,000) | **Full dataset (n=21,388)** |
|---|---|---|
| Train / val / test | 3,195 / 401 / 404 | **17,084 / 2,146 / 2,158** |
| Overall abnormal rate | 57.60% | 57.60% (subsample was representative) |
| LightGBM val AUROC | 0.7836 | **0.8695** |
| Logistic regression val AUROC | 0.8163 | **0.8144** (essentially unchanged) |
| LightGBM test AUROC (uncalibrated) | 0.808 | **0.846** |
| Logistic regression test AUROC (uncalibrated) | 0.750 | **0.795** |
| Model comparison (DeLong) | p=0.00012 | **p≈0** (z=-7.91, far more powered) |
| DWAC vs uncalibrated Brier, either model | not significant (CI included 0) | **significant** (CI excludes 0, both models) |
| Best AURC method | LightGBM/Platt (0.1414) | **LightGBM/DWAC (0.1080)** |
| Ablation: "Demographics only" vs full (logreg) | not significant | **significant** (more test power resolves it) |

**Real, direct consequence of using the full dataset**: LightGBM's AUROC rose
from 0.808 to 0.846, addressing roadmap MAJOR weakness #4 ("weak features;
AUROC 0.808 below benchmarks") — not by richer features (a Class-C/D item,
still not done), but simply by removing the subsampling that was discarding
81% of available real training data.

---

## Updated Table 1: Cohort (full dataset)

- N = 21,388, abnormal = 12,319 (57.60%)
- Train (folds 1-8): n=17,084, abnormal 57.60%
- Validation (fold 9): n=2,146, abnormal 57.41%
- Test (fold 10): n=2,158, abnormal 57.74%

Full breakdown: `outputs/tables/table_1_cohort.csv`

**Note on `gender` encoding**: this cohort was built directly from
`ptbxl_database.csv`'s raw `sex` field (0/1), not remapped to M/F strings
as the earlier subsample run did. Per PTB-XL's documented convention,
0=male, 1=female. `table_sex_subgroup.csv` uses these raw codes; relabel
before publication.

## Updated Table 2: Hyperparameters

- Logistic regression: C=0.01 (val AUROC 0.8144)
- LightGBM: learning_rate=0.05, best_iteration=300 (val AUROC 0.8695)

**Caveat worth disclosing**: LightGBM's early stopping (patience=20) did
not trigger at either candidate learning rate within the 300-tree cap —
both hit the ceiling. This means the model may still be improving with
more trees; the grid was kept identical to the subsample run's
methodology (same `n_estimators` cap, same learning-rate grid) rather than
re-tuned post hoc. A wider `n_estimators` ceiling is a reasonable
follow-up, not performed here to keep this a faithful re-run rather than
a new tuning experiment.

## Updated Table 5: Calibration performance (test set, n=2,158)

| Model | Method | Brier | ECE (10 bins) | MCE | NLL |
|---|---|---|---|---|---|
| lightgbm | dwac | 0.1583 | 0.0327 | 0.0852 | 0.4781 |
| lightgbm | platt | 0.1585 | 0.0291 | 0.1044 | 0.4777 |
| lightgbm | isotonic | 0.1586 | 0.0362 | 0.0995 | 0.4914 |
| lightgbm | uncalibrated | 0.1606 | 0.0515 | 0.0969 | 0.4848 |
| logistic_regression | platt | 0.1820 | 0.0194 | 0.0631 | 0.5388 |
| logistic_regression | dwac | 0.1824 | 0.0257 | 0.0694 | 0.5398 |
| logistic_regression | isotonic | 0.1828 | 0.0291 | 0.1667 | 0.5449 |
| logistic_regression | uncalibrated | 0.1854 | 0.0576 | 0.0949 | 0.5469 |

Full: `outputs/tables/table5_calibration_performance_test.csv`

## Updated Table 7: Ablation with bootstrap CIs (test set, n=2,158)

With the larger, better-powered test set, ablation conclusions **changed**
from the subsample run:

- **"Demographics only" is now significantly worse than the full model for
  BOTH algorithms** (logreg diff CI [-0.056, -0.026]; lightgbm diff CI
  [-0.120, -0.086]). On the subsample, logreg's demographics-only config
  was *not* significantly different from full — that was a real bootstrap
  finding but an underpowered one; the full dataset resolves it.
- **"Full minus Precordial leads" is now significantly worse than full for
  LightGBM** (diff CI [-0.042, -0.021]) but still **not significant for
  logistic regression** (diff CI [-0.013, 0.004]) — a genuine
  model-dependent finding, not an artifact.

Full: `outputs/tables/table7_ablation_with_bootstrap_ci.csv`

## Statistical validation summary (all real, full dataset)

- **DeLong**: LightGBM significantly outperforms logistic regression,
  z=-7.91, **p≈0** (far more decisive than the subsample's p=0.00012)
- **Paired bootstrap**: model-comparison AUROC/Brier/selective-risk
  differences all exclude zero; DWAC-vs-uncalibrated Brier differences now
  also exclude zero for both models (not detectable in the smaller subsample)
- **AURC**: LightGBM/DWAC now best (0.1080), an improvement over the
  subsample's LightGBM/Platt best (0.1414) — DWAC's advantage becomes
  visible with more data
- **Subgroups**: sex gap remains modest (LightGBM AUROC 0.852 vs 0.844,
  M vs F by raw code); age-band pattern persists — degraded AUROC in
  the youngest (<40, AUROC 0.62-0.64) and a highly imbalanced 80-89
  band (93% positive); `>89 (anonymized)` band n=32 in this larger test
  set (vs n=5 before), still correctly age-flagged rather than treated as
  a normal continuous value

---

## What is STILL not done (genuinely out of scope for a recomputation)

Per the roadmap's Class-C/D distinction, these still require new
experiments/data, not just more compute on existing features:
- Native multi-class superclass task (currently binary NORM-vs-abnormal)
- Richer ECG features or a 1D-CNN/deep-learning baseline on raw signal
- Prospective/external validation

The "4,000-record subsample" limitation specifically is now resolved.

---

## Addendum: AUPRC + per-row AUROC bootstrap CIs (added on request)

New table: `outputs/tables/table_discrimination_auroc_auprc_ci.csv` (+`.md`/`.tex`).
For every model x calibration-method row (8 total), reports AUROC and AUPRC
with 95% stratified bootstrap CIs (1000 resamples, seed=42), computed from
`test_predictions_all.parquet` — no new modeling, same real test-set
predictions used everywhere else in this run.

Sanity check: the full-model rows match the previously-reported ablation-table
CIs exactly (LightGBM uncalibrated 0.8464 [0.8305, 0.8618]; logistic
regression uncalibrated 0.7954 [0.7761, 0.8142]) — no drift between the two
computations.

AUROC/AUPRC are numerically identical across uncalibrated/Platt/DWAC within
a model (these calibration methods preserve rank order); isotonic regression
shows tiny differences (e.g. LightGBM AUROC 0.8457 vs 0.8464) because its
step-function output ties adjacent scores in flat regions, a real and
expected property of isotonic regression, not an error.

## Addendum: sex-coding convention — NOT independently verifiable from local files

I checked every documentation file bundled inside the PTB-XL v1.0.3 zip
itself (`LICENSE.txt`, `example_physionet.py`, `ptbxl_v102_changelog.txt`,
`ptbxl_v103_changelog.txt`) for an explicit sex-encoding statement — **none
of them state it**. The `0=Male, 1=Female` convention is documented on
PhysioNet's PTB-XL project page and in Wagner et al. (2020, *Scientific
Data*), but that source is external to the files available in this
project. The relabeling should be treated as "widely documented
convention, not locally verified" rather than "confirmed" until checked
against physionet.org/content/ptb-xl/1.0.3 directly.
