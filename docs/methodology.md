# Methodology

## Objective

Develop and rigorously calibrate machine learning models for predicting
in-hospital mortality, with an emphasis on the reliability of predicted
probabilities (not just discrimination) — a requirement for models used
to support clinical decisions.

## Data

This project does not include, download, or generate any real patient
data. All paths in `configs/config.yaml` are placeholders. Users must
supply their own governance-approved, de-identified dataset locally
before running the pipeline.

## Pipeline

1. **Extraction** (`sql/`) — placeholder cohort-extraction query.
2. **Loading** (`src/data_loader.py`) — reads raw/processed data from
   configured local paths.
3. **Preprocessing** (`src/preprocess.py`) — cleaning, imputation,
   encoding, stratified train/val/test split (seed=42).
4. **Modeling** (`src/models.py`) — logistic regression, random forest,
   XGBoost, LightGBM, and a PyTorch MLP baseline.
5. **Calibration** (`src/calibration.py`) — Platt scaling, isotonic
   regression, and temperature scaling (for the neural network).
6. **Evaluation** (`src/evaluate.py`) — AUROC, AUPRC, Brier score,
   log loss, Expected Calibration Error (ECE), Maximum Calibration
   Error (MCE).
7. **Visualization** (`src/plots.py`) — reliability diagrams, ROC and
   PR curves.
8. **Interpretability** — SHAP values for feature attribution (to be
   run in `notebooks/`).

## Reproducibility

All random number generators (Python `random`, NumPy, PyTorch) are
seeded with `42` via `src/reproducibility.set_global_seed()`. Every
run additionally logs environment/package versions and a timestamp to
`logs/run_metadata.json` via `src/reproducibility.log_run_metadata()`.
