# Confidence-Calibrated Machine Learning for Reliable In-Hospital Mortality Prediction

Research project scaffold for developing and rigorously **calibrating**
machine learning models that predict in-hospital mortality, with a focus
on reliable, well-calibrated probability estimates suitable for clinical
decision support.

> **No patient data included.** This repository contains no real,
> synthetic-generated, or downloaded medical data. All data paths in
> `configs/config.yaml` are placeholders — you must supply your own
> local, governance-approved, de-identified dataset before running
> anything.

## Project Structure

```
calibrated_clinical_ai/
├── data/
│   ├── raw/              # Placeholder location for raw local data (not included)
│   └── processed/        # Placeholder location for processed data (not included)
├── sql/                  # Placeholder cohort-extraction SQL
├── src/                  # Source code (data, models, calibration, evaluation, plots)
├── notebooks/            # Jupyter notebooks
├── outputs/
│   ├── tables/            # Generated result tables
│   └── figures/           # Generated plots (reliability diagrams, ROC/PR curves)
├── configs/
│   └── config.yaml        # Central configuration (paths, seeds, hyperparameters)
├── docs/                  # Methodology and documentation
├── tests/                 # Unit tests
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Setup

Requires **Python 3.11**.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Reproducibility

- A global random seed of **42** is enforced via
  `src/reproducibility.set_global_seed()` across Python `random`,
  NumPy, and PyTorch.
- Every run can log environment metadata (package versions, timestamp,
  seed) to `logs/run_metadata.json` via
  `src/reproducibility.log_run_metadata()`.
- Console + file logging is provided by
  `src/reproducibility.get_logger()`, writing to `logs/run.log`.

## Methodology

See [docs/methodology.md](docs/methodology.md) for the full pipeline:
extraction → preprocessing → modeling (logistic regression, random
forest, XGBoost, LightGBM, PyTorch MLP) → calibration (Platt scaling,
isotonic regression, temperature scaling) → evaluation (AUROC, AUPRC,
Brier score, ECE, MCE) → interpretability (SHAP).

## Using Your Own Data

1. Place your local, de-identified, governance-approved dataset at the
   path referenced by `configs/config.yaml -> paths.data_raw`.
2. Update `configs/config.yaml` (target column, ID column, model
   hyperparameters) as needed.
3. Run preprocessing (`src/preprocess.py`), then proceed through
   modeling, calibration, and evaluation.

## Testing

```bash
pytest tests/
```

## License

MIT (see `pyproject.toml`).
