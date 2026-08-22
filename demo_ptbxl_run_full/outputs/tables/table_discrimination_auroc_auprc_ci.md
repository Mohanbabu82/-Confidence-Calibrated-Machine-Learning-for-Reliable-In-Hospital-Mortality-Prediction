**AUROC and AUPRC with 95% stratified bootstrap confidence intervals, per model x calibration method (held-out test set, n=2158, full 21,388-record dataset).**

| model               | calibration_method   |   n_test | auroc                  | auprc                  |
|:--------------------|:---------------------|---------:|:-----------------------|:-----------------------|
| lightgbm            | dwac                 |     2158 | 0.8464 (0.8304–0.8618) | 0.8932 (0.8809–0.9043) |
| lightgbm            | isotonic             |     2158 | 0.8457 (0.8291–0.8610) | 0.8822 (0.8691–0.8941) |
| lightgbm            | platt                |     2158 | 0.8464 (0.8305–0.8618) | 0.8932 (0.8809–0.9043) |
| lightgbm            | uncalibrated         |     2158 | 0.8464 (0.8305–0.8618) | 0.8932 (0.8809–0.9043) |
| logistic_regression | dwac                 |     2158 | 0.7953 (0.7762–0.8146) | 0.8480 (0.8311–0.8636) |
| logistic_regression | isotonic             |     2158 | 0.7933 (0.7744–0.8121) | 0.8333 (0.8164–0.8496) |
| logistic_regression | platt                |     2158 | 0.7954 (0.7761–0.8142) | 0.8480 (0.8311–0.8636) |
| logistic_regression | uncalibrated         |     2158 | 0.7954 (0.7761–0.8142) | 0.8480 (0.8311–0.8636) |

*Abbreviations:*

- **AUROC**: Area Under the Receiver Operating Characteristic curve
- **AUPRC**: Area Under the Precision-Recall Curve
- **CI**: Confidence Interval

*Notes:*

- 1000 stratified bootstrap resamples (resampled within each outcome class, preserving observed class counts), seed=42.
- AUROC/AUPRC are numerically identical across Platt/DWAC/uncalibrated for a given model where the calibration transform preserves rank order; isotonic regression can introduce small AUROC/AUPRC differences due to tied outputs in flat regions of the fitted step function.
