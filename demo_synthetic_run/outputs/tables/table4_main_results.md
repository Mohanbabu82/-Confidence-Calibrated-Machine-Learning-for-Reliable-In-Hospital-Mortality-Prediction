**Table 4. Main results (test set).**

| model               | calibration_method   |   n_test | auroc                  | brier_score            | ece                    |     auprc |   primary_threshold |   sensitivity_at_primary_threshold |   specificity_at_primary_threshold |   ppv_at_primary_threshold |   npv_at_primary_threshold |   f1_score_at_primary_threshold |   balanced_accuracy_at_primary_threshold |
|:--------------------|:---------------------|---------:|:-----------------------|:-----------------------|:-----------------------|----------:|--------------------:|-----------------------------------:|-----------------------------------:|---------------------------:|---------------------------:|--------------------------------:|-----------------------------------------:|
| logistic_regression | uncalibrated         |       24 | 0.5455 (0.1818–0.9091) | 0.2376 (0.2300–0.2449) | 0.4019 (0.3947–0.4099) | 0.142157  |                 0.2 |                                  1 |                           0        |                  0.0833333 |                 nan        |                        0.153846 |                                 0.5      |
| logistic_regression | platt                |       24 | 0.5455 (0.1818–0.9091) | 0.2209 (0.1071–0.3553) | 0.2435 (0.1232–0.3851) | 0.142157  |                 0.2 |                                  0 |                           0.772727 |                  0         |                   0.894737 |                      nan        |                                 0.386364 |
| logistic_regression | isotonic             |       24 | 0.5795 (0.3182–0.9091) | 0.1822 (0.1062–0.2733) | 0.2237 (0.1195–0.3444) | 0.125     |                 0.2 |                                  0 |                           0.772727 |                  0         |                   0.894737 |                      nan        |                                 0.386364 |
| lightgbm            | uncalibrated         |       24 | 0.5000 (0.0909–0.9091) | 0.0877 (0.0685–0.1094) | 0.0632 (0.0252–0.1730) | 0.133333  |                 0.2 |                                  0 |                           0.863636 |                  0         |                   0.904762 |                      nan        |                                 0.431818 |
| lightgbm            | platt                |       24 | 0.5000 (0.0909–0.9091) | 0.0765 (0.0753–0.0777) | 0.0006 (0.0000–0.0032) | 0.133333  |                 0.2 |                                  0 |                           1        |                nan         |                   0.916667 |                      nan        |                                 0.5      |
| lightgbm            | isotonic             |       24 | 0.5227 (0.2045–0.8409) | 0.0808 (0.0672–0.0944) | 0.0655 (0.0060–0.1607) | 0.0871212 |                 0.2 |                                  0 |                           1        |                nan         |                   0.916667 |                      nan        |                                 0.5      |

*Abbreviations:*

- **AUROC**: Area Under the Receiver Operating Characteristic curve
- **AUPRC**: Area Under the Precision-Recall Curve
- **ECE**: Expected Calibration Error
- **CI**: Confidence Interval

*Notes:*

- 1000 stratified bootstrap resamples, 95% CI.
