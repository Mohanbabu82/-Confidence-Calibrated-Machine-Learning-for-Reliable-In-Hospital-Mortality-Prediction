**Table 3. Calibration metrics (validation).**

| model               | calibration_method   |   brier_score |   adaptive_ece |         mce |         nll |   calibration_intercept |   calibration_slope |   ece_10bins |   ece_15bins |
|:--------------------|:---------------------|--------------:|---------------:|------------:|------------:|------------------------:|--------------------:|-------------:|-------------:|
| lightgbm            | uncalibrated         |   0.0822911   |    0.146125    | 0.232588    | 0.302973    |              -2.19262   |           0.0951985 |  0.0454027   |  0.112082    |
| lightgbm            | platt                |   0.0763874   |    0.124948    | 1.99547e-05 | 0.286712    |              -0.355042  |           0.852233  |  1.99547e-05 |  1.99547e-05 |
| lightgbm            | isotonic             |   0.0714286   |    1.61908e-17 | 2.77556e-17 | 0.239235    |              -0.100431  |           0.944021  |  1.61908e-17 |  1.61908e-17 |
| logistic_regression | uncalibrated         |   0.23616     |    0.405445    | 0.484675    | 0.665438    |              -6.59645   |         175.291     |  0.405445    |  0.405445    |
| logistic_regression | platt                |   1.01832e-06 |    5.45606e-05 | 0.00264714  | 0.000434311 |              -0.0543133 |           1.62686   |  0.000433469 |  0.000433469 |
| logistic_regression | isotonic             |   0           |    0           | 0           | 1e-06       |              -0.0602093 |           0.998186  |  0           |  0           |

*Abbreviations:*

- **ECE**: Expected Calibration Error
- **MCE**: Maximum Calibration Error
- **NLL**: Negative Log-Likelihood

*Notes:*

- In-sample calibration.
