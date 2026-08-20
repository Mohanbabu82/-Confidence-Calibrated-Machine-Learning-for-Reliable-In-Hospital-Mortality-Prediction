**table3_calibration_metrics**

| model               | calibration_method   |   brier_score |   adaptive_ece |         mce |      nll |   calibration_intercept |   calibration_slope |   ece_10bins |   ece_15bins |
|:--------------------|:---------------------|--------------:|---------------:|------------:|---------:|------------------------:|--------------------:|-------------:|-------------:|
| lightgbm            | uncalibrated         |      0.1385   |    0.0972678   | 0.150752    | 0.446531 |             1.89302     |            3.47358  |  0.104148    |  0.103348    |
| lightgbm            | platt                |      0.126032 |    0.0288371   | 0.125359    | 0.393675 |            -0.00075471  |            0.999462 |  0.0154777   |  0.0223181   |
| lightgbm            | isotonic             |      0.122485 |    2.72467e-17 | 1.66533e-16 | 0.379379 |             7.937e-05   |            1.00012  |  3.72387e-17 |  3.37866e-17 |
| logistic_regression | uncalibrated         |      0.172356 |    0.217919    | 0.463902    | 0.514324 |            -1.47713     |            1.0216   |  0.217919    |  0.217919    |
| logistic_regression | platt                |      0.114302 |    0.0288447   | 0.171204    | 0.365207 |            -7.69395e-05 |            0.999849 |  0.0207468   |  0.0257973   |
| logistic_regression | isotonic             |      0.109561 |    2.38814e-17 | 1.66533e-16 | 0.348504 |            -0.000318644 |            0.999756 |  3.76551e-17 |  2.86114e-17 |

*Abbreviations:*

- **AUROC**: Area Under the Receiver Operating Characteristic curve

*Notes:*

- Synthetic dataset, N=5997
