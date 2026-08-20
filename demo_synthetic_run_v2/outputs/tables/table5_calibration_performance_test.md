**Table 5. Calibration performance comparison (held-out test set).**

| model               | calibration_method   |   brier_score |   adaptive_ece |       mce |      nll |   calibration_intercept |   calibration_slope |   ece_10bins |   ece_15bins |
|:--------------------|:---------------------|--------------:|---------------:|----------:|---------:|------------------------:|--------------------:|-------------:|-------------:|
| lightgbm            | isotonic             |      0.126309 |      0.0301169 | 0.0666667 | 0.449377 |              -0.705747  |            0.370642 |    0.0306928 |    0.031888  |
| lightgbm            | platt                |      0.126041 |      0.0285365 | 0.0798856 | 0.403222 |              -0.0760335 |            0.879568 |    0.021957  |    0.0286412 |
| lightgbm            | uncalibrated         |      0.13792  |      0.103635  | 0.197491  | 0.445433 |               1.58854   |            3.05447  |    0.0907674 |    0.104477  |
| logistic_regression | isotonic             |      0.117901 |      0.0328584 | 0.461806  | 0.388401 |              -0.45262   |            0.639892 |    0.0313811 |    0.0347206 |
| logistic_regression | platt                |      0.116973 |      0.0249736 | 0.185195  | 0.377023 |              -0.144969  |            0.862456 |    0.0324001 |    0.0317168 |
| logistic_regression | uncalibrated         |      0.172148 |      0.207193  | 0.482196  | 0.515824 |              -1.41898   |            0.880599 |    0.207193  |    0.207193  |
| lightgbm            | dwac                 |      0.126118 |      0.0325123 | 0.0639158 | 0.405833 |              -0.19067   |            0.784452 |    0.0293018 |    0.0297612 |
| logistic_regression | dwac                 |      0.116977 |      0.0267957 | 0.249968  | 0.377727 |              -0.189334  |            0.825593 |    0.0294395 |    0.0291494 |

*Abbreviations:*

- **ECE**: Expected Calibration Error
- **MCE**: Maximum Calibration Error
- **NLL**: Negative Log-Likelihood

*Notes:*

- Brier, ECE (10/15 bins), MCE, NLL, intercept, slope. Test set, computed once.
- dwac = Density-Weighted Adaptive Calibration, the one genuinely novel component in this pipeline (see Methods); Platt and isotonic are established, unmodified methods.
