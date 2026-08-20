**Table 5. Calibration performance comparison (held-out test set).**

| model               | calibration_method   |   brier_score |   adaptive_ece |      mce |      nll |   calibration_intercept |   calibration_slope |   ece_10bins |   ece_15bins |
|:--------------------|:---------------------|--------------:|---------------:|---------:|---------:|------------------------:|--------------------:|-------------:|-------------:|
| lightgbm            | dwac                 |      0.181383 |      0.0561731 | 0.216474 | 0.545863 |               0.109248  |            0.675806 |    0.0560969 |    0.0681404 |
| lightgbm            | isotonic             |      0.183296 |      0.0613126 | 0.174612 | 0.601161 |               0.0783345 |            0.45748  |    0.0618497 |    0.0742416 |
| lightgbm            | platt                |      0.180645 |      0.0638385 | 0.161103 | 0.538737 |               0.106909  |            0.716093 |    0.0633746 |    0.0765057 |
| lightgbm            | uncalibrated         |      0.182516 |      0.0643337 | 0.149848 | 0.544102 |               0.353395  |            0.789876 |    0.0769512 |    0.0814128 |
| logistic_regression | dwac                 |      0.205804 |      0.0799415 | 0.163927 | 0.627022 |               0.0119023 |            0.53788  |    0.078454  |    0.0936722 |
| logistic_regression | isotonic             |      0.208181 |      0.0866747 | 0.255875 | 0.947621 |               0.0791339 |            0.14455  |    0.0844521 |    0.099606  |
| logistic_regression | platt                |      0.204754 |      0.0800465 | 0.175267 | 0.621948 |               0.0176376 |            0.555687 |    0.0801809 |    0.0875041 |
| logistic_regression | uncalibrated         |      0.204939 |      0.072767  | 0.141417 | 0.604444 |               0.269208  |            0.755048 |    0.0695943 |    0.0927676 |

*Abbreviations:*

- **ECE**: Expected Calibration Error
- **MCE**: Maximum Calibration Error
- **NLL**: Negative Log-Likelihood

*Notes:*

- Brier, ECE (10/15 bins), MCE, NLL, intercept, slope. Test set, computed once.
- dwac = Density-Weighted Adaptive Calibration, the one genuinely novel component in this pipeline (see Methods); Platt and isotonic are established, unmodified methods.
