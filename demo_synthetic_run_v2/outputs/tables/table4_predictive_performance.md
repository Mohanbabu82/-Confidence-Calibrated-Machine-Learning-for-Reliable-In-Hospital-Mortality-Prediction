**Table 4. Predictive performance comparison (test set, AUROC/AUPRC + operating-point metrics).**

| model               | calibration_method   |   n_test | auroc                  |    auprc |   sensitivity |   specificity |      ppv |      npv |   f1_score |   balanced_accuracy |
|:--------------------|:---------------------|---------:|:-----------------------|---------:|--------------:|--------------:|---------:|---------:|-----------:|--------------------:|
| logistic_regression | uncalibrated         |     1200 | 0.7982 (0.7670–0.8299) | 0.519731 |      0.939535 |      0.370558 | 0.245742 | 0.965608 |   0.389585 |            0.655047 |
| logistic_regression | platt                |     1200 | 0.7982 (0.7670–0.8299) | 0.519731 |      0.637209 |      0.788832 | 0.397101 | 0.908772 |   0.489286 |            0.713021 |
| logistic_regression | isotonic             |     1200 | 0.7964 (0.7650–0.8278) | 0.47779  |      0.6      |      0.819289 | 0.420195 | 0.903695 |   0.494253 |            0.709645 |
| lightgbm            | uncalibrated         |     1200 | 0.7639 (0.7296–0.7959) | 0.41612  |      0.934884 |      0.330964 | 0.233721 | 0.958824 |   0.373953 |            0.632924 |
| lightgbm            | platt                |     1200 | 0.7639 (0.7296–0.7959) | 0.41612  |      0.651163 |      0.738071 | 0.351759 | 0.906484 |   0.45677  |            0.694617 |
| lightgbm            | isotonic             |     1200 | 0.7638 (0.7295–0.7958) | 0.396087 |      0.674419 |      0.728934 | 0.351942 | 0.911168 |   0.46252  |            0.701676 |

*Abbreviations:*

- **AUROC**: Area Under the Receiver Operating Characteristic curve
- **AUPRC**: Area Under the Precision-Recall Curve
- **PPV**: Positive Predictive Value (Precision)
- **NPV**: Negative Predictive Value

*Notes:*

- 95% CI: 1000 stratified bootstrap resamples.
