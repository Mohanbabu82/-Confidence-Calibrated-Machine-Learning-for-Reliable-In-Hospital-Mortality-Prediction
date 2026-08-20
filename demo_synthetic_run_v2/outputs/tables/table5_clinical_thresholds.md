**table5_clinical_thresholds**

| model               | calibration_method   |   threshold |   sensitivity |   specificity |        ppv |      npv |   f1_score |   balanced_accuracy |
|:--------------------|:---------------------|------------:|--------------:|--------------:|-----------:|---------:|-----------:|--------------------:|
| logistic_regression | uncalibrated         |         0.2 |      0.939535 |      0.370558 |   0.245742 | 0.965608 |   0.389585 |            0.655047 |
| logistic_regression | uncalibrated         |         0.5 |      0.693023 |      0.767513 |   0.39418  | 0.919708 |   0.50253  |            0.730268 |
| logistic_regression | platt                |         0.2 |      0.637209 |      0.788832 |   0.397101 | 0.908772 |   0.489286 |            0.713021 |
| logistic_regression | platt                |         0.5 |      0.32093  |      0.958376 |   0.627273 | 0.866055 |   0.424615 |            0.639653 |
| logistic_regression | isotonic             |         0.2 |      0.6      |      0.819289 |   0.420195 | 0.903695 |   0.494253 |            0.709645 |
| logistic_regression | isotonic             |         0.5 |      0.32093  |      0.958376 |   0.627273 | 0.866055 |   0.424615 |            0.639653 |
| lightgbm            | uncalibrated         |         0.2 |      0.934884 |      0.330964 |   0.233721 | 0.958824 |   0.373953 |            0.632924 |
| lightgbm            | uncalibrated         |         0.5 |      0        |      1        | nan        | 0.820833 | nan        |            0.5      |
| lightgbm            | platt                |         0.2 |      0.651163 |      0.738071 |   0.351759 | 0.906484 |   0.45677  |            0.694617 |
| lightgbm            | platt                |         0.5 |      0.134884 |      0.977665 |   0.568627 | 0.83812  |   0.218045 |            0.556274 |
| lightgbm            | isotonic             |         0.2 |      0.674419 |      0.728934 |   0.351942 | 0.911168 |   0.46252  |            0.701676 |
| lightgbm            | isotonic             |         0.5 |      0.153488 |      0.974619 |   0.568966 | 0.84063  |   0.241758 |            0.564054 |

*Abbreviations:*

- **AUROC**: Area Under the Receiver Operating Characteristic curve

*Notes:*

- Synthetic dataset, N=5997
