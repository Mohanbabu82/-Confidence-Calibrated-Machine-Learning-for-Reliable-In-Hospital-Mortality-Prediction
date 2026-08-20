**Table 5. Clinical thresholds (test set).**

| model               | calibration_method   |   threshold |   sensitivity |   specificity |         ppv |        npv |   f1_score |   balanced_accuracy |
|:--------------------|:---------------------|------------:|--------------:|--------------:|------------:|-----------:|-----------:|--------------------:|
| logistic_regression | uncalibrated         |         0.2 |           1   |      0        |   0.0833333 | nan        |   0.153846 |            0.5      |
| logistic_regression | uncalibrated         |         0.5 |           0.5 |      0.772727 |   0.166667  |   0.944444 |   0.25     |            0.636364 |
| logistic_regression | platt                |         0.2 |           0   |      0.772727 |   0         |   0.894737 | nan        |            0.386364 |
| logistic_regression | platt                |         0.5 |           0   |      0.818182 |   0         |   0.9      | nan        |            0.409091 |
| logistic_regression | isotonic             |         0.2 |           0   |      0.772727 |   0         |   0.894737 | nan        |            0.386364 |
| logistic_regression | isotonic             |         0.5 |           0   |      0.818182 |   0         |   0.9      | nan        |            0.409091 |
| lightgbm            | uncalibrated         |         0.2 |           0   |      0.863636 |   0         |   0.904762 | nan        |            0.431818 |
| lightgbm            | uncalibrated         |         0.5 |           0   |      1        | nan         |   0.916667 | nan        |            0.5      |
| lightgbm            | platt                |         0.2 |           0   |      1        | nan         |   0.916667 | nan        |            0.5      |
| lightgbm            | platt                |         0.5 |           0   |      1        | nan         |   0.916667 | nan        |            0.5      |
| lightgbm            | isotonic             |         0.2 |           0   |      1        | nan         |   0.916667 | nan        |            0.5      |
| lightgbm            | isotonic             |         0.5 |           0   |      1        | nan         |   0.916667 | nan        |            0.5      |

*Abbreviations:*

- **PPV**: Positive Predictive Value (Precision)
- **NPV**: Negative Predictive Value

*Notes:*

- Thresholds: 0.2, 0.5.
