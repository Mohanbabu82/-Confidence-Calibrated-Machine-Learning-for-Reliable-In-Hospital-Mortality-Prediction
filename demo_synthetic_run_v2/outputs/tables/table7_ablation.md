**Table 7. Feature-group ablation.**

| feature_configuration                    |   n_features | model               |   test_auroc |   test_brier |
|:-----------------------------------------|-------------:|:--------------------|-------------:|-------------:|
| Full model (all categories)              |           23 | logistic_regression |     0.798248 |     0.172148 |
| Full model (all categories)              |           23 | lightgbm            |     0.763929 |     0.13792  |
| Demographics only                        |            4 | logistic_regression |     0.579802 |     0.242431 |
| Demographics only                        |            4 | lightgbm            |     0.561863 |     0.156393 |
| Vital signs only                         |            6 | logistic_regression |     0.669616 |     0.222945 |
| Vital signs only                         |            6 | lightgbm            |     0.636475 |     0.150338 |
| Laboratory values only                   |           11 | logistic_regression |     0.72339  |     0.203852 |
| Laboratory values only                   |           11 | lightgbm            |     0.708176 |     0.143643 |
| Clinical interventions/output only       |            2 | logistic_regression |     0.525657 |     0.249618 |
| Clinical interventions/output only       |            2 | lightgbm            |     0.519129 |     0.158294 |
| Full minus Demographics                  |           19 | logistic_regression |     0.789673 |     0.179465 |
| Full minus Demographics                  |           19 | lightgbm            |     0.759828 |     0.139306 |
| Full minus Vital signs                   |           17 | logistic_regression |     0.732801 |     0.196601 |
| Full minus Vital signs                   |           17 | lightgbm            |     0.713009 |     0.142413 |
| Full minus Laboratory values             |           12 | logistic_regression |     0.69367  |     0.214329 |
| Full minus Laboratory values             |           12 | lightgbm            |     0.666587 |     0.147479 |
| Full minus Clinical interventions/output |           21 | logistic_regression |     0.798522 |     0.172883 |
| Full minus Clinical interventions/output |           21 | lightgbm            |     0.766741 |     0.137832 |

*Abbreviations:*

- **AUROC**: Area Under the Receiver Operating Characteristic curve

*Notes:*

- Hyperparameters fixed at the values selected for the full-feature model (logreg C=10.0; LightGBM learning_rate=0.01, 40 iterations); only the input feature set varies. Test set, single evaluation per configuration (no bootstrap CI).
