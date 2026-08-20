**Table 2. Hyperparameters.**

| model               | hyperparameter           | value    |
|:--------------------|:-------------------------|:---------|
| logistic_regression | penalty                  | l2       |
| logistic_regression | solver                   | lbfgs    |
| logistic_regression | C                        | 0.001    |
| logistic_regression | max_iter                 | 1000     |
| logistic_regression | class_weight             | balanced |
| logistic_regression | random_state             | 42       |
| logistic_regression | global_random_seed       | 42       |
| logistic_regression | val_auroc                | 1.0      |
| logistic_regression | runtime_seconds          | 0.04     |
| lightgbm            | learning_rate            | 0.1      |
| lightgbm            | n_estimators_upper_bound | 300      |
| lightgbm            | best_iteration           | 7        |
| lightgbm            | num_leaves               | 31       |
| lightgbm            | max_depth                | -1       |
| lightgbm            | early_stopping_rounds    | 20       |
| lightgbm            | scale_pos_weight         | 11.0     |
| lightgbm            | random_state             | 42       |
| lightgbm            | global_random_seed       | 42       |
| lightgbm            | val_auroc                | 0.5455   |
| lightgbm            | runtime_seconds          | 0.29     |

*Abbreviations:*

- **AUROC**: Area Under the Receiver Operating Characteristic curve

*Notes:*

- Selected on validation AUROC.
