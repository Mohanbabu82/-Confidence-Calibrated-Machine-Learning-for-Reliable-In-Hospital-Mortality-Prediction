**Table 7. Feature-group ablation (lead groups + demographics).**

| feature_configuration                        |   n_features | model               |   test_auroc |   test_brier |
|:---------------------------------------------|-------------:|:--------------------|-------------:|-------------:|
| Full model (all leads + demographics)        |           64 | logistic_regression |     0.749667 |     0.204939 |
| Full model (all leads + demographics)        |           64 | lightgbm            |     0.808147 |     0.182516 |
| Limb leads (I,II,III,aVR,aVL,aVF) only       |           30 | logistic_regression |     0.689607 |     0.222655 |
| Limb leads (I,II,III,aVR,aVL,aVF) only       |           30 | lightgbm            |     0.732952 |     0.210938 |
| Precordial leads (V1-V6) only                |           30 | logistic_regression |     0.628115 |     0.241281 |
| Precordial leads (V1-V6) only                |           30 | lightgbm            |     0.721683 |     0.219253 |
| Demographics (age,height,weight,gender) only |            4 | logistic_regression |     0.731609 |     0.213189 |
| Demographics (age,height,weight,gender) only |            4 | lightgbm            |     0.707    |     0.222599 |
| Full minus Limb leads                        |           34 | logistic_regression |     0.719047 |     0.215674 |
| Full minus Limb leads                        |           34 | lightgbm            |     0.761137 |     0.205208 |
| Full minus Precordial leads                  |           34 | logistic_regression |     0.763547 |     0.199156 |
| Full minus Precordial leads                  |           34 | lightgbm            |     0.790001 |     0.189121 |
| Full minus Demographics                      |           60 | logistic_regression |     0.695279 |     0.224022 |
| Full minus Demographics                      |           60 | lightgbm            |     0.779384 |     0.194538 |

*Abbreviations:*

- **AUROC**: Area Under the Receiver Operating Characteristic curve

*Notes:*

- Hyperparameters fixed at the values selected for the full-feature model (logreg C=0.01; LightGBM learning_rate=0.05, 100 iterations); only the input feature set varies. Test set, single evaluation per configuration (no bootstrap CI).
