# Experiment record

This document preserves the reported metric snapshots as separate records because the available materials describe more than one feature/classifier experiment.

## A. Feature-engineering experiment highlighted in the résumé

The project notes record a best grouped-validation result of:

| Metric | Reported value |
|---|---:|
| ROC-AUC | 0.6704 |
| Balanced accuracy | 0.6271 |
| Macro-F1 | 0.6114 |

The working notes describe feature matrices of 560, 896, and 1,456 columns across successive feature-engineering variants, covariance features with shape `(5440, 56, 56)`, and xDAWN-informed extraction. The recorded best score is retained as a historical result; the original exact run script and split artefacts were not recovered with the public project package.

## B. Covariance / tangent-space methodology-deck benchmark

A separate methodology presentation describes the following pipeline:

1. Four-fold `GroupKFold` using `groups.npy`.
2. Fit xDAWN covariance estimation on each training fold (3 filters, SCM estimator).
3. Fit the Riemannian tangent-space reference on training-fold covariance matrices.
4. Append `Session` and `Feedback_Progress` metadata.
5. Fit `StandardScaler` on the training fold only and apply it unchanged to the validation fold.
6. Collect out-of-fold class-1 probabilities.

The deck records these results:

| Model | Accuracy | Balanced accuracy | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Elastic Net logistic regression (`C=0.03`) | 0.6575 | 0.6454 | 0.6243 | 0.6959 |
| RBF SVM (`C=1.0`) | 0.7276 | 0.6102 | 0.6177 | 0.7079 |
| Random forest (300 trees, min leaf 5) | 0.7276 | 0.5860 | 0.5864 | 0.6783 |

Elastic Net had the best balanced accuracy and macro-F1 in this benchmark; RBF SVM had the best ROC-AUC. These scores are stored in `results/methodology_deck_metrics.csv`.

## Interpretation

The two sections are different experiment records. The feature-engineering result in Section A is the result highlighted for the project profile. Section B is the model-comparison snapshot described by the methodology presentation. The available source material does not establish that these are the same data split, same feature matrix, or same final run. Therefore no cross-section ranking or improvement claim is made.
