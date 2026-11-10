# Reproducibility and scope

## What is included

- A clean Python implementation of the covariance/tangent-space and metadata-fusion workflow described in the methodology presentation.
- Four-fold grouped out-of-fold evaluation for Elastic Net logistic regression, RBF SVM, and Random Forest.
- Separate historical result records from the project notes and methodology deck.

## What is not included

- Original INRIA BCI data files, extracted EEG arrays, labels, group IDs, or participant/session metadata.
- The exact notebook/script that produced the résumé-highlighted ROC-AUC 0.6704 result.
- Saved classifiers, fitted transforms, or predictions from the original run.

The implementation should be treated as a transparent reproduction scaffold for the documented covariance/tangent-space workflow, not proof that all historical figures can be regenerated exactly.

## Validation cautions

- Confirm that `groups.npy` encodes the independence unit intended by the study. If the goal is cross-participant generalization, group by participant.
- Keep feature fitting, tangent-space reference estimation, and scaling inside each training fold. The included code does this.
- Hyperparameters in the methodology slides are the tested settings, not evidence of exhaustive optimization.
- Evaluate the final workflow on a separately held-out test set before making deployment or generalization claims.
