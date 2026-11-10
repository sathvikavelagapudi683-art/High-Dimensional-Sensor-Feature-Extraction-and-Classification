from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .data import EEGDataset
from .models import build_models


def _fold_features(X_train, y_train, X_valid, meta_train, meta_valid):
    """Fit all EEG feature transforms on train data only and apply to validation."""
    cov = XdawnCovariances(nfilter=3, estimator="scm")
    cov_train = cov.fit_transform(X_train, y_train)
    cov_valid = cov.transform(X_valid)

    tangent = TangentSpace(metric="riemann")
    eeg_train = tangent.fit_transform(cov_train)
    eeg_valid = tangent.transform(cov_valid)

    combined_train = np.concatenate([eeg_train, meta_train], axis=1)
    combined_valid = np.concatenate([eeg_valid, meta_valid], axis=1)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(combined_train)
    valid_scaled = scaler.transform(combined_valid)
    return train_scaled, valid_scaled


def _metrics(y_true, probability):
    predicted = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
    }


def run_grouped_cv(dataset: EEGDataset, output_dir: str | Path, n_splits: int = 4, random_state: int = 42):
    """Run grouped OOF evaluation and save metrics, fold metrics, and probabilities."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splitter = GroupKFold(n_splits=n_splits)
    model_names = list(build_models(random_state).keys())
    oof = {name: np.full(len(dataset.labels), np.nan, dtype=float) for name in model_names}
    fold_rows = []

    splits = list(splitter.split(dataset.epochs, dataset.labels, dataset.groups))
    for fold_number, (train_idx, valid_idx) in enumerate(splits, start=1):
        X_train, X_valid = dataset.epochs[train_idx], dataset.epochs[valid_idx]
        y_train, y_valid = dataset.labels[train_idx], dataset.labels[valid_idx]
        if len(np.unique(y_train)) != 2:
            raise ValueError(f"Fold {fold_number} training set does not contain both classes.")
        if len(np.unique(y_valid)) != 2:
            raise ValueError(
                f"Fold {fold_number} validation set does not contain both classes; ROC-AUC is undefined. "
                "Review the group assignments or use a different grouped split design."
            )

        train_features, valid_features = _fold_features(
            X_train,
            y_train,
            X_valid,
            dataset.metadata[train_idx],
            dataset.metadata[valid_idx],
        )
        for name, model in build_models(random_state).items():
            model.fit(train_features, y_train)
            probability = model.predict_proba(valid_features)[:, 1]
            oof[name][valid_idx] = probability
            fold_rows.append({"fold": fold_number, "model": name, **_metrics(y_valid, probability)})

    if any(not np.isfinite(values).all() for values in oof.values()):
        raise RuntimeError("Some epochs did not receive out-of-fold predictions.")

    summary_rows = []
    prediction_frame = pd.DataFrame({"y_true": dataset.labels, "group": dataset.groups.astype(str)})
    for name, probability in oof.items():
        summary_rows.append({"model": name, **_metrics(dataset.labels, probability)})
        prediction_frame[name + "__probability_class_1"] = probability

    pd.DataFrame(summary_rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    prediction_frame.to_csv(output / "oof_predictions.csv", index=False)
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)
