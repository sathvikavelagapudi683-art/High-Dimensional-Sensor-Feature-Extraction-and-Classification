from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EEGDataset:
    epochs: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    metadata: np.ndarray


def load_dataset(data_dir: str | Path) -> EEGDataset:
    """Load aligned epoch, label, group, and metadata files from ``data_dir``."""
    root = Path(data_dir)
    required = ["X_epochs.npy", "y.npy", "groups.npy", "metadata.csv"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required input(s) in {root}: {', '.join(missing)}. "
            "See docs/DATA_FORMAT.md for the expected layout."
        )

    epochs = np.load(root / "X_epochs.npy", allow_pickle=False)
    labels = np.load(root / "y.npy", allow_pickle=False).reshape(-1)
    groups = np.load(root / "groups.npy", allow_pickle=False).reshape(-1)
    table = pd.read_csv(root / "metadata.csv")

    expected_columns = ["Session", "Feedback_Progress"]
    absent = [name for name in expected_columns if name not in table.columns]
    if absent:
        raise ValueError(f"metadata.csv is missing columns: {', '.join(absent)}")

    try:
        metadata = table[expected_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Session and Feedback_Progress must be numeric; encode them before running.") from exc

    n = len(labels)
    if epochs.ndim != 3:
        raise ValueError(f"X_epochs.npy must be 3-D (epochs, channels, time); got {epochs.shape}")
    if epochs.shape[0] != n or len(groups) != n or metadata.shape[0] != n:
        raise ValueError(
            "Input row counts do not match: "
            f"epochs={epochs.shape[0]}, labels={n}, groups={len(groups)}, metadata={metadata.shape[0]}"
        )
    if not np.isfinite(epochs).all() or not np.isfinite(metadata).all():
        raise ValueError("EEG epochs and metadata must contain only finite numeric values.")
    classes = np.unique(labels)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(f"Expected binary labels encoded as 0 and 1; found {classes.tolist()}")
    if len(np.unique(groups)) < 4:
        raise ValueError("Four-fold GroupKFold requires at least four distinct groups.")

    return EEGDataset(epochs=epochs, labels=labels.astype(int), groups=groups, metadata=metadata)
