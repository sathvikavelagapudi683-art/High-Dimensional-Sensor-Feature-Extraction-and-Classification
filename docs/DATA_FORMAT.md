# Input data format

The cross-validation script expects four aligned files in the chosen data directory (default: `data/processed/`).

| File | Expected content | Shape / columns |
|---|---|---|
| `X_epochs.npy` | Epoch-major EEG array | `(n_epochs, n_channels, n_times)`; project record: `(5440, 56, 260)` |
| `y.npy` | Binary class labels | `(n_epochs,)`, encoded as `0` and `1` |
| `groups.npy` | Group key for each epoch | `(n_epochs,)`; ideally participant IDs for subject-independent validation |
| `metadata.csv` | Two numeric metadata columns | `Session`, `Feedback_Progress`, one row per epoch |

Every file must refer to the same epoch ordering. Do not shuffle one file independently.

The pipeline deliberately requires `Session` and `Feedback_Progress` as numeric fields, matching the methodology record. If the source metadata stores these as text, encode them consistently before running and document the encoding. Do not silently substitute row numbers for missing metadata.

The pipeline does not include raw data, subject identifiers, or participant metadata in this public repository. Obtain the dataset through its official research distribution and place local copies in the expected folder.
