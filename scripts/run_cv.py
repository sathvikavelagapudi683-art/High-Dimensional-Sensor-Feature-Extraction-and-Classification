from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p300_pipeline.data import load_dataset  # noqa: E402
from p300_pipeline.evaluation import run_grouped_cv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run four-fold grouped CV for the P300 EEG project.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "run")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        dataset = load_dataset(args.data_dir)
        print(f"Loaded {len(dataset.labels):,} epochs | shape={dataset.epochs.shape} | groups={len(set(dataset.groups))}")
        metrics, _ = run_grouped_cv(dataset, args.output_dir, n_splits=args.folds, random_state=args.seed)
        print("\nOut-of-fold metrics:")
        print(metrics.to_string(index=False))
        print(f"\nSaved outputs to: {args.output_dir}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
