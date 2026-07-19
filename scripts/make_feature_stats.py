"""Compute per-feature population mean/std from the training dataset.

Used by api/main.py to fill in a missing _pz (per-participant z-score)
feature on the fly from a raw value, when a /predict caller only supplies a
handful of raw signals (e.g. a manual-entry demo) instead of a full feature
row. This is a population-baseline approximation, not the participant's own
baseline -- documented as such in api/main.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path("data/processed/dataset.parquet")
DEFAULT_OUTPUT_PATH = Path("models/feature_stats.json")
REFERENCE_COLUMNS = {"id", "day_in_study", "study_interval", "label"}


def compute_feature_stats(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict:
    data_path = Path(data_path)
    output_path = Path(output_path)
    assert data_path.exists(), f"dataset not found: {data_path}"

    df = pd.read_parquet(data_path)
    raw_columns = [c for c in df.columns if c not in REFERENCE_COLUMNS and not c.endswith("_pz")]

    stats = {}
    for col in raw_columns:
        series = pd.to_numeric(df[col], errors="coerce")
        mean = series.mean()
        std = series.std()
        stats[col] = {
            "mean": None if pd.isna(mean) else float(mean),
            "std": None if pd.isna(std) else float(std),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2))
    print(f"[make_feature_stats] wrote {output_path} ({len(stats)} features)")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-feature mean/std for the on-the-fly _pz fallback")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()
    compute_feature_stats(data_path=args.data, output_path=args.output)


if __name__ == "__main__":
    main()
