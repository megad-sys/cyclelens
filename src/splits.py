"""Freeze the benchmark split: participant-grouped, majority-phase balanced.

Loads data/processed/dataset.parquet, splits PARTICIPANTS (never days) into
train/val/test (~70/15/15) via sklearn GroupShuffleSplit, run separately
within each "participant's majority phase" bucket so all four classes are
represented in every split. Writes data/processed/splits.json (id lists only
-- safe to open-source) and reports/benchmark_card.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DEFAULT_DATASET_PATH = Path("data/processed/dataset.parquet")
DEFAULT_SPLITS_PATH = Path("data/processed/splits.json")
DEFAULT_CARD_PATH = Path("reports/benchmark_card.md")

ID_COL = "id"
DAY_COL = "day_in_study"
LABEL_COL = "label"
REFERENCE_COLUMNS = {"id", "day_in_study", "study_interval", "label"}
LABEL_NAMES = {0: "Menstrual", 1: "Follicular", 2: "Fertility", 3: "Luteal"}

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
MIN_GROUP_SHUFFLE_SPLIT_SIZE = 4  # below this, fall back to a deterministic slice


def _majority_label_per_id(df: pd.DataFrame) -> pd.Series:
    return df.groupby(ID_COL)[LABEL_COL].agg(lambda s: s.value_counts().idxmax())


def _deterministic_slice(ids: np.ndarray, seed: int) -> tuple[list, list, list]:
    """Shuffle+slice a too-small-for-GroupShuffleSplit bucket of ids by fraction."""
    ids = list(ids)
    rng = np.random.RandomState(seed)
    shuffled = [ids[i] for i in rng.permutation(len(ids))]
    n = len(shuffled)
    n_train = min(max(1, round(TRAIN_FRAC * n)), n)
    n_val = min(round(VAL_FRAC * n), n - n_train)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def _split_bucket(bucket_df: pd.DataFrame, seed: int) -> tuple[set, set, set]:
    """Split one majority-phase bucket of participants into train/val/test."""
    ids = bucket_df[ID_COL].unique()
    if len(ids) < MIN_GROUP_SHUFFLE_SPLIT_SIZE:
        train, val, test = _deterministic_slice(ids, seed)
        return set(train), set(val), set(test)

    gss_train = GroupShuffleSplit(n_splits=1, train_size=TRAIN_FRAC, random_state=seed)
    train_idx, rest_idx = next(gss_train.split(bucket_df, groups=bucket_df[ID_COL]))
    train_ids = set(bucket_df.iloc[train_idx][ID_COL].unique())

    rest_df = bucket_df.iloc[rest_idx]
    rest_ids = rest_df[ID_COL].unique()
    if len(rest_ids) < 2:
        return train_ids, set(rest_ids), set()

    gss_val = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
    val_idx, test_idx = next(gss_val.split(rest_df, groups=rest_df[ID_COL]))
    val_ids = set(rest_df.iloc[val_idx][ID_COL].unique())
    test_ids = set(rest_df.iloc[test_idx][ID_COL].unique())
    return train_ids, val_ids, test_ids


def make_splits(df: pd.DataFrame, seed: int = SEED) -> dict[str, list[int]]:
    assert ID_COL in df.columns and LABEL_COL in df.columns, "dataset missing id/label columns"

    majority = _majority_label_per_id(df)
    train_ids: set = set()
    val_ids: set = set()
    test_ids: set = set()

    for label_value in sorted(majority.unique()):
        bucket_ids = majority[majority == label_value].index
        bucket_df = df[df[ID_COL].isin(bucket_ids)]
        b_train, b_val, b_test = _split_bucket(bucket_df, seed)
        train_ids |= b_train
        val_ids |= b_val
        test_ids |= b_test

    assert train_ids.isdisjoint(val_ids), "train/val id overlap"
    assert train_ids.isdisjoint(test_ids), "train/test id overlap"
    assert val_ids.isdisjoint(test_ids), "val/test id overlap"

    all_ids = set(df[ID_COL].unique())
    assigned = train_ids | val_ids | test_ids
    assert assigned == all_ids, f"ids not assigned to exactly one split: {all_ids - assigned}"

    return {
        "train": sorted(int(i) for i in train_ids),
        "val": sorted(int(i) for i in val_ids),
        "test": sorted(int(i) for i in test_ids),
    }


def _markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _split_summary(df: pd.DataFrame, splits: dict[str, list[int]]) -> dict[str, dict]:
    summary = {}
    for split_name, ids in splits.items():
        split_df = df[df[ID_COL].isin(ids)]
        counts = split_df[LABEL_COL].value_counts().sort_index()
        total = len(split_df)
        summary[split_name] = dict(
            n_participants=len(ids),
            n_days=total,
            counts={label: int(counts.get(label, 0)) for label in LABEL_NAMES},
        )
    return summary


def _write_benchmark_card(df: pd.DataFrame, splits: dict[str, list[int]], out_path: Path) -> None:
    feature_columns = [c for c in df.columns if c not in REFERENCE_COLUMNS]
    summary = _split_summary(df, splits)

    lines = [
        "# Benchmark Card",
        "",
        "## Task",
        "4-class daily cycle-phase prediction: given one day's wearable + self-report "
        "features for a participant (see reports/feature_dictionary.md), predict which "
        "hormone-validated menstrual-cycle phase that day falls in.",
        "",
        "## Label mapping",
        _markdown_table(["label", "phase"], [[k, v] for k, v in LABEL_NAMES.items()]),
        "",
        f"## Feature list ({len(feature_columns)} features)",
        ", ".join(f"`{c}`" for c in feature_columns),
        "",
        "## Split methodology",
        f"Participant-grouped (sklearn `GroupShuffleSplit` on `id`, seed={SEED}), target "
        f"{int(TRAIN_FRAC*100)}/{int(VAL_FRAC*100)}/{int(TEST_FRAC*100)} train/val/test. "
        "To keep all four classes represented in every split, participants are first bucketed "
        "by their own majority (most frequent) phase, and the group split is run independently "
        "within each bucket before recombining. No participant ID appears in more than one split "
        f"(buckets with fewer than {MIN_GROUP_SHUFFLE_SPLIT_SIZE} participants use a deterministic "
        "seeded shuffle+slice instead of GroupShuffleSplit, which requires enough groups per side "
        "to split meaningfully).",
        "",
        "## Split sizes",
        _markdown_table(
            ["split", "n participants", "n participant-days"],
            [[name, summary[name]["n_participants"], summary[name]["n_days"]] for name in ["train", "val", "test"]],
        ),
        "",
        "## Class balance per split",
        _markdown_table(
            ["split", *LABEL_NAMES.values()],
            [
                [
                    name,
                    *[
                        f"{summary[name]['counts'][label]} "
                        f"({100 * summary[name]['counts'][label] / summary[name]['n_days']:.1f}%)"
                        if summary[name]["n_days"] else "0 (n/a)"
                        for label in LABEL_NAMES
                    ],
                ]
                for name in ["train", "val", "test"]
            ],
        ),
        "",
        "## Evaluation metric",
        "- **Primary**: macro-F1 (unweighted mean of per-class F1; treats all 4 phases equally "
        "regardless of their class imbalance).",
        "- **Secondary**: per-class F1, balanced accuracy.",
        "",
        "## Baselines the model must beat",
        "1. **Global majority class**: fit the single most frequent label on TRAIN, predict it "
        "for every row of the evaluation split.",
        "2. **Personal-majority**: for each evaluation day, predict that participant's own most "
        "common phase computed leave-one-out over their OTHER days within the same split "
        "(excluding the day being predicted). This isolates how much signal comes purely from "
        "knowing a person's typical phase distribution, independent of that day's biosignals.",
        "3. **Calendar heuristic (day-within-cycle)**: NOT implemented in this stage -- see "
        "Future Work below.",
        "",
        "## Future work: calendar heuristic baseline",
        "A day-within-cycle heuristic (e.g. bucket days by time-since-last-menstruation-onset, "
        "predict the historically most common phase for that bucket) was not added here because "
        "it cannot be derived cleanly from the current tables: there is no explicit cycle-number "
        "or cycle-start column, `day_in_study` is a fixed calendar index (not reset per cycle), "
        "and study_interval has a real multi-month gap between Interval 1 (2022) and Interval 2 "
        "(2024) that would break a naive day-since-onset count for participants who span both. "
        "A robust version would need to: (1) detect menstruation-onset days per participant as "
        "`phase == Menstrual` where the previous LABELED day's phase != Menstrual, (2) compute "
        "day-within-cycle relative to the most recent onset, resetting across the Interval 1/2 "
        "gap, and (3) bin day-within-cycle and take the historical majority phase per bin from "
        "TRAIN only. That is real feature-engineering work, not a naive baseline, so it is left "
        "for a follow-up stage rather than forced in here.",
        "",
        "## Correctness invariant",
        "Every split is at the PARTICIPANT level: train/val/test share zero participant ids. "
        "Enforced by `tests/test_splits.py`.",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[splits] wrote {out_path}")


def build_splits(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    splits_path: str | Path = DEFAULT_SPLITS_PATH,
    card_path: str | Path = DEFAULT_CARD_PATH,
    seed: int = SEED,
) -> dict[str, list[int]]:
    dataset_path = Path(dataset_path)
    splits_path = Path(splits_path)
    card_path = Path(card_path)
    assert dataset_path.exists(), f"dataset not found: {dataset_path}"

    df = pd.read_parquet(dataset_path)
    print(f"[splits] loaded {dataset_path}: shape={df.shape}")

    splits = make_splits(df, seed=seed)
    for name in ("train", "val", "test"):
        n_days = int(df[ID_COL].isin(splits[name]).sum())
        print(f"[splits] {name}: {len(splits[name])} participants, {n_days} participant-days")

    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(splits, indent=2))
    print(f"[splits] wrote {splits_path}")

    _write_benchmark_card(df, splits, card_path)

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the participant-grouped benchmark split")
    parser.add_argument("--data", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--splits-out", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--card-out", default=str(DEFAULT_CARD_PATH))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    build_splits(dataset_path=args.data, splits_path=args.splits_out, card_path=args.card_out, seed=args.seed)


if __name__ == "__main__":
    main()
