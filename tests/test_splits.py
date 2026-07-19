import numpy as np
import pandas as pd

from src.splits import make_splits


def _make_participant_rows(pid: int, majority_label: int, n_days: int = 10) -> pd.DataFrame:
    """n_days rows for one participant: 60% majority_label, rest spread over the others."""
    other_labels = [label for label in (0, 1, 2, 3) if label != majority_label]
    n_majority = round(0.6 * n_days)
    labels = [majority_label] * n_majority
    for i in range(n_days - n_majority):
        labels.append(other_labels[i % len(other_labels)])
    return pd.DataFrame({
        "id": pid,
        "day_in_study": range(1, n_days + 1),
        "label": labels,
    })


def _build_synthetic_dataset() -> pd.DataFrame:
    frames = []
    pid = 1

    # Large buckets (>= MIN_GROUP_SHUFFLE_SPLIT_SIZE): exercise the real GroupShuffleSplit path.
    for _ in range(20):
        frames.append(_make_participant_rows(pid, majority_label=3))  # Luteal-majority
        pid += 1
    for _ in range(6):
        frames.append(_make_participant_rows(pid, majority_label=1))  # Follicular-majority
        pid += 1

    # Small buckets (< MIN_GROUP_SHUFFLE_SPLIT_SIZE): exercise the deterministic-slice fallback.
    for _ in range(3):
        frames.append(_make_participant_rows(pid, majority_label=0))  # Menstrual-majority
        pid += 1
    for _ in range(2):
        frames.append(_make_participant_rows(pid, majority_label=2))  # Fertility-majority
        pid += 1

    return pd.concat(frames, ignore_index=True)


def test_splits_have_zero_id_overlap_and_full_coverage():
    df = _build_synthetic_dataset()
    all_ids = set(df["id"].unique())

    splits = make_splits(df, seed=42)
    train_ids, val_ids, test_ids = set(splits["train"]), set(splits["val"]), set(splits["test"])

    # CRITICAL property: zero overlap between any two splits.
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Every id lands in exactly one split.
    assert train_ids | val_ids | test_ids == all_ids
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(all_ids)

    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        n_days = int(df["id"].isin(ids).sum())
        print(f"[test_splits] {split_name}: {len(ids)} participants, {n_days} participant-days")

    assert len(train_ids) > 0
    assert len(val_ids) > 0
    assert len(test_ids) > 0


def test_splits_are_seed_deterministic():
    df = _build_synthetic_dataset()
    splits_a = make_splits(df, seed=42)
    splits_b = make_splits(df, seed=42)
    assert splits_a == splits_b
