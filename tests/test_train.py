import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.train import train_model


def _build_synthetic_dataset(n_ids: int = 10, days_per_id: int = 30, seed: int = 0) -> pd.DataFrame:
    """~300 rows across 10 fake participants, with signal_feature strongly
    correlated with the label (separated by 10 units, noise std=1) so the
    model has a clearly learnable signal, plus two uninformative noise
    features so it's not a single-column trivial fit."""
    rng = np.random.RandomState(seed)
    rows = []
    for pid in range(1, n_ids + 1):
        for day in range(1, days_per_id + 1):
            label = (day - 1) % 4
            rows.append({
                "id": pid,
                "day_in_study": day,
                "study_interval": 1,
                "label": label,
                "signal_feature": label * 10.0 + rng.normal(0, 1.0),
                "noise_feature_1": rng.normal(0, 5.0),
                "noise_feature_2": rng.uniform(0, 1.0),
            })
    return pd.DataFrame(rows)


def _write_fixture(tmp_path):
    dataset = _build_synthetic_dataset()
    data_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(data_path, index=False)

    all_ids = sorted(int(i) for i in dataset["id"].unique())
    splits = {
        "train": all_ids[:6],
        "val": all_ids[6:8],
        "test": all_ids[8:10],
    }
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits))

    return dataset, data_path, splits, splits_path


def test_train_model_creates_model_and_beats_majority_baseline(tmp_path):
    dataset, data_path, splits, splits_path = _write_fixture(tmp_path)

    model_out_path = tmp_path / "models" / "lgbm.txt"
    metrics_out_path = tmp_path / "reports" / "train_metrics.json"

    metrics = train_model(
        data_path=data_path,
        splits_path=splits_path,
        model_out_path=model_out_path,
        metrics_out_path=metrics_out_path,
        seed=42,
        num_boost_round=100,
        early_stopping_rounds=20,
    )

    # (a) models/*.txt is created
    assert model_out_path.exists()
    assert metrics_out_path.exists()

    # (b) macro-F1 on the held-out (val) split beats the global-majority baseline
    train_df = dataset[dataset["id"].isin(splits["train"])]
    val_df = dataset[dataset["id"].isin(splits["val"])]
    global_majority_label = train_df["label"].value_counts().idxmax()
    baseline_pred = np.full(len(val_df), global_majority_label)
    baseline_macro_f1 = f1_score(val_df["label"], baseline_pred, average="macro", zero_division=0)

    assert metrics["val_macro_f1"] > baseline_macro_f1
    assert metrics["val_macro_f1"] > 0.8  # the injected signal is easily separable
