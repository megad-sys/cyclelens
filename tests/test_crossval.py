import numpy as np
import pandas as pd

from src.crossval import cross_validate


def _build_synthetic_dataset(n_ids: int = 20, days_per_id: int = 20, seed: int = 0) -> pd.DataFrame:
    """~400 rows across 20 fake participants -- enough for 5-fold GroupKFold
    (4 held-out participants/fold) plus a grouped inner train/val carve-out
    within each fold's training partition. signal_feature is strongly
    correlated with the label so the model should clearly beat majority."""
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


def test_crossval_runs_five_folds_and_beats_global_majority(tmp_path):
    dataset = _build_synthetic_dataset()
    data_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(data_path, index=False)

    output_path = tmp_path / "reports" / "crossval.json"

    results = cross_validate(
        data_path=data_path,
        output_path=output_path,
        n_splits=5,
        seed=42,
        num_boost_round=100,
        early_stopping_rounds=20,
    )

    assert output_path.exists()
    assert len(results["folds"]) == 5
    for fold in results["folds"]:
        assert fold["n_held_out_rows"] > 0

    assert results["summary"]["model"]["mean_macro_f1"] > results["summary"]["global_majority"]["mean_macro_f1"]
    assert results["summary"]["model"]["mean_macro_f1"] > 0.7  # strong injected signal
