import json

import numpy as np
import pandas as pd

from src.explain import _EXPLAINER_CACHE, explain_one, run_explain
from src.train import train_model


def _build_synthetic_dataset(n_ids: int = 10, days_per_id: int = 30, seed: int = 0) -> pd.DataFrame:
    """Same injected-signal fixture as tests/test_train.py: signal_feature is
    strongly correlated with the label, plus two uninformative noise features."""
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
    splits = {"train": all_ids[:6], "val": all_ids[6:8], "test": all_ids[8:10]}
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits))

    model_path = tmp_path / "models" / "lgbm.txt"
    metrics_path = tmp_path / "reports" / "train_metrics.json"
    train_model(
        data_path=data_path, splits_path=splits_path, model_out_path=model_path,
        metrics_out_path=metrics_path, seed=42, num_boost_round=100, early_stopping_rounds=20,
    )

    return dataset, data_path, splits_path, model_path


def test_explain_one_returns_ranked_driver_list(tmp_path):
    _EXPLAINER_CACHE.clear()
    _, data_path, splits_path, model_path = _write_fixture(tmp_path)

    feature_dict = {"signal_feature": 32.0, "noise_feature_1": 0.5, "noise_feature_2": 0.3}
    drivers = explain_one(feature_dict, model_path=model_path, top_k=5)

    assert isinstance(drivers, list)
    assert len(drivers) > 0
    for driver in drivers:
        assert set(driver.keys()) == {"feature", "shap", "value", "direction"}
        assert driver["direction"] in {"increases", "decreases", "neutral"}

    # the injected signal feature should dominate the ranking
    assert drivers[0]["feature"] == "signal_feature"


def test_explain_one_handles_missing_features(tmp_path):
    _EXPLAINER_CACHE.clear()
    _, data_path, splits_path, model_path = _write_fixture(tmp_path)

    drivers = explain_one({"signal_feature": 5.0}, model_path=model_path, top_k=3)
    assert len(drivers) > 0
    missing_feature_drivers = [d for d in drivers if d["feature"] != "signal_feature"]
    assert all(d["value"] is None for d in missing_feature_drivers)


def test_run_explain_writes_importance_json_and_plots(tmp_path):
    _EXPLAINER_CACHE.clear()
    _, data_path, splits_path, model_path = _write_fixture(tmp_path)

    global_plot_path = tmp_path / "reports" / "shap_global.png"
    by_class_plot_path = tmp_path / "reports" / "shap_by_class.png"
    importance_json_path = tmp_path / "reports" / "shap_importance.json"

    result = run_explain(
        model_path=model_path, data_path=data_path, splits_path=splits_path, split_name="test",
        global_plot_path=global_plot_path, by_class_plot_path=by_class_plot_path,
        importance_json_path=importance_json_path,
    )

    assert importance_json_path.exists()
    assert global_plot_path.exists()
    assert by_class_plot_path.exists()

    saved = json.loads(importance_json_path.read_text())
    assert saved["global"] == result["global"]
    assert len(saved["global"]) == 3  # signal_feature + 2 noise features
    assert "signal_feature" in saved["global"]
    # the injected signal should dominate every other feature's importance
    top_feature = max(saved["global"], key=saved["global"].get)
    assert top_feature == "signal_feature"
