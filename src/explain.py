"""SHAP explainability for the cycle-phase model.

Bulk mode (run_explain / CLI) computes SHAP values over an entire split and
saves global + per-class importance plots and a ranked-importance JSON.

explain_one() is the single-prediction entry point: self-contained (needs
only a model file path and a feature dict), with a per-model-path explainer
cache so repeated calls (e.g. from an API) don't reload the model each time.
Stage 8's agent/API layer reuses this function directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.splits import ID_COL, LABEL_NAMES

DEFAULT_MODEL_PATH = Path("models/lgbm.txt")
DEFAULT_DATA_PATH = Path("data/processed/dataset.parquet")
DEFAULT_SPLITS_PATH = Path("data/processed/splits.json")
DEFAULT_SPLIT_NAME = "test"
DEFAULT_GLOBAL_PLOT_PATH = Path("reports/shap_global.png")
DEFAULT_BY_CLASS_PLOT_PATH = Path("reports/shap_by_class.png")
DEFAULT_IMPORTANCE_JSON_PATH = Path("reports/shap_importance.json")

NUM_CLASSES = len(LABEL_NAMES)
LABEL_VALUES = sorted(LABEL_NAMES)

# Keyed by model_path so repeated calls with the same model reuse the loaded
# booster + explainer, but distinct model paths (e.g. across tests) don't
# collide with each other.
_EXPLAINER_CACHE: dict[str, tuple] = {}


def _get_explainer(model_path: str | Path = DEFAULT_MODEL_PATH) -> tuple:
    key = str(model_path)
    if key not in _EXPLAINER_CACHE:
        booster = lgb.Booster(model_file=key)
        feature_columns = booster.feature_name()
        explainer = shap.TreeExplainer(booster)
        _EXPLAINER_CACHE[key] = (explainer, booster, feature_columns)
    return _EXPLAINER_CACHE[key]


def _compute_shap_array(explainer: shap.TreeExplainer, X: pd.DataFrame, num_classes: int) -> np.ndarray:
    """Normalizes SHAP's output to shape (num_classes, n_samples, n_features),
    regardless of SHAP version / whether it returns a list-of-arrays or a
    single 3D array, and regardless of that array's axis order."""
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        return np.stack([np.asarray(a) for a in raw], axis=0)
    arr = np.asarray(raw)
    assert arr.ndim == 3, f"unexpected SHAP output shape: {arr.shape}"
    if arr.shape[0] == num_classes:
        return arr
    if arr.shape[-1] == num_classes:
        return np.transpose(arr, (2, 0, 1))
    raise AssertionError(f"cannot locate the class axis in SHAP output shape {arr.shape} "
                          f"for num_classes={num_classes}")


def explain_one(feature_dict: dict, model_path: str | Path = DEFAULT_MODEL_PATH, top_k: int = 5) -> list[dict]:
    """Top-`top_k` SHAP drivers for a SINGLE prediction, for the model's
    predicted class. Each item: {feature, shap, value, direction}, where
    direction is relative to the predicted phase's probability
    ("increases"/"decreases"/"neutral"). Missing features in feature_dict are
    treated as NaN (LightGBM handles missing values natively)."""
    explainer, booster, feature_columns = _get_explainer(model_path)

    row = pd.DataFrame([[feature_dict.get(col, np.nan) for col in feature_columns]], columns=feature_columns)
    proba = booster.predict(row)[0]
    predicted_class = int(np.argmax(proba))

    shap_array = _compute_shap_array(explainer, row, NUM_CLASSES)  # (num_classes, 1, n_features)
    shap_row = shap_array[predicted_class, 0, :]
    values = row.iloc[0].to_numpy()

    order = np.argsort(-np.abs(shap_row))[:top_k]
    drivers = []
    for idx in order:
        val = values[idx]
        val_is_missing = val is None or (isinstance(val, float) and np.isnan(val))
        shap_val = float(shap_row[idx])
        drivers.append({
            "feature": feature_columns[idx],
            "shap": shap_val,
            "value": None if val_is_missing else float(val),
            "direction": "increases" if shap_val > 0 else ("decreases" if shap_val < 0 else "neutral"),
        })
    return drivers


def _plot_global_importance(ranking: list[tuple[str, float]], out_path: Path, top_n: int = 20) -> None:
    top = ranking[:top_n]
    names = [name for name, _ in top][::-1]
    values = [value for _, value in top][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(names))))
    ax.barh(names, values, color="#3b7ea1")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"Global feature importance (top {len(top)} of {len(ranking)})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _plot_by_class_importance(feature_columns: list[str], mean_abs_by_class: dict[str, np.ndarray],
                               out_path: Path, top_n: int = 10) -> None:
    phases = list(LABEL_NAMES.values())
    fig, axes = plt.subplots(1, len(phases), figsize=(5 * len(phases), 6))
    for ax, phase in zip(axes, phases):
        arr = mean_abs_by_class[phase]
        order = np.argsort(-arr)[:top_n]
        names = [feature_columns[i] for i in order][::-1]
        values = [arr[i] for i in order][::-1]
        ax.barh(names, values, color="#8e5572")
        ax.set_title(phase)
        ax.set_xlabel("mean |SHAP|")
    fig.suptitle("Top drivers by predicted phase")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def run_explain(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    data_path: str | Path = DEFAULT_DATA_PATH,
    splits_path: str | Path = DEFAULT_SPLITS_PATH,
    split_name: str = DEFAULT_SPLIT_NAME,
    global_plot_path: str | Path = DEFAULT_GLOBAL_PLOT_PATH,
    by_class_plot_path: str | Path = DEFAULT_BY_CLASS_PLOT_PATH,
    importance_json_path: str | Path = DEFAULT_IMPORTANCE_JSON_PATH,
    top_n_plot: int = 20,
) -> dict:
    model_path = Path(model_path)
    data_path = Path(data_path)
    splits_path = Path(splits_path)
    global_plot_path = Path(global_plot_path)
    by_class_plot_path = Path(by_class_plot_path)
    importance_json_path = Path(importance_json_path)

    assert model_path.exists(), f"model not found: {model_path}"
    assert data_path.exists(), f"dataset not found: {data_path}"
    assert splits_path.exists(), f"splits not found: {splits_path}"

    df = pd.read_parquet(data_path)
    splits = json.loads(splits_path.read_text())
    assert split_name in splits, f"unknown split '{split_name}', expected one of {list(splits)}"

    booster = lgb.Booster(model_file=str(model_path))
    feature_columns = booster.feature_name()

    split_df = df[df[ID_COL].isin(splits[split_name])]
    assert len(split_df) > 0, f"split '{split_name}' has zero rows"
    X = split_df[feature_columns]
    print(f"[explain] split='{split_name}': {len(X)} rows, {len(feature_columns)} features")

    explainer = shap.TreeExplainer(booster)
    shap_array = _compute_shap_array(explainer, X, NUM_CLASSES)  # (num_classes, n_samples, n_features)

    mean_abs_global = np.abs(shap_array).mean(axis=(0, 1))
    global_ranking = sorted(zip(feature_columns, mean_abs_global), key=lambda t: -t[1])

    mean_abs_by_class = {
        LABEL_NAMES[c]: np.abs(shap_array[i]).mean(axis=0) for i, c in enumerate(LABEL_VALUES)
    }

    _plot_global_importance(global_ranking, global_plot_path, top_n=top_n_plot)
    print(f"[explain] wrote {global_plot_path}")
    _plot_by_class_importance(feature_columns, mean_abs_by_class, by_class_plot_path)
    print(f"[explain] wrote {by_class_plot_path}")

    importance_data = {
        "split": split_name,
        "n_samples": len(split_df),
        "global": {name: float(value) for name, value in global_ranking},
        "by_class": {
            phase: {
                name: float(value)
                for name, value in sorted(zip(feature_columns, arr), key=lambda t: -t[1])
            }
            for phase, arr in mean_abs_by_class.items()
        },
    }
    importance_json_path.parent.mkdir(parents=True, exist_ok=True)
    importance_json_path.write_text(json.dumps(importance_data, indent=2))
    print(f"[explain] wrote {importance_json_path}")

    top5 = global_ranking[:5]
    print("[explain] top-5 global drivers: " + ", ".join(f"{name}={value:.4f}" for name, value in top5))

    return importance_data


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP explainability for the mcPHASES cycle-phase model")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--split", default=DEFAULT_SPLIT_NAME)
    args = parser.parse_args()
    run_explain(model_path=args.model, data_path=args.data, splits_path=args.splits, split_name=args.split)


if __name__ == "__main__":
    main()
