"""Train the LightGBM multiclass cycle-phase baseline.

Loads the frozen benchmark split (src/splits.py), trains on TRAIN with
class-balanced sample weights, early-stops on VAL, and saves the model plus
train/val macro-F1 to reports/train_metrics.json. LightGBM handles NaNs
natively, so no rows are dropped for missing features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

from src.splits import ID_COL, LABEL_COL, REFERENCE_COLUMNS

DEFAULT_DATA_PATH = Path("data/processed/dataset.parquet")
DEFAULT_SPLITS_PATH = Path("data/processed/splits.json")
DEFAULT_MODEL_OUT_PATH = Path("models/lgbm.txt")
DEFAULT_METRICS_OUT_PATH = Path("reports/train_metrics.json")

DEFAULT_SEED = 42
DEFAULT_NUM_BOOST_ROUND = 1000
DEFAULT_EARLY_STOPPING_ROUNDS = 50
NUM_CLASSES = 4

# Regularized for ~28 training participants (Stage 4b): shallow trees, large
# min_child_samples, L1/L2, feature/bagging subsampling, slow learning rate.
# Shared with src/crossval.py so cross-validation trains the SAME model.
# `seed` is intentionally excluded here -- callers merge it in per training run.
LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": NUM_CLASSES,
    "metric": "multi_logloss",
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
    "num_leaves": 15,
    "min_child_samples": 40,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "learning_rate": 0.03,
}


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in REFERENCE_COLUMNS]


def train_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    splits_path: str | Path = DEFAULT_SPLITS_PATH,
    model_out_path: str | Path = DEFAULT_MODEL_OUT_PATH,
    metrics_out_path: str | Path = DEFAULT_METRICS_OUT_PATH,
    seed: int = DEFAULT_SEED,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
) -> dict:
    data_path = Path(data_path)
    splits_path = Path(splits_path)
    model_out_path = Path(model_out_path)
    metrics_out_path = Path(metrics_out_path)

    assert data_path.exists(), f"dataset not found: {data_path}"
    assert splits_path.exists(), f"splits not found: {splits_path}"

    df = pd.read_parquet(data_path)
    splits = json.loads(splits_path.read_text())
    assert "train" in splits and "val" in splits, "splits.json missing 'train' or 'val'"

    feature_cols = _feature_columns(df)
    train_df = df[df[ID_COL].isin(splits["train"])]
    val_df = df[df[ID_COL].isin(splits["val"])]
    assert len(train_df) > 0, "train split has zero rows"
    assert len(val_df) > 0, "val split has zero rows"
    print(f"[train] train: {len(train_df)} rows ({train_df[ID_COL].nunique()} participants); "
          f"val: {len(val_df)} rows ({val_df[ID_COL].nunique()} participants); "
          f"{len(feature_cols)} features")

    X_train, y_train = train_df[feature_cols], train_df[LABEL_COL]
    X_val, y_val = val_df[feature_cols], val_df[LABEL_COL]

    sample_weight = compute_sample_weight("balanced", y_train)

    train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weight, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, free_raw_data=False)

    params = {**LGBM_PARAMS, "seed": seed}

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                   lgb.log_evaluation(period=0)],
    )

    train_pred = np.argmax(booster.predict(X_train, num_iteration=booster.best_iteration), axis=1)
    val_pred = np.argmax(booster.predict(X_val, num_iteration=booster.best_iteration), axis=1)
    train_macro_f1 = f1_score(y_train, train_pred, average="macro", zero_division=0)
    val_macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
    overfitting_gap = train_macro_f1 - val_macro_f1
    print(f"[train] best_iteration={booster.best_iteration}  train_macro_f1={train_macro_f1:.4f}  "
          f"val_macro_f1={val_macro_f1:.4f}  overfitting_gap={overfitting_gap:.4f}")

    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_out_path), num_iteration=booster.best_iteration)
    print(f"[train] wrote {model_out_path}")

    metrics = {
        "seed": seed,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "best_iteration": booster.best_iteration,
        "n_train_rows": len(train_df),
        "n_val_rows": len(val_df),
        "train_macro_f1": float(train_macro_f1),
        "val_macro_f1": float(val_macro_f1),
        "overfitting_gap": float(overfitting_gap),
    }
    metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[train] wrote {metrics_out_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the mcPHASES cycle-phase LightGBM baseline")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_OUT_PATH))
    parser.add_argument("--metrics-out", default=str(DEFAULT_METRICS_OUT_PATH))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-boost-round", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    args = parser.parse_args()
    train_model(
        data_path=args.data,
        splits_path=args.splits,
        model_out_path=args.model_out,
        metrics_out_path=args.metrics_out,
        seed=args.seed,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )


if __name__ == "__main__":
    main()
