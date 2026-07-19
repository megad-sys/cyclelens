"""5-fold grouped cross-validation: a stable macro-F1 estimate.

Runs GroupKFold(n_splits=5) on `id` over the FULL dataset (all participants,
ignoring the frozen train/val/test split from src/splits.py), retraining the
SAME regularized model (src.train.LGBM_PARAMS) each fold. Within each fold's
training partition, a small grouped inner train/val carve-out is used purely
for early stopping -- the held-out fold is never touched during training.
Also computes the global-majority and personal-majority baselines per fold.

This is the stable headline estimate; the frozen test split from
src/evaluate.py remains the final held-out number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.evaluate import _global_majority_predictions, _personal_majority_predictions
from src.splits import ID_COL, LABEL_COL, LABEL_NAMES
from src.train import DEFAULT_EARLY_STOPPING_ROUNDS, DEFAULT_NUM_BOOST_ROUND, LGBM_PARAMS, _feature_columns

DEFAULT_DATA_PATH = Path("data/processed/dataset.parquet")
DEFAULT_OUTPUT_PATH = Path("reports/crossval.json")
DEFAULT_N_SPLITS = 5
DEFAULT_SEED = 42
INNER_VAL_FRACTION = 0.15  # of each fold's training participants, held out for early stopping only


def _inner_train_val_ids(ids: np.ndarray, val_frac: float, seed: int) -> tuple[set, set]:
    """Deterministic seeded shuffle+slice -- a lightweight grouped split used only
    to give LightGBM an early-stopping validation set that never touches the
    held-out fold."""
    ids = list(ids)
    rng = np.random.RandomState(seed)
    shuffled = [ids[i] for i in rng.permutation(len(ids))]
    n = len(shuffled)
    if n < 2:
        return set(shuffled), set()
    n_val = min(max(1, round(val_frac * n)), n - 1)
    return set(shuffled[n_val:]), set(shuffled[:n_val])


def _train_fold_model(train_df: pd.DataFrame, feature_cols: list[str], seed: int,
                       num_boost_round: int, early_stopping_rounds: int) -> lgb.Booster:
    inner_train_ids, inner_val_ids = _inner_train_val_ids(
        train_df[ID_COL].unique(), INNER_VAL_FRACTION, seed
    )
    inner_train_df = train_df[train_df[ID_COL].isin(inner_train_ids)]
    inner_val_df = train_df[train_df[ID_COL].isin(inner_val_ids)] if inner_val_ids else None

    X_train, y_train = inner_train_df[feature_cols], inner_train_df[LABEL_COL]
    sample_weight = compute_sample_weight("balanced", y_train)
    train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weight, free_raw_data=False)
    params = {**LGBM_PARAMS, "seed": seed}

    if inner_val_df is not None and len(inner_val_df) > 0:
        X_val, y_val = inner_val_df[feature_cols], inner_val_df[LABEL_COL]
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, free_raw_data=False)
        booster = lgb.train(
            params, train_set, num_boost_round=num_boost_round,
            valid_sets=[val_set], valid_names=["inner_val"],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
    else:
        # degenerate fold (too few training participants for an inner val split):
        # train for a fixed, smaller number of rounds with no early stopping.
        booster = lgb.train(params, train_set, num_boost_round=min(num_boost_round, 200))

    return booster


def cross_validate(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
) -> dict:
    data_path = Path(data_path)
    output_path = Path(output_path)
    assert data_path.exists(), f"dataset not found: {data_path}"

    df = pd.read_parquet(data_path).reset_index(drop=True)
    feature_cols = _feature_columns(df)
    n_participants = df[ID_COL].nunique()
    assert n_participants >= n_splits, f"need at least {n_splits} participants, got {n_participants}"
    print(f"[crossval] loaded {data_path}: shape={df.shape}, {n_participants} participants, "
          f"{n_splits}-fold GroupKFold")

    gkf = GroupKFold(n_splits=n_splits)
    fold_records = []

    for fold_idx, (train_pos, held_out_pos) in enumerate(gkf.split(df, groups=df[ID_COL])):
        train_df = df.iloc[train_pos]
        held_out_df = df.iloc[held_out_pos]

        booster = _train_fold_model(train_df, feature_cols, seed + fold_idx,
                                     num_boost_round, early_stopping_rounds)

        model_pred = np.argmax(
            booster.predict(held_out_df[feature_cols], num_iteration=booster.best_iteration), axis=1
        )
        global_pred, global_majority_label = _global_majority_predictions(train_df, len(held_out_df))
        personal_pred = _personal_majority_predictions(held_out_df, fallback_label=global_majority_label)

        y_true = held_out_df[LABEL_COL].to_numpy()
        model_f1 = f1_score(y_true, model_pred, average="macro", zero_division=0)
        global_f1 = f1_score(y_true, global_pred, average="macro", zero_division=0)
        personal_f1 = f1_score(y_true, personal_pred, average="macro", zero_division=0)

        record = {
            "fold": fold_idx,
            "n_held_out_participants": int(held_out_df[ID_COL].nunique()),
            "n_held_out_rows": len(held_out_df),
            "best_iteration": getattr(booster, "best_iteration", None),
            "model_macro_f1": float(model_f1),
            "global_majority_macro_f1": float(global_f1),
            "personal_majority_macro_f1": float(personal_f1),
        }
        fold_records.append(record)
        print(f"[crossval] fold {fold_idx}: {record['n_held_out_participants']} held-out participants, "
              f"{record['n_held_out_rows']} rows -- model={model_f1:.4f}  "
              f"global-majority={global_f1:.4f}  personal-majority={personal_f1:.4f}")

    summary = {}
    for key in ["model_macro_f1", "global_majority_macro_f1", "personal_majority_macro_f1"]:
        values = np.array([r[key] for r in fold_records])
        summary[key.replace("_macro_f1", "")] = {
            "mean_macro_f1": float(values.mean()),
            "std_macro_f1": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }

    results = {
        "n_splits": n_splits,
        "n_participants": n_participants,
        "seed": seed,
        "folds": fold_records,
        "summary": summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"[crossval] wrote {output_path}")

    _print_summary_table(summary, n_splits)

    return results


def _print_summary_table(summary: dict, n_splits: int) -> None:
    print(f"\n{n_splits}-fold cross-validation summary (mean +/- std macro-F1)\n")
    print(f"{'approach':<20}{'mean macro-F1':>16}{'std':>10}")
    for key, label in [("model", "model"), ("global_majority", "global-majority"),
                        ("personal_majority", "personal-majority")]:
        s = summary[key]
        print(f"{label:<20}{s['mean_macro_f1']:>16.4f}{s['std_macro_f1']:>10.4f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold grouped cross-validation for the cycle-phase model")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-boost-round", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    args = parser.parse_args()
    cross_validate(
        data_path=args.data,
        output_path=args.output,
        n_splits=args.n_splits,
        seed=args.seed,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )


if __name__ == "__main__":
    main()
