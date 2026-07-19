"""Evaluate the trained model against the frozen benchmark card baselines.

Computes macro-F1, per-class F1, and balanced accuracy on the chosen split
for the model, the global-majority baseline, and the personal-majority
baseline (leave-one-out per participant, as defined in
reports/benchmark_card.md), saves reports/eval_test.json and a confusion
matrix PNG, and prints a comparison table.
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
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score, confusion_matrix, f1_score

from src.splits import ID_COL, LABEL_COL, LABEL_NAMES, REFERENCE_COLUMNS

DEFAULT_MODEL_PATH = Path("models/lgbm.txt")
DEFAULT_DATA_PATH = Path("data/processed/dataset.parquet")
DEFAULT_SPLITS_PATH = Path("data/processed/splits.json")
DEFAULT_SPLIT_NAME = "test"
DEFAULT_OUTPUT_PATH = Path("reports/eval_test.json")
DEFAULT_CONFUSION_PATH = Path("reports/confusion_test.png")

LABEL_VALUES = sorted(LABEL_NAMES)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in REFERENCE_COLUMNS]


def _global_majority_predictions(train_df: pd.DataFrame, n: int) -> tuple[np.ndarray, int]:
    majority_label = int(train_df[LABEL_COL].value_counts().idxmax())
    return np.full(n, majority_label), majority_label


def _personal_majority_predictions(split_df: pd.DataFrame, fallback_label: int) -> np.ndarray:
    """Leave-one-out per participant: predict their most common OTHER label in this split."""
    preds = pd.Series(index=split_df.index, dtype="int64")
    for _, group in split_df.groupby(ID_COL):
        counts_full = group[LABEL_COL].value_counts().reindex(LABEL_VALUES, fill_value=0).to_dict()
        for idx, row in group.iterrows():
            loo_counts = dict(counts_full)
            loo_counts[row[LABEL_COL]] -= 1
            total = sum(loo_counts.values())
            if total == 0:
                preds.loc[idx] = fallback_label
            else:
                preds.loc[idx] = max(LABEL_VALUES, key=lambda label: (loo_counts[label], -label))
    return preds.reindex(split_df.index).to_numpy()


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=LABEL_VALUES, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=LABEL_VALUES, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    return {
        "macro_f1": float(macro_f1),
        "per_class_f1": {LABEL_NAMES[label]: float(f1) for label, f1 in zip(LABEL_VALUES, per_class_f1)},
        "balanced_accuracy": float(balanced_acc),
    }


def _print_comparison_table(results: dict) -> None:
    print(f"\nComparison on split='{results['split']}' "
          f"(n={results['n_rows']} rows, {results['n_participants']} participants)\n")
    header = f"{'approach':<20}{'macro-F1':>10}  " + "  ".join(f"{name:>12}" for name in LABEL_NAMES.values())
    print(header)
    for key, label in [("model", "model"), ("global_majority", "global-majority"),
                        ("personal_majority", "personal-majority")]:
        m = results[key]
        row = f"{label:<20}{m['macro_f1']:>10.4f}  " + "  ".join(
            f"{m['per_class_f1'][name]:>12.4f}" for name in LABEL_NAMES.values()
        )
        print(row)
    print()


def evaluate(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    data_path: str | Path = DEFAULT_DATA_PATH,
    splits_path: str | Path = DEFAULT_SPLITS_PATH,
    split_name: str = DEFAULT_SPLIT_NAME,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    confusion_path: str | Path = DEFAULT_CONFUSION_PATH,
) -> dict:
    model_path = Path(model_path)
    data_path = Path(data_path)
    splits_path = Path(splits_path)
    output_path = Path(output_path)
    confusion_path = Path(confusion_path)

    assert model_path.exists(), f"model not found: {model_path}"
    assert data_path.exists(), f"dataset not found: {data_path}"
    assert splits_path.exists(), f"splits not found: {splits_path}"

    df = pd.read_parquet(data_path)
    splits = json.loads(splits_path.read_text())
    assert split_name in splits, f"unknown split '{split_name}', expected one of {list(splits)}"

    feature_cols = _feature_columns(df)
    train_df = df[df[ID_COL].isin(splits["train"])]
    eval_df = df[df[ID_COL].isin(splits[split_name])].copy()
    assert len(eval_df) > 0, f"split '{split_name}' has zero rows"
    print(f"[evaluate] split='{split_name}': {len(eval_df)} rows, {eval_df[ID_COL].nunique()} participants")

    booster = lgb.Booster(model_file=str(model_path))
    model_proba = booster.predict(eval_df[feature_cols])
    model_pred = np.argmax(model_proba, axis=1)

    global_pred, global_majority_label = _global_majority_predictions(train_df, len(eval_df))
    personal_pred = _personal_majority_predictions(eval_df, fallback_label=global_majority_label)

    y_true = eval_df[LABEL_COL].to_numpy()

    results = {
        "split": split_name,
        "n_rows": len(eval_df),
        "n_participants": int(eval_df[ID_COL].nunique()),
        "model": _compute_metrics(y_true, model_pred),
        "global_majority": _compute_metrics(y_true, global_pred),
        "personal_majority": _compute_metrics(y_true, personal_pred),
    }

    cm = confusion_matrix(y_true, model_pred, labels=LABEL_VALUES)
    fig, ax = plt.subplots(figsize=(5, 5))
    display = ConfusionMatrixDisplay(cm, display_labels=[LABEL_NAMES[label] for label in LABEL_VALUES])
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix ({split_name}, model)")
    fig.tight_layout()
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(confusion_path, dpi=100)
    plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))

    _print_comparison_table(results)
    print(f"[evaluate] wrote {output_path}")
    print(f"[evaluate] wrote {confusion_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the mcPHASES cycle-phase model vs baselines")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--split", default=DEFAULT_SPLIT_NAME)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--confusion-out", default=str(DEFAULT_CONFUSION_PATH))
    args = parser.parse_args()
    evaluate(
        model_path=args.model,
        data_path=args.data,
        splits_path=args.splits,
        split_name=args.split,
        output_path=args.output,
        confusion_path=args.confusion_out,
    )


if __name__ == "__main__":
    main()
