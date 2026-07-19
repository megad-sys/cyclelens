"""Exploratory data analysis over the real mcPHASES tables. READ-ONLY.

Does not build the modeling parquet and does not drop any columns — this is
investigation only, to inform Stage 2 feature engineering. Writes
reports/eda.md (all findings, as markdown tables) and reports/eda/*.png
(a few plots).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.load import list_tables

DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_PLOTS_DIR = Path("reports/eda")

LABEL_TABLE = "hormones_and_selfreport"
LABEL_COL = "phase"
DAY_COL = "day_in_study"
ID_COL = "id"

# Biological order of the 4 cycle phases (also used for plot y-axis ordering).
PHASE_ORDER = ["Menstrual", "Follicular", "Fertility", "Luteal"]

SYMPTOM_COLUMNS = [
    "cramps", "moodswing", "appetite", "headaches", "sorebreasts", "fatigue",
    "sleepissue", "stress", "foodcravings", "indigestion", "bloating",
    "exerciselevel",
]

# Tables whose per-day key is not literally 'day_in_study'.
DAY_COLUMN_OVERRIDES = {
    "computed_temperature": "sleep_start_day_in_study",
    "exercise": "start_day_in_study",
    "sleep": "sleep_start_day_in_study",
}

# Tables with no day column at all (one row, or a few, per participant).
STATIC_TABLES = {"height_and_weight", "subject-info"}

# Signals used for BOTH value-sanity checks (Stage 1.5 item 4) and coverage
# after joining onto the label spine (item 5). The 4 named in item 6
# ("signal check") are a subset of these, filtered at use time.
SIGNAL_SPECS = [
    dict(key="resting_hr", label="resting HR (bpm)", table="resting_heart_rate",
         day_col=None, value_col="value", low=25, high=180, filter_col=None),
    dict(key="nightly_temperature", label="nightly temperature", table="computed_temperature",
         day_col="sleep_start_day_in_study", value_col="nightly_temperature", low=-15, high=115,
         filter_col=None),
    dict(key="glucose_mean", label="glucose (mg/dL)", table="glucose",
         day_col=None, value_col="glucose_value", low=20, high=600, filter_col=None),
    dict(key="hrv_rmssd", label="HRV rmssd (ms)", table="heart_rate_variability_details",
         day_col=None, value_col="rmssd", low=0, high=300, filter_col=None),
    dict(key="respiratory_rate", label="respiratory rate (breaths/min)", table="respiratory_rate_summary",
         day_col=None, value_col="full_sleep_breathing_rate", low=5, high=40, filter_col=None),
    dict(key="sleep_minutesasleep", label="sleep minutes asleep", table="sleep",
         day_col="sleep_start_day_in_study", value_col="minutesasleep", low=0, high=900,
         filter_col="mainsleep"),
]

# The subset of SIGNAL_SPECS keys used in the phase-separation "signal check".
SIGNAL_CHECK_KEYS = ("resting_hr", "nightly_temperature", "hrv_rmssd", "glucose_mean")


# ---------------------------------------------------------------------------
# Loading helpers (column-selective, so the multi-million-row tables stay cheap)
# ---------------------------------------------------------------------------

def _resolve_raw_columns(path: Path, wanted: list[str]) -> dict[str, str]:
    """Map each wanted (stripped) column name to the actual raw header name."""
    if path.suffix.lower() == ".parquet":
        header = pd.read_parquet(path).columns.tolist()
    else:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    stripped_to_raw = {str(c).strip(): c for c in header}
    missing = [w for w in wanted if w not in stripped_to_raw]
    assert not missing, f"columns {missing} not found in {path} (available: {header})"
    return {w: stripped_to_raw[w] for w in wanted}


def _load_columns(name: str, columns: list[str], data_dir: Path) -> pd.DataFrame:
    """Load only `columns` from table `name`. Tolerant of csv/parquet, strips whitespace."""
    csv_path = data_dir / f"{name}.csv"
    parquet_path = data_dir / f"{name}.parquet"
    if csv_path.exists():
        path = csv_path
    elif parquet_path.exists():
        path = parquet_path
    else:
        raise FileNotFoundError(f"table '{name}' not found as .csv or .parquet in {data_dir}")

    raw_map = _resolve_raw_columns(path, columns)
    raw_cols = list(raw_map.values())

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path, columns=raw_cols)
    else:
        df = pd.read_csv(path, usecols=raw_cols)

    df = df.rename(columns={raw: stripped for stripped, raw in raw_map.items()})
    return df[columns]


def _table_columns(name: str, data_dir: Path) -> list[str]:
    csv_path = data_dir / f"{name}.csv"
    parquet_path = data_dir / f"{name}.parquet"
    path = csv_path if csv_path.exists() else parquet_path
    assert path.exists(), f"table '{name}' not found as .csv or .parquet in {data_dir}"
    if path.suffix.lower() == ".parquet":
        header = pd.read_parquet(path).columns.tolist()
    else:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    return [str(c).strip() for c in header]


def _resolve_day_column(name: str, columns: list[str]) -> str | None:
    if name in STATIC_TABLES:
        return None
    if DAY_COL in columns:
        return DAY_COL
    if name in DAY_COLUMN_OVERRIDES and DAY_COLUMN_OVERRIDES[name] in columns:
        return DAY_COLUMN_OVERRIDES[name]
    candidates = sorted(c for c in columns if "day_in_study" in c)
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def _markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _pct(numerator: float, denominator: float) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100 * numerator / denominator:.1f}%"


# ---------------------------------------------------------------------------
# Section 1: label class balance + per-participant + sample timelines
# ---------------------------------------------------------------------------

def section_label_balance(hormones_df: pd.DataFrame, plots_dir: Path) -> str:
    lines = ["## 1. Label: Class Balance", ""]

    labeled = hormones_df.dropna(subset=[LABEL_COL])
    n_dropped = len(hormones_df) - len(labeled)
    lines.append(f"Total rows: {len(hormones_df)}; unlabeled (NaN phase) dropped from analysis: {n_dropped}.")
    lines.append("")

    lines.append("### Overall")
    counts = labeled[LABEL_COL].value_counts()
    total = counts.sum()
    rows = [[phase, int(counts.get(phase, 0)), _pct(counts.get(phase, 0), total)] for phase in PHASE_ORDER]
    lines.append(_markdown_table(["phase", "count", "pct"], rows))
    lines.append("")

    lines.append("### Per participant (id x phase counts)")
    pivot = pd.crosstab(labeled[ID_COL], labeled[LABEL_COL])
    for phase in PHASE_ORDER:
        if phase not in pivot.columns:
            pivot[phase] = 0
    pivot = pivot[PHASE_ORDER]
    pivot["total"] = pivot.sum(axis=1)
    rows = [[pid, *[int(v) for v in row]] for pid, row in zip(pivot.index, pivot.values)]
    lines.append(_markdown_table(["id", *PHASE_ORDER, "total"], rows))
    lines.append("")

    lines.append("### Sample participant phase timelines")
    plots_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = (
        labeled.groupby(ID_COL).size().sort_values(ascending=False).head(3).sort_index().index.tolist()
    )
    phase_code = {p: i for i, p in enumerate(PHASE_ORDER)}
    for pid in sample_ids:
        participant = labeled[labeled[ID_COL] == pid].sort_values(DAY_COL)
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.step(
            participant[DAY_COL],
            participant[LABEL_COL].map(phase_code),
            where="post",
            marker="o",
            markersize=3,
        )
        ax.set_yticks(range(len(PHASE_ORDER)))
        ax.set_yticklabels(PHASE_ORDER)
        ax.set_xlabel("day_in_study")
        ax.set_title(f"participant {pid}: phase over time")
        fig.tight_layout()
        out_path = plots_dir / f"phase_timeline_id_{pid}.png"
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        lines.append(f"- `{out_path.as_posix()}`")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 2: categorical symptom columns
# ---------------------------------------------------------------------------

def section_symptom_columns(hormones_df: pd.DataFrame) -> str:
    lines = ["## 2. Categorical Symptom Columns", ""]

    present = [c for c in SYMPTOM_COLUMNS if c in hormones_df.columns]
    missing = [c for c in SYMPTOM_COLUMNS if c not in hormones_df.columns]
    if missing:
        lines.append(f"_Not present in this table: {', '.join(missing)}._")
        lines.append("")

    for col in present:
        lines.append(f"### {col}")
        counts = hormones_df[col].value_counts(dropna=False)
        rows = [[str(v), int(c)] for v, c in counts.items()]
        lines.append(_markdown_table(["value", "count"], rows))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 3: join granularity across all tables
# ---------------------------------------------------------------------------

def section_join_granularity(table_names: list[str], data_dir: Path) -> str:
    lines = ["## 3. Join Granularity (rows per id x day key)", ""]
    rows = []

    for name in table_names:
        columns = _table_columns(name, data_dir)
        day_col = _resolve_day_column(name, columns)

        if day_col is None:
            df = _load_columns(name, [ID_COL], data_dir)
            per_id = df.groupby(ID_COL).size()
            rows.append([
                name, "(static, no day column)", len(df), "n/a",
                f"{per_id.mean():.1f}", f"{per_id.median():.0f}", f"{per_id.max():.0f}", "n/a",
            ])
            del df
            continue

        df = _load_columns(name, [ID_COL, day_col], data_dir)
        key_counts = df.groupby([ID_COL, day_col]).size()
        n_single = int((key_counts == 1).sum())
        rows.append([
            name, day_col, len(df), len(key_counts),
            f"{key_counts.mean():.1f}", f"{key_counts.median():.0f}", f"{key_counts.max():.0f}",
            _pct(n_single, len(key_counts)),
        ])
        del df

    lines.append(_markdown_table(
        ["table", "day column", "rows", "unique (id,day) keys", "avg rows/key", "median", "max",
         "% keys already 1-row"],
        rows,
    ))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 4: value sanity checks
# ---------------------------------------------------------------------------

def _load_signal_raw(spec: dict, data_dir: Path) -> pd.DataFrame:
    day_col = spec["day_col"] or DAY_COL
    columns = [ID_COL, day_col, spec["value_col"]]
    if spec["filter_col"]:
        columns.append(spec["filter_col"])
    df = _load_columns(spec["table"], columns, data_dir)
    if spec["filter_col"]:
        df = df[df[spec["filter_col"]] == True]  # noqa: E712
        df = df.drop(columns=[spec["filter_col"]])
    df = df.rename(columns={day_col: DAY_COL})
    return df


def section_value_sanity(data_dir: Path) -> str:
    lines = ["## 4. Value Sanity Checks", ""]
    lines.append(
        "_Plausible ranges are approximate literature-based bounds for flagging outliers, "
        "not clinical cutoffs._"
    )
    lines.append("")

    rows = []
    for spec in SIGNAL_SPECS:
        df = _load_signal_raw(spec, data_dir)
        values = pd.to_numeric(df[spec["value_col"]], errors="coerce").dropna()
        if values.empty:
            rows.append([spec["label"], f"{spec['table']}.{spec['value_col']}",
                         "n/a", "n/a", "n/a", f"[{spec['low']}, {spec['high']}]", 0, "n/a"])
            del df
            continue
        n_implausible = int(((values < spec["low"]) | (values > spec["high"])).sum())
        rows.append([
            spec["label"], f"{spec['table']}.{spec['value_col']}",
            f"{values.min():.1f}", f"{values.median():.1f}", f"{values.max():.1f}",
            f"[{spec['low']}, {spec['high']}]", n_implausible, _pct(n_implausible, len(values)),
        ])
        del df

    lines.append(_markdown_table(
        ["signal", "table.column", "min", "median", "max", "plausible range",
         "n implausible", "% implausible"],
        rows,
    ))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sections 5 & 6: coverage after join onto the label spine, and signal-by-phase
# ---------------------------------------------------------------------------

def _build_label_spine(hormones_df: pd.DataFrame) -> pd.DataFrame:
    labeled = hormones_df.dropna(subset=[LABEL_COL])
    return labeled[[ID_COL, DAY_COL, LABEL_COL]].drop_duplicates(subset=[ID_COL, DAY_COL])


def _aggregate_signal_daily(spec: dict, data_dir: Path) -> pd.DataFrame:
    df = _load_signal_raw(spec, data_dir)
    df[spec["value_col"]] = pd.to_numeric(df[spec["value_col"]], errors="coerce")
    daily = df.groupby([ID_COL, DAY_COL], as_index=False)[spec["value_col"]].mean()
    return daily


def section_coverage_after_join(label_spine: pd.DataFrame, data_dir: Path) -> str:
    lines = ["## 5. Coverage After Join (onto the label spine)", ""]
    n_labeled_days = len(label_spine)
    lines.append(f"Label spine: {n_labeled_days} labeled (id, day_in_study) rows.")
    lines.append("")

    rows = []
    for spec in SIGNAL_SPECS:
        daily = _aggregate_signal_daily(spec, data_dir)
        joined = label_spine.merge(daily, on=[ID_COL, DAY_COL], how="left")
        n_non_null = int(joined[spec["value_col"]].notna().sum())
        rows.append([spec["label"], n_non_null, n_labeled_days, _pct(n_non_null, n_labeled_days)])
        del daily, joined

    lines.append(_markdown_table(["signal", "non-null days", "labeled days", "coverage %"], rows))
    lines.append("")
    return "\n".join(lines)


def section_signal_by_phase(label_spine: pd.DataFrame, data_dir: Path) -> str:
    lines = ["## 6. Signal Check: mean +/- std by phase", ""]

    for key in SIGNAL_CHECK_KEYS:
        spec = next(s for s in SIGNAL_SPECS if s["key"] == key)
        daily = _aggregate_signal_daily(spec, data_dir)
        joined = label_spine.merge(daily, on=[ID_COL, DAY_COL], how="inner")

        lines.append(f"### {spec['label']}")
        grouped = joined.groupby(LABEL_COL)[spec["value_col"]].agg(["mean", "std", "count"])
        rows = []
        for phase in PHASE_ORDER:
            if phase in grouped.index:
                m, s, n = grouped.loc[phase]
                rows.append([phase, f"{m:.2f}", f"{s:.2f}", int(n)])
            else:
                rows.append([phase, "n/a", "n/a", 0])
        lines.append(_markdown_table(["phase", "mean", "std", "n"], rows))
        lines.append("")
        del daily, joined

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_eda(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    plots_dir: str | Path = DEFAULT_PLOTS_DIR,
) -> Path:
    data_dir = Path(data_dir)
    reports_dir = Path(reports_dir)
    plots_dir = Path(plots_dir)
    assert data_dir.exists(), f"data dir not found: {data_dir}"
    reports_dir.mkdir(parents=True, exist_ok=True)

    table_names = list_tables(data_dir)
    assert table_names, f"no tables found in {data_dir}"
    assert LABEL_TABLE in table_names, f"expected label table '{LABEL_TABLE}' not found in {data_dir}"

    hormones_cols = [ID_COL, DAY_COL, LABEL_COL, *SYMPTOM_COLUMNS]
    available = _table_columns(LABEL_TABLE, data_dir)
    hormones_cols = [c for c in hormones_cols if c in available]
    hormones_df = _load_columns(LABEL_TABLE, hormones_cols, data_dir)
    print(f"[eda] loaded {LABEL_TABLE}: shape={hormones_df.shape}")

    sections = [
        "# Exploratory Data Analysis Report",
        "",
        "Read-only investigation. No modeling table built, no columns dropped here.",
        "",
        section_label_balance(hormones_df, plots_dir),
        section_symptom_columns(hormones_df),
        section_join_granularity(table_names, data_dir),
        section_value_sanity(data_dir),
    ]

    label_spine = _build_label_spine(hormones_df)
    sections.append(section_coverage_after_join(label_spine, data_dir))
    sections.append(section_signal_by_phase(label_spine, data_dir))

    out_path = reports_dir / "eda.md"
    out_path.write_text("\n".join(sections))
    print(f"[eda] wrote {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="mcPHASES exploratory data analysis (read-only)")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--plots-dir", default=str(DEFAULT_PLOTS_DIR))
    args = parser.parse_args()
    run_eda(data_dir=args.data_dir, reports_dir=args.reports_dir, plots_dir=args.plots_dir)


if __name__ == "__main__":
    main()
