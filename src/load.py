"""Data ingestion + schema discovery for the mcPHASES raw tables.

Reads tables from data/raw/ (CSV or Parquet, one file per table, filename
without extension = table name). Writes two discovery reports:
  - reports/schema.md        shape, id/day_in_study coverage, per-column dtype
                              and non-null % for every table.
  - reports/label_source.md  candidate table(s)/column(s) holding the 4-class
                              cycle-phase label, with value counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_REPORTS_DIR = Path("reports")

LABEL_KEYWORDS = ("phase", "menstruation", "follicular", "ovulation", "luteal")

SUPPORTED_SUFFIXES = (".csv", ".parquet")


def list_tables(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[str]:
    """Return sorted table names (filename stem) for every csv/parquet file in data_dir."""
    data_dir = Path(data_dir)
    assert data_dir.exists(), f"data dir not found: {data_dir}"
    tables = sorted(
        {p.stem for p in data_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES}
    )
    print(f"[load] list_tables: found {len(tables)} table(s) in {data_dir}")
    return tables


def load_table(name: str, data_dir: str | Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load one table by name from data_dir. Tolerant of csv/parquet. Strips column whitespace."""
    data_dir = Path(data_dir)
    csv_path = data_dir / f"{name}.csv"
    parquet_path = data_dir / f"{name}.parquet"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        raise FileNotFoundError(f"table '{name}' not found as .csv or .parquet in {data_dir}")

    df.columns = [str(c).strip() for c in df.columns]
    assert df.shape[1] > 0, f"table '{name}' loaded with zero columns"
    print(f"[load] load_table('{name}'): shape={df.shape}")
    return df


def _schema_section(table_name: str, df: pd.DataFrame) -> list[str]:
    lines = [f"## {table_name}", f"- rows: {len(df)}"]

    if "id" in df.columns:
        lines.append(f"- unique id count: {df['id'].nunique()}")
    else:
        lines.append("- unique id count: n/a (no 'id' column)")

    if "day_in_study" in df.columns:
        day_col = pd.to_numeric(df["day_in_study"], errors="coerce")
        lines.append(f"- day_in_study range: [{day_col.min()}, {day_col.max()}]")
    else:
        lines.append("- day_in_study range: n/a (no 'day_in_study' column)")

    lines.append("")
    lines.append("| column | dtype | non-null % |")
    lines.append("|---|---|---|")
    n_rows = len(df)
    for col in df.columns:
        non_null_pct = (df[col].notna().sum() / n_rows * 100) if n_rows else 0.0
        lines.append(f"| {col} | {df[col].dtype} | {non_null_pct:.1f}% |")
    lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def _label_section(table_name: str, col: str, df: pd.DataFrame) -> list[str]:
    return [
        f"## {table_name}.{col}",
        "",
        "```",
        df[col].value_counts(dropna=False).to_string(),
        "```",
        "",
    ]


def profile(data_dir: str | Path = DEFAULT_DATA_DIR, reports_dir: str | Path = DEFAULT_REPORTS_DIR) -> None:
    """Process one table at a time (never holding more than one in memory) and
    incrementally build both reports, so this scales to large raw files."""
    data_dir = Path(data_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    table_names = list_tables(data_dir)
    assert table_names, f"no tables found in {data_dir}"

    schema_lines = ["# Schema Report", ""]
    label_sections: list[str] = []
    n_matches = 0

    for name in table_names:
        df = load_table(name, data_dir)
        schema_lines.extend(_schema_section(name, df))

        for col in df.columns:
            if any(keyword in str(col).lower() for keyword in LABEL_KEYWORDS):
                label_sections.extend(_label_section(name, col, df))
                n_matches += 1

        del df

    (reports_dir / "schema.md").write_text("\n".join(schema_lines))
    print(f"[load] wrote {reports_dir / 'schema.md'}")

    label_lines = ["# Label Source Candidates", ""]
    label_lines.append(f"Searched every table's columns for keywords: {', '.join(LABEL_KEYWORDS)}")
    label_lines.append("")
    if not label_sections:
        label_lines.append("No candidate label columns found in any table.")
    else:
        label_lines.extend(label_sections)

    (reports_dir / "label_source.md").write_text("\n".join(label_lines))
    print(f"[load] wrote {reports_dir / 'label_source.md'} ({n_matches} candidate column(s) found)")


def main() -> None:
    parser = argparse.ArgumentParser(description="mcPHASES raw data loader / schema profiler")
    parser.add_argument("--profile", action="store_true", help="profile all tables in data/raw/")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    args = parser.parse_args()

    if args.profile:
        profile(data_dir=args.data_dir, reports_dir=args.reports_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
