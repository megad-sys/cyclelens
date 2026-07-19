import pandas as pd

from src.load import list_tables, load_table


def _write_fixture_tables(tmp_path):
    table_a = pd.DataFrame(
        {
            " id ": [1, 1, 2],
            " day_in_study ": [0, 1, 0],
            " value  ": [10.0, 11.0, 20.0],
        }
    )
    table_a.to_csv(tmp_path / "table_a.csv", index=False)

    table_b = pd.DataFrame(
        {
            "id": [1, 2],
            " phase ": ["luteal", "ovulation"],
        }
    )
    table_b.to_parquet(tmp_path / "table_b.parquet", index=False)

    return tmp_path


def test_list_tables_finds_csv_and_parquet(tmp_path):
    _write_fixture_tables(tmp_path)

    tables = list_tables(tmp_path)

    assert tables == ["table_a", "table_b"]


def test_load_table_strips_whitespace_from_csv_columns(tmp_path):
    _write_fixture_tables(tmp_path)

    df = load_table("table_a", tmp_path)

    assert list(df.columns) == ["id", "day_in_study", "value"]
    assert len(df) == 3


def test_load_table_reads_parquet_and_strips_columns(tmp_path):
    _write_fixture_tables(tmp_path)

    df = load_table("table_b", tmp_path)

    assert list(df.columns) == ["id", "phase"]
    assert len(df) == 2


def test_load_table_missing_raises(tmp_path):
    _write_fixture_tables(tmp_path)

    try:
        load_table("does_not_exist", tmp_path)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for missing table")
