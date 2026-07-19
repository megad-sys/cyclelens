"""Build the modeling table: one row per (id, day_in_study).

Joins wearable, CGM, and self-report signals onto the hormones_and_selfreport
label spine. Applies documented data-quality fixes (glucose unit mixing,
sensor-dropout sentinels) BEFORE aggregating intraday tables to daily
granularity. The six columns the label is derived from (phase, lh, estrogen,
pdg, flow_volume, flow_color) are never loaded and are asserted absent from
the output — they would leak the label into the features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_OUTPUT_PATH = Path("data/processed/dataset.parquet")
DEFAULT_FEATURE_DICT_PATH = Path("reports/feature_dictionary.md")

ID_COL = "id"
DAY_COL = "day_in_study"
STUDY_INTERVAL_COL = "study_interval"

LABEL_TABLE = "hormones_and_selfreport"
PHASE_COL = "phase"
PHASE_TO_LABEL = {"Menstrual": 0, "Follicular": 1, "Fertility": 2, "Luteal": 3}

# The label is derived from these. Never loaded, never allowed as features.
LEAKAGE_COLUMNS = ["phase", "lh", "estrogen", "pdg", "flow_volume", "flow_color"]

# All hormones_and_selfreport symptom columns EXCEPT the flow_* leakage columns.
SYMPTOM_COLUMNS = [
    "appetite", "exerciselevel", "headaches", "cramps", "sorebreasts", "fatigue",
    "sleepissue", "moodswing", "stress", "foodcravings", "indigestion", "bloating",
]

# hormones_and_selfreport symptoms are a 0-5 Likert scale per the mcPHASES README
# ("0 = Not at all" ... "5 = Very high"); observed text labels vary slightly
# between columns ("Very Low" vs "Very Low/Little"), both mapped to 1. A few rows
# contain raw numeric-string codes (e.g. "2") instead of text -- handled by the
# numeric fallback in _encode_ordinal.
ORDINAL_MAP = {
    "not at all": 0.0,
    "very low": 1.0,
    "very low/little": 1.0,
    "low": 2.0,
    "moderate": 3.0,
    "high": 4.0,
    "very high": 5.0,
}

GLUCOSE_MMOL_THRESHOLD = 30.0      # below this, value is assumed mmol/L
GLUCOSE_MMOL_TO_MGDL = 18.0182
GLUCOSE_PLAUSIBLE_RANGE = (20.0, 400.0)

REFERENCE_COLUMNS = [ID_COL, DAY_COL, STUDY_INTERVAL_COL]

# Continuous physiological features get a per-participant z-scored (_pz) twin:
# (value - participant_mean) / participant_std, computed over that participant's
# OWN days only. Leakage-safe because every participant is wholly within one
# split -- no cross-participant or cross-split information is used.
PARTICIPANT_ZSCORE_COLUMNS = [
    "resting_hr", "nightly_temperature", "nightly_temperature_std", "hrv_rmssd",
    "hrv_high_frequency", "hrv_low_frequency", "respiratory_rate", "sleep_minutesasleep",
    "sleep_efficiency", "sleep_score_overall", "steps_total", "stress_score",
    "glucose_mean", "demographic_vo2_max",
]


# ---------------------------------------------------------------------------
# Loading helpers (column-selective; mirrors src/eda.py so the multi-million
# row tables -- steps, glucose, HRV -- stay cheap to load)
# ---------------------------------------------------------------------------

def _resolve_raw_columns(path: Path, wanted: list[str]) -> dict[str, str]:
    if path.suffix.lower() == ".parquet":
        header = pd.read_parquet(path).columns.tolist()
    else:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    stripped_to_raw = {str(c).strip(): c for c in header}
    missing = [w for w in wanted if w not in stripped_to_raw]
    assert not missing, f"columns {missing} not found in {path} (available: {header})"
    return {w: stripped_to_raw[w] for w in wanted}


def _load_columns(name: str, columns: list[str], data_dir: Path) -> pd.DataFrame:
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
    print(f"[features] loaded {name}[{columns}]: shape={df.shape}")
    return df[columns]


def _available_columns(name: str, data_dir: Path) -> list[str]:
    csv_path = data_dir / f"{name}.csv"
    parquet_path = data_dir / f"{name}.parquet"
    path = csv_path if csv_path.exists() else parquet_path
    assert path.exists(), f"table '{name}' not found as .csv or .parquet in {data_dir}"
    if path.suffix.lower() == ".parquet":
        header = pd.read_parquet(path).columns.tolist()
    else:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    return [str(c).strip() for c in header]


def _markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ordinal symptom encoding
# ---------------------------------------------------------------------------

def _encode_ordinal_value(value) -> float:
    if pd.isna(value):
        return np.nan
    key = str(value).strip().lower()
    if key in ORDINAL_MAP:
        return ORDINAL_MAP[key]
    try:
        numeric = float(key)
    except ValueError:
        return np.nan
    return numeric if 0.0 <= numeric <= 5.0 else np.nan


def _encode_ordinal(series: pd.Series) -> pd.Series:
    return series.map(_encode_ordinal_value).astype(float)


# ---------------------------------------------------------------------------
# Label spine + symptom features
# ---------------------------------------------------------------------------

def _build_label_spine(data_dir: Path, feature_meta: list[dict]) -> pd.DataFrame:
    available = _available_columns(LABEL_TABLE, data_dir)
    symptom_cols = [c for c in SYMPTOM_COLUMNS if c in available]
    missing_symptoms = [c for c in SYMPTOM_COLUMNS if c not in available]
    if missing_symptoms:
        print(f"[features] symptom columns not present, skipped: {missing_symptoms}")

    wanted = [ID_COL, STUDY_INTERVAL_COL, DAY_COL, PHASE_COL, *symptom_cols]
    raw = _load_columns(LABEL_TABLE, wanted, data_dir)

    labeled = raw.dropna(subset=[PHASE_COL]).drop_duplicates(subset=[ID_COL, DAY_COL]).copy()
    mapped = labeled[PHASE_COL].map(PHASE_TO_LABEL)
    assert mapped.notna().all(), (
        f"unmapped phase value(s) found: {sorted(labeled.loc[mapped.isna(), PHASE_COL].unique())}"
    )
    labeled["label"] = mapped.astype(int)

    dataset = labeled[[ID_COL, STUDY_INTERVAL_COL, DAY_COL, "label"]].copy()

    for col in symptom_cols:
        dataset[col] = _encode_ordinal(labeled[col])
        feature_meta.append(dict(
            name=col, source=f"{LABEL_TABLE}.{col}", units="ordinal 0-5 (Likert)",
            notes="text label mapped via ORDINAL_MAP; stray numeric-string codes parsed directly",
        ))

    print(f"[features] label spine: {len(dataset)} labeled (id, day_in_study) rows")
    return dataset


# ---------------------------------------------------------------------------
# Per-table daily feature builders
# ---------------------------------------------------------------------------

def _add_resting_hr(dataset: pd.DataFrame, data_dir: Path, feature_meta: list[dict]) -> pd.DataFrame:
    df = _load_columns("resting_heart_rate", [ID_COL, DAY_COL, "value"], data_dir)
    df["value"] = df["value"].replace(0, np.nan)
    daily = df.groupby([ID_COL, DAY_COL], as_index=False)["value"].mean()
    daily = daily.rename(columns={"value": "resting_hr"})
    feature_meta.append(dict(
        name="resting_hr", source="resting_heart_rate.value", units="bpm",
        notes="0 (sensor dropout sentinel) replaced with NaN before daily mean",
    ))
    return dataset.merge(daily, on=[ID_COL, DAY_COL], how="left")


def _add_computed_temperature(dataset: pd.DataFrame, data_dir: Path, feature_meta: list[dict]) -> pd.DataFrame:
    day_col = "sleep_start_day_in_study"
    cols = [ID_COL, day_col, "nightly_temperature", "baseline_relative_nightly_standard_deviation"]
    df = _load_columns("computed_temperature", cols, data_dir)
    df = df.rename(columns={day_col: DAY_COL})
    daily = df.groupby([ID_COL, DAY_COL], as_index=False)[
        ["nightly_temperature", "baseline_relative_nightly_standard_deviation"]
    ].mean()
    daily = daily.rename(columns={"baseline_relative_nightly_standard_deviation": "nightly_temperature_std"})
    feature_meta.append(dict(
        name="nightly_temperature", source="computed_temperature.nightly_temperature", units="degrees C",
        notes="keyed on sleep_start_day_in_study -> day_in_study; daily mean",
    ))
    feature_meta.append(dict(
        name="nightly_temperature_std", source="computed_temperature.baseline_relative_nightly_standard_deviation",
        units="degrees C", notes="Fitbit-computed within-night std relative to personal baseline; daily mean",
    ))
    return dataset.merge(daily, on=[ID_COL, DAY_COL], how="left")


def _add_hrv(dataset: pd.DataFrame, data_dir: Path, feature_meta: list[dict]) -> pd.DataFrame:
    cols = [ID_COL, DAY_COL, "rmssd", "high_frequency", "low_frequency"]
    df = _load_columns("heart_rate_variability_details", cols, data_dir)
    daily = df.groupby([ID_COL, DAY_COL], as_index=False)[["rmssd", "high_frequency", "low_frequency"]].mean()
    daily = daily.rename(columns={
        "rmssd": "hrv_rmssd", "high_frequency": "hrv_high_frequency", "low_frequency": "hrv_low_frequency",
    })
    for name, src in [("hrv_rmssd", "rmssd"), ("hrv_high_frequency", "high_frequency"),
                       ("hrv_low_frequency", "low_frequency")]:
        feature_meta.append(dict(
            name=name, source=f"heart_rate_variability_details.{src}", units="ms (rmssd) / power (freq bands)",
            notes="daily mean of 5-minute sleep-window recordings",
        ))
    return dataset.merge(daily, on=[ID_COL, DAY_COL], how="left")


def _add_respiratory_rate(dataset: pd.DataFrame, data_dir: Path, feature_meta: list[dict]) -> pd.DataFrame:
    df = _load_columns("respiratory_rate_summary", [ID_COL, DAY_COL, "full_sleep_breathing_rate"], data_dir)
    df["full_sleep_breathing_rate"] = df["full_sleep_breathing_rate"].replace(0, np.nan)
    daily = df.groupby([ID_COL, DAY_COL], as_index=False)["full_sleep_breathing_rate"].mean()
    daily = daily.rename(columns={"full_sleep_breathing_rate": "respiratory_rate"})
    feature_meta.append(dict(
        name="respiratory_rate", source="respiratory_rate_summary.full_sleep_breathing_rate",
        units="breaths/min", notes="0 (sensor dropout sentinel) replaced with NaN before daily mean",
    ))
    return dataset.merge(daily, on=[ID_COL, DAY_COL], how="left")


def _add_sleep(dataset: pd.DataFrame, data_dir: Path, feature_meta: list[dict]) -> pd.DataFrame:
    day_col = "sleep_start_day_in_study"
    cols = [ID_COL, day_col, "minutesasleep", "efficiency", "minutestofallasleep", "timeinbed", "mainsleep"]
    df = _load_columns("sleep", cols, data_dir)
    df = df[df["mainsleep"] == True].drop(columns=["mainsleep"])  # noqa: E712
    df = df.rename(columns={day_col: DAY_COL})
    value_cols = ["minutesasleep", "efficiency", "minutestofallasleep", "timeinbed"]
    daily = df.groupby([ID_COL, DAY_COL], as_index=False)[value_cols].mean()
    daily = daily.rename(columns={
        "minutesasleep": "sleep_minutesasleep", "efficiency": "sleep_efficiency",
        "minutestofallasleep": "sleep_minutes_to_fall_asleep", "timeinbed": "sleep_time_in_bed",
    })
    for name, src in [("sleep_minutesasleep", "minutesasleep"), ("sleep_efficiency", "efficiency"),
                       ("sleep_minutes_to_fall_asleep", "minutestofallasleep"),
                       ("sleep_time_in_bed", "timeinbed")]:
        feature_meta.append(dict(
            name=name, source=f"sleep.{src}", units="minutes (or %) ",
            notes="mainsleep==True rows only; keyed on sleep_start_day_in_study; daily mean",
        ))
    return dataset.merge(daily, on=[ID_COL, DAY_COL], how="left")


def _add_sleep_score(dataset: pd.DataFrame, data_dir: Path, feature_meta: list[dict]) -> pd.DataFrame:
    cols = [ID_COL, DAY_COL, "overall_score", "deep_sleep_in_minutes", "restlessness"]
    df = _load_columns("sleep_score", cols, data_dir)
    daily = df.groupby([ID_COL, DAY_COL], as_index=False)[
        ["overall_score", "deep_sleep_in_minutes", "restlessness"]
    ].mean()
    daily = daily.rename(columns={
        "overall_score": "sleep_score_overall", "deep_sleep_in_minutes": "sleep_score_deep_minutes",
        "restlessness": "sleep_score_restlessness",
    })
    for name, src in [("sleep_score_overall", "overall_score"),
                       ("sleep_score_deep_minutes", "deep_sleep_in_minutes"),
                       ("sleep_score_restlessness", "restlessness")]:
        feature_meta.append(dict(name=name, source=f"sleep_score.{src}", units="score / minutes",
                                  notes="daily mean"))
    return dataset.merge(daily, on=[ID_COL, DAY_COL], how="left")


def _add_steps(dataset: pd.DataFrame, data_dir: Path, feature_meta: list[dict]) -> pd.DataFrame:
    df = _load_columns("steps", [ID_COL, DAY_COL, "steps"], data_dir)
    daily = df.groupby([ID_COL, DAY_COL], as_index=False)["steps"].sum()
    daily = daily.rename(columns={"steps": "steps_total"})
    feature_meta.append(dict(
        name="steps_total", source="steps.steps", units="steps/day", notes="daily SUM of intraday step counts",
    ))
    return dataset.merge(daily, on=[ID_COL, DAY_COL], how="left")


def _add_active_minutes(dataset: pd.DataFrame, data_dir: Path, feature_meta: list[dict]) -> pd.DataFrame:
    cols = [ID_COL, DAY_COL, "sedentary", "lightly", "moderately", "very"]
    df = _load_columns("active_minutes", cols, data_dir)
    daily = df.groupby([ID_COL, DAY_COL], as_index=False)[["sedentary", "lightly", "moderately", "very"]].mean()
    daily = daily.rename(columns={
        "sedentary": "active_minutes_sedentary", "lightly": "active_minutes_lightly",
        "moderately": "active_minutes_moderately", "very": "active_minutes_very",
    })
    for name, src in [("active_minutes_sedentary", "sedentary"), ("active_minutes_lightly", "lightly"),
                       ("active_minutes_moderately", "moderately"), ("active_minutes_very", "very")]:
        feature_meta.append(dict(name=name, source=f"active_minutes.{src}", units="minutes/day",
                                  notes="already ~daily; mean guards rare duplicate rows"))
    return dataset.merge(daily, on=[ID_COL, DAY_COL], how="left")


def _add_stress_score(dataset: pd.DataFrame, data_dir: Path, feature_meta: list[dict]) -> pd.DataFrame:
    df = _load_columns("stress_score", [ID_COL, DAY_COL, "stress_score"], data_dir)
    daily = df.groupby([ID_COL, DAY_COL], as_index=False)["stress_score"].mean()
    feature_meta.append(dict(name="stress_score", source="stress_score.stress_score", units="score",
                              notes="daily mean"))
    return dataset.merge(daily, on=[ID_COL, DAY_COL], how="left")


def _add_glucose(dataset: pd.DataFrame, data_dir: Path, feature_meta: list[dict]) -> pd.DataFrame:
    df = _load_columns("glucose", [ID_COL, DAY_COL, "glucose_value"], data_dir)
    df["glucose_value"] = pd.to_numeric(df["glucose_value"], errors="coerce")

    is_mmol = df["glucose_value"] < GLUCOSE_MMOL_THRESHOLD
    df.loc[is_mmol, "glucose_value"] = df.loc[is_mmol, "glucose_value"] * GLUCOSE_MMOL_TO_MGDL

    low, high = GLUCOSE_PLAUSIBLE_RANGE
    out_of_range = (df["glucose_value"] < low) | (df["glucose_value"] > high)
    df.loc[out_of_range, "glucose_value"] = np.nan

    grouped = df.groupby([ID_COL, DAY_COL])["glucose_value"]
    daily = grouped.agg(glucose_mean="mean", glucose_std="std").reset_index()
    daily["glucose_cv"] = daily["glucose_std"] / daily["glucose_mean"]

    for name, notes in [
        ("glucose_mean", "daily mean, mg/dL, after mmol/L->mg/dL fix and [20,400] clip to NaN"),
        ("glucose_std", "daily std, mg/dL, same fix applied"),
        ("glucose_cv", "glucose_std / glucose_mean (coefficient of variation)"),
    ]:
        feature_meta.append(dict(name=name, source="glucose.glucose_value", units="mg/dL (mean/std) or unitless (cv)",
                                  notes=notes))
    return dataset.merge(daily, on=[ID_COL, DAY_COL], how="left")


def _add_vo2_max(dataset: pd.DataFrame, data_dir: Path, feature_meta: list[dict]) -> pd.DataFrame:
    df = _load_columns("demographic_vo2_max", [ID_COL, DAY_COL, "demographic_vo2_max"], data_dir)
    daily = df.groupby([ID_COL, DAY_COL], as_index=False)["demographic_vo2_max"].mean()
    feature_meta.append(dict(name="demographic_vo2_max", source="demographic_vo2_max.demographic_vo2_max",
                              units="mL/kg/min", notes="daily mean"))
    return dataset.merge(daily, on=[ID_COL, DAY_COL], how="left")


def _add_static_participant_features(dataset: pd.DataFrame, data_dir: Path, feature_meta: list[dict]) -> pd.DataFrame:
    subject_available = _available_columns("subject-info", data_dir)
    subject_wanted = [ID_COL]
    if "age_of_first_menarche" in subject_available:
        subject_wanted.append("age_of_first_menarche")
    else:
        print("[features] 'age_of_first_menarche' not present in subject-info, skipped")
    if "age" in subject_available:
        subject_wanted.append("age")
        feature_meta.append(dict(name="age", source="subject-info.age", units="years",
                                  notes="static per participant"))
    else:
        print(
            "[features] 'age' not present in subject-info (only 'birth_year' is available, "
            "with no per-row reference date to derive age from) -- skipped, not fabricated"
        )

    if len(subject_wanted) > 1:
        subject = _load_columns("subject-info", subject_wanted, data_dir)
        dataset = dataset.merge(subject, on=ID_COL, how="left")
        if "age_of_first_menarche" in subject.columns:
            feature_meta.append(dict(name="age_of_first_menarche", source="subject-info.age_of_first_menarche",
                                      units="years", notes="static per participant"))

    hw_available = _available_columns("height_and_weight", data_dir)
    hw_wanted = [c for c in [ID_COL, "height_2022", "weight_2022", "height_2024", "weight_2024"]
                 if c in hw_available]
    if len(hw_wanted) > 1:
        hw = _load_columns("height_and_weight", hw_wanted, data_dir)
        height_cm = hw.get("height_2022")
        if "height_2024" in hw.columns:
            height_cm = height_cm.combine_first(hw["height_2024"]) if height_cm is not None else hw["height_2024"]
        weight_kg = hw.get("weight_2022")
        if "weight_2024" in hw.columns:
            weight_kg = weight_kg.combine_first(hw["weight_2024"]) if weight_kg is not None else hw["weight_2024"]
        if height_cm is not None and weight_kg is not None:
            hw["bmi"] = weight_kg / (height_cm / 100.0) ** 2
            dataset = dataset.merge(hw[[ID_COL, "bmi"]], on=ID_COL, how="left")
            feature_meta.append(dict(
                name="bmi", source="height_and_weight.{height,weight}_{2022,2024}", units="kg/m^2",
                notes="static per participant; coalesces 2022 survey then 2024 (cm/kg per dataset README)",
            ))

    return dataset


def _add_per_participant_zscores(dataset: pd.DataFrame, feature_meta: list[dict]) -> pd.DataFrame:
    for col in PARTICIPANT_ZSCORE_COLUMNS:
        if col not in dataset.columns:
            print(f"[features] '{col}' not present, skipping its _pz z-score")
            continue
        pz_col = f"{col}_pz"
        grouped = dataset.groupby(ID_COL)[col]
        participant_mean = grouped.transform("mean")
        participant_std = grouped.transform("std")  # sample std (ddof=1) per participant
        z = (dataset[col] - participant_mean) / participant_std
        z = z.where(participant_std.notna() & (participant_std != 0), np.nan)
        dataset[pz_col] = z
        feature_meta.append(dict(
            name=pz_col, source=f"per-participant z-score of {col}", units="z-score (unitless)",
            notes="(value - participant_mean) / participant_std, mean/std over that participant's "
                  "own days only; NaN if the participant's std is 0 or undefined (e.g. <2 non-NaN days)",
        ))
    return dataset


def _add_rolling_features(dataset: pd.DataFrame, feature_meta: list[dict]) -> pd.DataFrame:
    dataset = dataset.sort_values([ID_COL, DAY_COL]).reset_index(drop=True)
    for base_col, roll_col in [("resting_hr", "resting_hr_roll3"), ("nightly_temperature", "nightly_temperature_roll3")]:
        if base_col in dataset.columns:
            dataset[roll_col] = dataset.groupby(ID_COL)[base_col].transform(
                lambda s: s.rolling(window=3, min_periods=1).mean()
            )
            feature_meta.append(dict(
                name=roll_col, source=f"rolling(3) of {base_col}", units="same as source",
                notes="row-based (not calendar-gap-aware) rolling mean, grouped by id, min_periods=1",
            ))
    return dataset


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_dataset(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    feature_dict_path: str | Path = DEFAULT_FEATURE_DICT_PATH,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    feature_dict_path = Path(feature_dict_path)
    assert data_dir.exists(), f"data dir not found: {data_dir}"

    feature_meta: list[dict] = []

    dataset = _build_label_spine(data_dir, feature_meta)
    dataset = _add_resting_hr(dataset, data_dir, feature_meta)
    dataset = _add_computed_temperature(dataset, data_dir, feature_meta)
    dataset = _add_hrv(dataset, data_dir, feature_meta)
    dataset = _add_respiratory_rate(dataset, data_dir, feature_meta)
    dataset = _add_sleep(dataset, data_dir, feature_meta)
    dataset = _add_sleep_score(dataset, data_dir, feature_meta)
    dataset = _add_steps(dataset, data_dir, feature_meta)
    dataset = _add_active_minutes(dataset, data_dir, feature_meta)
    dataset = _add_stress_score(dataset, data_dir, feature_meta)
    dataset = _add_glucose(dataset, data_dir, feature_meta)
    dataset = _add_vo2_max(dataset, data_dir, feature_meta)
    dataset = _add_static_participant_features(dataset, data_dir, feature_meta)
    dataset = _add_per_participant_zscores(dataset, feature_meta)
    dataset = _add_rolling_features(dataset, feature_meta)

    assert not dataset.duplicated(subset=[ID_COL, DAY_COL]).any(), "duplicate (id, day_in_study) keys in output"
    for col in LEAKAGE_COLUMNS:
        assert col not in dataset.columns, f"leakage column '{col}' present in output"

    print(f"[features] final dataset shape: {dataset.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    print(f"[features] wrote {output_path}")

    _write_feature_dictionary(dataset, feature_meta, feature_dict_path)

    return dataset


def _write_feature_dictionary(dataset: pd.DataFrame, feature_meta: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Feature Dictionary",
        "",
        "One row per (id, day_in_study). Built by `src/features.py` from the mcPHASES raw tables.",
        "",
        "## Reference / index columns (NOT features)",
        "- `id`: participant identifier",
        "- `day_in_study`: normalized day index",
        "- `study_interval`: Interval 1 (Jan-Apr 2022) or Interval 2 (Jul-Oct 2024), from hormones_and_selfreport",
        "",
        "## Target",
        "- `label`: {Menstrual:0, Follicular:1, Fertility:2, Luteal:3}, from hormones_and_selfreport.phase",
        "",
        "## Leakage columns -- never loaded, never present as features",
        f"`{', '.join(LEAKAGE_COLUMNS)}` (the label is derived from `phase`; `lh`/`estrogen`/`pdg` are the "
        "hormone measurements behind the phase call; `flow_volume`/`flow_color` are excluded per spec).",
        "",
        "## Data-quality fixes applied before aggregation",
        f"- **Glucose unit mixing**: `glucose.glucose_value` values < {GLUCOSE_MMOL_THRESHOLD} are assumed "
        f"mmol/L and multiplied by {GLUCOSE_MMOL_TO_MGDL} to convert to mg/dL (the dataset README states the "
        "column is mmol/L, but raw values include a max of 253, physiologically impossible for mmol/L and "
        "consistent with a mixed-unit export). After conversion, values outside "
        f"[{GLUCOSE_PLAUSIBLE_RANGE[0]}, {GLUCOSE_PLAUSIBLE_RANGE[1]}] mg/dL are set to NaN.",
        "- **Sensor dropout sentinels**: `resting_heart_rate.value` and "
        "`respiratory_rate_summary.full_sleep_breathing_rate` use 0 to mean 'no reading' (0 bpm / 0 "
        "breaths-per-minute is not physiologically possible); replaced with NaN before daily aggregation.",
        "",
        "## Ordinal symptom encoding",
        "Text Likert labels mapped to a 0-5 numeric scale (see mcPHASES README):",
        "",
        _markdown_table(["text label", "code"], [[k, v] for k, v in ORDINAL_MAP.items()]),
        "",
        "A few rows contain raw numeric-string codes (e.g. \"2\") instead of text; these are parsed directly "
        "as floats and kept if in [0, 5], else treated as missing.",
        "",
        "## Per-participant normalization",
        f"Each of `{', '.join(PARTICIPANT_ZSCORE_COLUMNS)}` also gets a `_pz` twin: "
        "`(value - participant_mean) / participant_std`, with mean/std computed over that participant's "
        "OWN days only (leakage-safe -- every participant is wholly within one split, so this never uses "
        "another participant's or another split's data). Raw (non-normalized) versions are kept alongside. "
        "`_pz` is NaN wherever the participant's std is 0 or undefined (e.g. fewer than 2 non-NaN days for "
        "that signal). Intent: let the model learn each participant's *relative* deviation from their own "
        "baseline, not just population-level absolute values, which should generalize better across the "
        "held-out participants in val/test.",
        "",
        "## Features",
        "",
        _markdown_table(["feature", "source", "units", "notes"],
                         [[m["name"], m["source"], m["units"], m["notes"]] for m in feature_meta]),
        "",
        "## Known limitations / assumptions",
        "- `age` was NOT derived: subject-info has no `age` column, only `birth_year`, and there is no "
        "per-row reference date to compute age from without guessing -- left out rather than fabricated.",
        "- `bmi` coalesces the 2022 and 2024 height/weight surveys into one static value per participant; "
        "it does not vary by day even though a participant's true weight may have changed between surveys.",
        "- Rolling features (`resting_hr_roll3`, `nightly_temperature_roll3`) are row-based (mean of the "
        "current + up to 2 preceding labeled days for that participant), not calendar-gap-aware -- if a "
        "participant has missing labeled days, the window can span more than 3 calendar days.",
        "- NaNs are preserved throughout (no imputation); coverage varies by signal per reports/eda.md.",
        "",
        f"Final shape: {dataset.shape[0]} rows x {dataset.shape[1]} columns.",
    ]

    out_path.write_text("\n".join(lines))
    print(f"[features] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the mcPHASES cycle-phase modeling table")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--feature-dict", default=str(DEFAULT_FEATURE_DICT_PATH))
    args = parser.parse_args()
    build_dataset(data_dir=args.data_dir, output_path=args.output, feature_dict_path=args.feature_dict)


if __name__ == "__main__":
    main()
