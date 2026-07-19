import numpy as np
import pandas as pd

from src.features import LEAKAGE_COLUMNS, build_dataset


def _write_fixture(tmp_path):
    ids = [1, 2]
    days = [1, 2, 3, 4, 5]

    hormones = pd.DataFrame({
        "id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "study_interval": [1] * 11,
        "day_in_study": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5],
        "phase": [
            "Menstrual", "Follicular", "Fertility", "Luteal", "Menstrual", None,
            "Luteal", "Fertility", "Follicular", "Menstrual", "Luteal",
        ],
        # leakage columns -- present in the raw table, must NEVER reach the output
        "lh": [1.2, 3.4, 5.6, 7.8, 1.1, 2.2, 1.5, 3.6, 5.8, 7.1, 1.9],
        "estrogen": [0.1] * 11,
        "pdg": [0.2] * 11,
        "flow_volume": ["Not at all"] * 11,
        "flow_color": ["not at all"] * 11,
        "cramps": ["Not at all", "Very Low/Little", "Low", "Moderate", "High", "High",
                   "2", "Very High", "Not at all", "Low", "Moderate"],
        "moodswing": ["Not at all", "Low", "Moderate", "High", "Very High", "High",
                      "Not at all", "Very Low/Little", "Low", "Moderate", "High"],
        "appetite": ["Moderate", "Low", "High", "Very Low", "Not at all", "Low",
                     "Moderate", "High", "Very Low", "Not at all", "Moderate"],
        "stress": ["Low", "Moderate", "High", "Not at all", "Very High", "Moderate",
                   "1", "Low", "Moderate", "High", "Not at all"],
    })
    hormones.to_csv(tmp_path / "hormones_and_selfreport.csv", index=False)

    resting_hr = pd.DataFrame({
        "id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "day_in_study": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "value": [60.0, 62.0, 0.0, 61.0, 63.0, 70.0, 0.0, 71.0, 72.0, 69.0],
    })
    resting_hr.to_csv(tmp_path / "resting_heart_rate.csv", index=False)

    computed_temperature = pd.DataFrame({
        "id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "sleep_start_day_in_study": days + days,
        "nightly_temperature": [33.1, 33.2, 33.4, 33.6, 33.9, 34.0, 34.1, 33.8, 33.7, 34.2],
        "baseline_relative_nightly_standard_deviation": [0.10, 0.12, 0.15, 0.11, 0.14,
                                                           0.20, 0.18, 0.22, 0.19, 0.21],
    })
    computed_temperature.to_csv(tmp_path / "computed_temperature.csv", index=False)

    hrv = pd.DataFrame({
        "id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "day_in_study": [1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "rmssd": [40.0, 44.0, 42.0, 38.0, 35.0, 36.0, 50.0, 48.0, 47.0, 46.0, 45.0],
        "high_frequency": [1.0, 1.2, 1.1, 1.3, 0.9, 1.05, 2.0, 2.1, 1.9, 2.2, 2.05],
        "low_frequency": [2.0, 2.2, 2.1, 2.3, 1.9, 2.05, 3.0, 3.1, 2.9, 3.2, 3.05],
    })
    hrv.to_csv(tmp_path / "heart_rate_variability_details.csv", index=False)

    resp = pd.DataFrame({
        "id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "day_in_study": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "full_sleep_breathing_rate": [15.0, 16.0, 15.5, 0.0, 14.8, 16.2, 15.9, 16.5, 15.1, 14.9],
    })
    resp.to_csv(tmp_path / "respiratory_rate_summary.csv", index=False)

    sleep = pd.DataFrame({
        "id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "sleep_start_day_in_study": [1, 2, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "minutesasleep": [420, 400, 30, 410, 405, 415, 380, 390, 395, 400, 385],
        "efficiency": [90, 91, 50, 89, 92, 90, 88, 87, 86, 89, 90],
        "minutestofallasleep": [10, 12, 5, 11, 9, 10, 15, 14, 13, 12, 11],
        "timeinbed": [450, 430, 40, 440, 435, 445, 410, 420, 425, 430, 415],
        "mainsleep": [True, True, False, True, True, True, True, True, True, True, True],
    })
    sleep.to_csv(tmp_path / "sleep.csv", index=False)

    sleep_score = pd.DataFrame({
        "id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "day_in_study": days + days,
        "overall_score": [80, 82, 78, 81, 83, 75, 77, 79, 76, 78],
        "deep_sleep_in_minutes": [60] * 10,
        "restlessness": [0.1] * 10,
    })
    sleep_score.to_csv(tmp_path / "sleep_score.csv", index=False)

    steps_rows = []
    for pid in ids:
        for day in days:
            base = 1000 + 50 * day + (200 if pid == 2 else 0)  # varies by day and id
            for intraday_steps in (base, base // 2, 500):
                steps_rows.append({"id": pid, "day_in_study": day, "steps": intraday_steps})
    steps = pd.DataFrame(steps_rows)
    steps.to_csv(tmp_path / "steps.csv", index=False)

    active_minutes = pd.DataFrame({
        "id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "day_in_study": days + days,
        "sedentary": [500] * 10,
        "lightly": [200] * 10,
        "moderately": [30] * 10,
        "very": [10] * 10,
    })
    active_minutes.to_csv(tmp_path / "active_minutes.csv", index=False)

    stress_score = pd.DataFrame({
        "id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "day_in_study": days + days,
        "stress_score": [55, 60, 58, 62, 57, 65, 63, 61, 66, 64],
    })
    stress_score.to_csv(tmp_path / "stress_score.csv", index=False)

    glucose = pd.DataFrame({
        "id": [1, 1, 1, 1, 2, 2, 2, 2],
        "day_in_study": [1, 1, 2, 2, 1, 1, 2, 2],
        "glucose_value": [6.5, 95.0, 5.9, 999.0, 100.0, 105.0, 7.0, 110.0],
        # 6.5, 5.9, 7.0 are mmol/L (<30) -> converted; 999.0 is out-of-range after treated as mg/dL -> NaN
    })
    glucose.to_csv(tmp_path / "glucose.csv", index=False)

    vo2 = pd.DataFrame({
        "id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "day_in_study": days + days,
        "demographic_vo2_max": [34.0, 34.5, 35.0, 35.5, 36.0, 30.0, 30.5, 31.0, 31.5, 32.0],
    })
    vo2.to_csv(tmp_path / "demographic_vo2_max.csv", index=False)

    subject_info = pd.DataFrame({
        "id": [1, 2],
        "birth_year": [1990, 1988],
        "age_of_first_menarche": [13, 12],
    })
    subject_info.to_csv(tmp_path / "subject-info.csv", index=False)

    height_and_weight = pd.DataFrame({
        "id": [1, 2],
        "height_2022": [165.0, np.nan],
        "weight_2022": [60.0, np.nan],
        "height_2024": [np.nan, 170.0],
        "weight_2024": [np.nan, 68.0],
    })
    height_and_weight.to_csv(tmp_path / "height_and_weight.csv", index=False)

    return tmp_path


def test_build_dataset_on_synthetic_fixture(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    _write_fixture(data_dir)

    output_path = tmp_path / "processed" / "dataset.parquet"
    feature_dict_path = tmp_path / "reports" / "feature_dictionary.md"

    dataset = build_dataset(data_dir=data_dir, output_path=output_path, feature_dict_path=feature_dict_path)

    # (a) one row per (id, day_in_study), no duplicate keys
    assert not dataset.duplicated(subset=["id", "day_in_study"]).any()
    assert len(dataset) == 10  # 5 valid days x 2 ids; the NaN-phase day 6 for id 1 is dropped

    # (b) label values in {0,1,2,3}
    assert dataset["label"].isin([0, 1, 2, 3]).all()

    # (c) none of the six leakage columns appear as features
    for col in LEAKAGE_COLUMNS:
        assert col not in dataset.columns

    # (d) rolling features never mix two ids: id=2's first labeled day rolls over
    # only its own first value, never id=1's tail values.
    id2_first_row = dataset[(dataset["id"] == 2)].sort_values("day_in_study").iloc[0]
    assert id2_first_row["resting_hr_roll3"] == id2_first_row["resting_hr"]

    # (e) no feature column is 100% NaN
    non_feature_cols = {"id", "day_in_study", "study_interval", "label"}
    for col in dataset.columns:
        if col in non_feature_cols:
            continue
        assert not dataset[col].isna().all(), f"feature column '{col}' is 100% NaN"

    # (f) no 0 values remain in the resting-HR or respiratory-rate features
    assert not (dataset["resting_hr"] == 0).any()
    assert not (dataset["respiratory_rate"] == 0).any()

    # (g) glucose feature values, where present, fall in [20, 400]
    assert dataset["glucose_mean"].dropna().between(20, 400).all()

    # (h) per-participant z-scored (_pz) features have ~0 mean within each participant
    pz_columns = [c for c in dataset.columns if c.endswith("_pz")]
    assert len(pz_columns) == 14, f"expected 14 _pz columns, got {len(pz_columns)}: {pz_columns}"
    for col in pz_columns:
        per_participant_mean = dataset.groupby("id")[col].mean().dropna()
        assert len(per_participant_mean) > 0, f"{col}: no participant has a defined z-score"
        assert np.allclose(per_participant_mean.to_numpy(), 0.0, atol=1e-8), (
            f"{col}: per-participant mean not ~0: {per_participant_mean.to_dict()}"
        )

    assert output_path.exists()
    assert feature_dict_path.exists()
