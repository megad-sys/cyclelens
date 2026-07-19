import pandas as pd

from src.eda import run_eda


def _write_fixture(tmp_path):
    hormones = pd.DataFrame({
        "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
        "day_in_study": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3],
        "phase": [
            "Menstrual", "Follicular", "Fertility", "Luteal",
            "Menstrual", "Follicular", "Fertility", None,
            "Menstrual", "Follicular", "Luteal",
        ],
        "cramps": ["Mild", "None", "None", "None", "Severe", "None", "None", "None", "Mild", "None", "None"],
        "moodswing": ["Yes", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes"],
    })
    hormones.to_csv(tmp_path / "hormones_and_selfreport.csv", index=False)

    resting_hr = pd.DataFrame({
        "id": [1, 1, 2, 2, 3],
        "day_in_study": [1, 2, 1, 2, 1],
        "value": [60.0, 62.0, 300.0, 58.0, 65.0],  # 300 is an implausible outlier
    })
    resting_hr.to_csv(tmp_path / "resting_heart_rate.csv", index=False)

    computed_temperature = pd.DataFrame({
        "id": [1, 1, 2, 3],
        "sleep_start_day_in_study": [1, 2, 1, 1],
        "nightly_temperature": [33.1, 33.4, 34.0, 32.8],
    })
    computed_temperature.to_csv(tmp_path / "computed_temperature.csv", index=False)

    glucose = pd.DataFrame({
        "id": [1, 1, 2, 3],
        "day_in_study": [1, 1, 1, 1],
        "glucose_value": [95.0, 101.0, 88.0, 900.0],  # 900 is implausible
    })
    glucose.to_csv(tmp_path / "glucose.csv", index=False)

    hrv = pd.DataFrame({
        "id": [1, 1, 2, 3],
        "day_in_study": [1, 2, 1, 1],
        "rmssd": [40.0, 42.0, 35.0, 38.0],
    })
    hrv.to_csv(tmp_path / "heart_rate_variability_details.csv", index=False)

    resp = pd.DataFrame({
        "id": [1, 2, 3],
        "day_in_study": [1, 1, 1],
        "full_sleep_breathing_rate": [15.0, 16.5, 14.2],
    })
    resp.to_csv(tmp_path / "respiratory_rate_summary.csv", index=False)

    sleep = pd.DataFrame({
        "id": [1, 1, 2, 3],
        "sleep_start_day_in_study": [1, 2, 1, 1],
        "minutesasleep": [420, 400, 380, 450],
        "mainsleep": [True, True, True, True],
    })
    sleep.to_csv(tmp_path / "sleep.csv", index=False)

    height_and_weight = pd.DataFrame({
        "id": [1, 2, 3],
        "height_2022": [165.0, 170.0, 160.0],
        "weight_2022": [60.0, 70.0, 55.0],
    })
    height_and_weight.to_csv(tmp_path / "height_and_weight.csv", index=False)

    return tmp_path


def test_run_eda_creates_report_and_plots(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    _write_fixture(data_dir)

    reports_dir = tmp_path / "reports"
    plots_dir = tmp_path / "reports" / "eda"

    out_path = run_eda(data_dir=data_dir, reports_dir=reports_dir, plots_dir=plots_dir)

    assert out_path.exists()
    content = out_path.read_text()
    assert "# Exploratory Data Analysis Report" in content
    assert "## 1. Label: Class Balance" in content
    assert "## 2. Categorical Symptom Columns" in content
    assert "## 3. Join Granularity" in content
    assert "## 4. Value Sanity Checks" in content
    assert "## 5. Coverage After Join" in content
    assert "## 6. Signal Check" in content

    png_files = list(plots_dir.glob("phase_timeline_id_*.png"))
    assert len(png_files) > 0
