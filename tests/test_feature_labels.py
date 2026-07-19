from src.feature_labels import humanize_feature, narrative_drivers


def test_humanize_vo2_confound():
    assert "VO₂" in humanize_feature("demographic_vo2_max_pz") or "cardio" in humanize_feature("demographic_vo2_max_pz")


def test_narrative_drivers_deprioritizes_static_traits():
    drivers = [
        {"feature": "demographic_vo2_max_pz", "shap": 0.5},
        {"feature": "hrv_rmssd", "shap": 0.3},
        {"feature": "resting_hr", "shap": 0.2},
    ]
    ordered = narrative_drivers(drivers)
    assert ordered[0]["feature"] == "hrv_rmssd"
