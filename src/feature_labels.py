"""Plain-language names for model features (UI + LLM prompts)."""

from __future__ import annotations

# Static per-participant traits — deprioritize in user-facing narratives (see README confound note).
NARRATIVE_SKIP = frozenset({
    "demographic_vo2_max", "demographic_vo2_max_pz", "bmi", "age_of_first_menarche",
})

FEATURE_LABELS: dict[str, str] = {
    "resting_hr": "resting heart rate",
    "resting_hr_pz": "resting heart rate vs your baseline",
    "resting_hr_roll3": "3-day average resting heart rate",
    "nightly_temperature": "nightly wrist temperature",
    "nightly_temperature_pz": "nightly temperature vs your baseline",
    "nightly_temperature_std": "nightly temperature variability",
    "nightly_temperature_std_pz": "temperature variability vs your baseline",
    "nightly_temperature_roll3": "3-day average nightly temperature",
    "hrv_rmssd": "heart rate variability (HRV)",
    "hrv_rmssd_pz": "HRV vs your baseline",
    "hrv_high_frequency": "HRV high-frequency band",
    "hrv_high_frequency_pz": "HRV high-frequency vs your baseline",
    "hrv_low_frequency": "HRV low-frequency band",
    "hrv_low_frequency_pz": "HRV low-frequency vs your baseline",
    "respiratory_rate": "respiratory rate during sleep",
    "respiratory_rate_pz": "respiratory rate vs your baseline",
    "sleep_score_overall": "sleep score",
    "sleep_score_overall_pz": "sleep score vs your baseline",
    "sleep_minutesasleep": "sleep duration",
    "sleep_efficiency": "sleep efficiency",
    "sleep_score_deep_minutes": "deep sleep minutes",
    "sleep_score_restlessness": "sleep restlessness",
    "steps_total": "daily steps",
    "steps_total_pz": "daily steps vs your baseline",
    "stress_score": "stress score",
    "stress_score_pz": "stress score vs your baseline",
    "glucose_mean": "daily glucose average",
    "glucose_mean_pz": "glucose vs your baseline",
    "glucose_std": "glucose variability",
    "glucose_cv": "glucose coefficient of variation",
    "active_minutes_lightly": "light activity minutes",
    "active_minutes_moderately": "moderate activity minutes",
    "active_minutes_very": "vigorous activity minutes",
    "active_minutes_sedentary": "sedentary minutes",
    "demographic_vo2_max": "cardio fitness (VO₂ max)",
    "demographic_vo2_max_pz": "cardio fitness vs population average",
    "cramps": "reported cramps",
    "fatigue": "reported fatigue",
    "headaches": "reported headaches",
    "bloating": "reported bloating",
    "moodswing": "reported mood swings",
    "stress": "reported stress",
    "appetite": "reported appetite",
    "exerciselevel": "reported exercise level",
    "foodcravings": "reported food cravings",
    "sorebreasts": "reported breast tenderness",
    "sleepissue": "reported sleep issues",
    "indigestion": "reported indigestion",
}


def humanize_feature(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    cleaned = name.removesuffix("_pz").removesuffix("_roll3").replace("_", " ")
    return cleaned


def narrative_drivers(drivers: list[dict], top_k: int = 3) -> list[dict]:
    """Prefer wearable / symptom signals over static confounds for LLM copy."""
    preferred = [d for d in drivers if d.get("feature") not in NARRATIVE_SKIP]
    fallback = [d for d in drivers if d.get("feature") in NARRATIVE_SKIP]
    ordered = preferred + fallback
    return ordered[:top_k]
