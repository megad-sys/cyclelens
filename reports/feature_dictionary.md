# Feature Dictionary

One row per (id, day_in_study). Built by `src/features.py` from the mcPHASES raw tables.

## Reference / index columns (NOT features)
- `id`: participant identifier
- `day_in_study`: normalized day index
- `study_interval`: Interval 1 (Jan-Apr 2022) or Interval 2 (Jul-Oct 2024), from hormones_and_selfreport

## Target
- `label`: {Menstrual:0, Follicular:1, Fertility:2, Luteal:3}, from hormones_and_selfreport.phase

## Leakage columns -- never loaded, never present as features
`phase, lh, estrogen, pdg, flow_volume, flow_color` (the label is derived from `phase`; `lh`/`estrogen`/`pdg` are the hormone measurements behind the phase call; `flow_volume`/`flow_color` are excluded per spec).

## Data-quality fixes applied before aggregation
- **Glucose unit mixing**: `glucose.glucose_value` values < 30.0 are assumed mmol/L and multiplied by 18.0182 to convert to mg/dL (the dataset README states the column is mmol/L, but raw values include a max of 253, physiologically impossible for mmol/L and consistent with a mixed-unit export). After conversion, values outside [20.0, 400.0] mg/dL are set to NaN.
- **Sensor dropout sentinels**: `resting_heart_rate.value` and `respiratory_rate_summary.full_sleep_breathing_rate` use 0 to mean 'no reading' (0 bpm / 0 breaths-per-minute is not physiologically possible); replaced with NaN before daily aggregation.

## Ordinal symptom encoding
Text Likert labels mapped to a 0-5 numeric scale (see mcPHASES README):

| text label | code |
|---|---|
| not at all | 0.0 |
| very low | 1.0 |
| very low/little | 1.0 |
| low | 2.0 |
| moderate | 3.0 |
| high | 4.0 |
| very high | 5.0 |

A few rows contain raw numeric-string codes (e.g. "2") instead of text; these are parsed directly as floats and kept if in [0, 5], else treated as missing.

## Per-participant normalization
Each of `resting_hr, nightly_temperature, nightly_temperature_std, hrv_rmssd, hrv_high_frequency, hrv_low_frequency, respiratory_rate, sleep_minutesasleep, sleep_efficiency, sleep_score_overall, steps_total, stress_score, glucose_mean, demographic_vo2_max` also gets a `_pz` twin: `(value - participant_mean) / participant_std`, with mean/std computed over that participant's OWN days only (leakage-safe -- every participant is wholly within one split, so this never uses another participant's or another split's data). Raw (non-normalized) versions are kept alongside. `_pz` is NaN wherever the participant's std is 0 or undefined (e.g. fewer than 2 non-NaN days for that signal). Intent: let the model learn each participant's *relative* deviation from their own baseline, not just population-level absolute values, which should generalize better across the held-out participants in val/test.

## Features

| feature | source | units | notes |
|---|---|---|---|
| appetite | hormones_and_selfreport.appetite | ordinal 0-5 (Likert) | text label mapped via ORDINAL_MAP; stray numeric-string codes parsed directly |
| exerciselevel | hormones_and_selfreport.exerciselevel | ordinal 0-5 (Likert) | text label mapped via ORDINAL_MAP; stray numeric-string codes parsed directly |
| headaches | hormones_and_selfreport.headaches | ordinal 0-5 (Likert) | text label mapped via ORDINAL_MAP; stray numeric-string codes parsed directly |
| cramps | hormones_and_selfreport.cramps | ordinal 0-5 (Likert) | text label mapped via ORDINAL_MAP; stray numeric-string codes parsed directly |
| sorebreasts | hormones_and_selfreport.sorebreasts | ordinal 0-5 (Likert) | text label mapped via ORDINAL_MAP; stray numeric-string codes parsed directly |
| fatigue | hormones_and_selfreport.fatigue | ordinal 0-5 (Likert) | text label mapped via ORDINAL_MAP; stray numeric-string codes parsed directly |
| sleepissue | hormones_and_selfreport.sleepissue | ordinal 0-5 (Likert) | text label mapped via ORDINAL_MAP; stray numeric-string codes parsed directly |
| moodswing | hormones_and_selfreport.moodswing | ordinal 0-5 (Likert) | text label mapped via ORDINAL_MAP; stray numeric-string codes parsed directly |
| stress | hormones_and_selfreport.stress | ordinal 0-5 (Likert) | text label mapped via ORDINAL_MAP; stray numeric-string codes parsed directly |
| foodcravings | hormones_and_selfreport.foodcravings | ordinal 0-5 (Likert) | text label mapped via ORDINAL_MAP; stray numeric-string codes parsed directly |
| indigestion | hormones_and_selfreport.indigestion | ordinal 0-5 (Likert) | text label mapped via ORDINAL_MAP; stray numeric-string codes parsed directly |
| bloating | hormones_and_selfreport.bloating | ordinal 0-5 (Likert) | text label mapped via ORDINAL_MAP; stray numeric-string codes parsed directly |
| resting_hr | resting_heart_rate.value | bpm | 0 (sensor dropout sentinel) replaced with NaN before daily mean |
| nightly_temperature | computed_temperature.nightly_temperature | degrees C | keyed on sleep_start_day_in_study -> day_in_study; daily mean |
| nightly_temperature_std | computed_temperature.baseline_relative_nightly_standard_deviation | degrees C | Fitbit-computed within-night std relative to personal baseline; daily mean |
| hrv_rmssd | heart_rate_variability_details.rmssd | ms (rmssd) / power (freq bands) | daily mean of 5-minute sleep-window recordings |
| hrv_high_frequency | heart_rate_variability_details.high_frequency | ms (rmssd) / power (freq bands) | daily mean of 5-minute sleep-window recordings |
| hrv_low_frequency | heart_rate_variability_details.low_frequency | ms (rmssd) / power (freq bands) | daily mean of 5-minute sleep-window recordings |
| respiratory_rate | respiratory_rate_summary.full_sleep_breathing_rate | breaths/min | 0 (sensor dropout sentinel) replaced with NaN before daily mean |
| sleep_minutesasleep | sleep.minutesasleep | minutes (or %)  | mainsleep==True rows only; keyed on sleep_start_day_in_study; daily mean |
| sleep_efficiency | sleep.efficiency | minutes (or %)  | mainsleep==True rows only; keyed on sleep_start_day_in_study; daily mean |
| sleep_minutes_to_fall_asleep | sleep.minutestofallasleep | minutes (or %)  | mainsleep==True rows only; keyed on sleep_start_day_in_study; daily mean |
| sleep_time_in_bed | sleep.timeinbed | minutes (or %)  | mainsleep==True rows only; keyed on sleep_start_day_in_study; daily mean |
| sleep_score_overall | sleep_score.overall_score | score / minutes | daily mean |
| sleep_score_deep_minutes | sleep_score.deep_sleep_in_minutes | score / minutes | daily mean |
| sleep_score_restlessness | sleep_score.restlessness | score / minutes | daily mean |
| steps_total | steps.steps | steps/day | daily SUM of intraday step counts |
| active_minutes_sedentary | active_minutes.sedentary | minutes/day | already ~daily; mean guards rare duplicate rows |
| active_minutes_lightly | active_minutes.lightly | minutes/day | already ~daily; mean guards rare duplicate rows |
| active_minutes_moderately | active_minutes.moderately | minutes/day | already ~daily; mean guards rare duplicate rows |
| active_minutes_very | active_minutes.very | minutes/day | already ~daily; mean guards rare duplicate rows |
| stress_score | stress_score.stress_score | score | daily mean |
| glucose_mean | glucose.glucose_value | mg/dL (mean/std) or unitless (cv) | daily mean, mg/dL, after mmol/L->mg/dL fix and [20,400] clip to NaN |
| glucose_std | glucose.glucose_value | mg/dL (mean/std) or unitless (cv) | daily std, mg/dL, same fix applied |
| glucose_cv | glucose.glucose_value | mg/dL (mean/std) or unitless (cv) | glucose_std / glucose_mean (coefficient of variation) |
| demographic_vo2_max | demographic_vo2_max.demographic_vo2_max | mL/kg/min | daily mean |
| age_of_first_menarche | subject-info.age_of_first_menarche | years | static per participant |
| bmi | height_and_weight.{height,weight}_{2022,2024} | kg/m^2 | static per participant; coalesces 2022 survey then 2024 (cm/kg per dataset README) |
| resting_hr_pz | per-participant z-score of resting_hr | z-score (unitless) | (value - participant_mean) / participant_std, mean/std over that participant's own days only; NaN if the participant's std is 0 or undefined (e.g. <2 non-NaN days) |
| nightly_temperature_pz | per-participant z-score of nightly_temperature | z-score (unitless) | (value - participant_mean) / participant_std, mean/std over that participant's own days only; NaN if the participant's std is 0 or undefined (e.g. <2 non-NaN days) |
| nightly_temperature_std_pz | per-participant z-score of nightly_temperature_std | z-score (unitless) | (value - participant_mean) / participant_std, mean/std over that participant's own days only; NaN if the participant's std is 0 or undefined (e.g. <2 non-NaN days) |
| hrv_rmssd_pz | per-participant z-score of hrv_rmssd | z-score (unitless) | (value - participant_mean) / participant_std, mean/std over that participant's own days only; NaN if the participant's std is 0 or undefined (e.g. <2 non-NaN days) |
| hrv_high_frequency_pz | per-participant z-score of hrv_high_frequency | z-score (unitless) | (value - participant_mean) / participant_std, mean/std over that participant's own days only; NaN if the participant's std is 0 or undefined (e.g. <2 non-NaN days) |
| hrv_low_frequency_pz | per-participant z-score of hrv_low_frequency | z-score (unitless) | (value - participant_mean) / participant_std, mean/std over that participant's own days only; NaN if the participant's std is 0 or undefined (e.g. <2 non-NaN days) |
| respiratory_rate_pz | per-participant z-score of respiratory_rate | z-score (unitless) | (value - participant_mean) / participant_std, mean/std over that participant's own days only; NaN if the participant's std is 0 or undefined (e.g. <2 non-NaN days) |
| sleep_minutesasleep_pz | per-participant z-score of sleep_minutesasleep | z-score (unitless) | (value - participant_mean) / participant_std, mean/std over that participant's own days only; NaN if the participant's std is 0 or undefined (e.g. <2 non-NaN days) |
| sleep_efficiency_pz | per-participant z-score of sleep_efficiency | z-score (unitless) | (value - participant_mean) / participant_std, mean/std over that participant's own days only; NaN if the participant's std is 0 or undefined (e.g. <2 non-NaN days) |
| sleep_score_overall_pz | per-participant z-score of sleep_score_overall | z-score (unitless) | (value - participant_mean) / participant_std, mean/std over that participant's own days only; NaN if the participant's std is 0 or undefined (e.g. <2 non-NaN days) |
| steps_total_pz | per-participant z-score of steps_total | z-score (unitless) | (value - participant_mean) / participant_std, mean/std over that participant's own days only; NaN if the participant's std is 0 or undefined (e.g. <2 non-NaN days) |
| stress_score_pz | per-participant z-score of stress_score | z-score (unitless) | (value - participant_mean) / participant_std, mean/std over that participant's own days only; NaN if the participant's std is 0 or undefined (e.g. <2 non-NaN days) |
| glucose_mean_pz | per-participant z-score of glucose_mean | z-score (unitless) | (value - participant_mean) / participant_std, mean/std over that participant's own days only; NaN if the participant's std is 0 or undefined (e.g. <2 non-NaN days) |
| demographic_vo2_max_pz | per-participant z-score of demographic_vo2_max | z-score (unitless) | (value - participant_mean) / participant_std, mean/std over that participant's own days only; NaN if the participant's std is 0 or undefined (e.g. <2 non-NaN days) |
| resting_hr_roll3 | rolling(3) of resting_hr | same as source | row-based (not calendar-gap-aware) rolling mean, grouped by id, min_periods=1 |
| nightly_temperature_roll3 | rolling(3) of nightly_temperature | same as source | row-based (not calendar-gap-aware) rolling mean, grouped by id, min_periods=1 |

## Known limitations / assumptions
- `age` was NOT derived: subject-info has no `age` column, only `birth_year`, and there is no per-row reference date to compute age from without guessing -- left out rather than fabricated.
- `bmi` coalesces the 2022 and 2024 height/weight surveys into one static value per participant; it does not vary by day even though a participant's true weight may have changed between surveys.
- Rolling features (`resting_hr_roll3`, `nightly_temperature_roll3`) are row-based (mean of the current + up to 2 preceding labeled days for that participant), not calendar-gap-aware -- if a participant has missing labeled days, the window can span more than 3 calendar days.
- NaNs are preserved throughout (no imputation); coverage varies by signal per reports/eda.md.

Final shape: 5658 rows x 58 columns.