# Schema Report

## active_minutes
- rows: 5552
- unique id count: 42
- day_in_study range: [1, 1004]

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| day_in_study | int64 | 100.0% |
| sedentary | float64 | 66.6% |
| lightly | int64 | 100.0% |
| moderately | int64 | 100.0% |
| very | int64 | 100.0% |

---

## active_zone_minutes
- rows: 154482
- unique id count: 42
- day_in_study range: [1, 1004]

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| day_in_study | int64 | 100.0% |
| timestamp | object | 100.0% |
| heart_zone_id | object | 100.0% |
| total_minutes | int64 | 100.0% |

---

## altitude
- rows: 90878
- unique id count: 42
- day_in_study range: [1, 1004]

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| day_in_study | int64 | 100.0% |
| timestamp | object | 100.0% |
| altitude | int64 | 100.0% |

---

## calories
- rows: 20166975
- unique id count: 42
- day_in_study range: [1, 1004]

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| day_in_study | int64 | 100.0% |
| timestamp | object | 100.0% |
| calories | float64 | 100.0% |

---

## computed_temperature
- rows: 5575
- unique id count: 42
- day_in_study range: n/a (no 'day_in_study' column)

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| sleep_start_day_in_study | int64 | 100.0% |
| sleep_start_timestamp | object | 100.0% |
| sleep_end_day_in_study | int64 | 100.0% |
| sleep_end_timestamp | object | 100.0% |
| type | object | 100.0% |
| temperature_samples | int64 | 100.0% |
| nightly_temperature | float64 | 100.0% |
| baseline_relative_sample_sum | float64 | 93.2% |
| baseline_relative_sample_sum_of_squares | float64 | 93.2% |
| baseline_relative_nightly_standard_deviation | float64 | 93.2% |
| baseline_relative_sample_standard_deviation | float64 | 93.2% |

---

## demographic_vo2_max
- rows: 11482
- unique id count: 42
- day_in_study range: [1, 1004]

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| day_in_study | int64 | 100.0% |
| demographic_vo2_max | float64 | 100.0% |
| demographic_vo2_max_error | float64 | 100.0% |
| filtered_demographic_vo2_max | float64 | 100.0% |
| filtered_demographic_vo2_max_error | float64 | 100.0% |

---

## distance
- rows: 7666949
- unique id count: 42
- day_in_study range: [1, 1004]

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| day_in_study | int64 | 100.0% |
| timestamp | object | 100.0% |
| distance | float64 | 100.0% |

---

## estimated_oxygen_variation
- rows: 3070312
- unique id count: 42
- day_in_study range: [1, 1004]

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| day_in_study | int64 | 100.0% |
| timestamp | object | 100.0% |
| infrared_to_red_signal_ratio | float64 | 100.0% |

---

## exercise
- rows: 7282
- unique id count: 31
- day_in_study range: n/a (no 'day_in_study' column)

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| start_day_in_study | int64 | 100.0% |
| start_timestamp | object | 100.0% |
| last_modified_day_in_study | int64 | 100.0% |
| last_modified_timestamp | object | 100.0% |
| original_start_day_in_study | int64 | 100.0% |
| original_start_timestamp | object | 100.0% |
| originalduration | int64 | 100.0% |
| activityname | object | 100.0% |
| activitytypeid | int64 | 100.0% |
| activitylevel | object | 100.0% |
| averageheartrate | float64 | 99.3% |
| calories | int64 | 100.0% |
| duration | int64 | 100.0% |
| activeduration | int64 | 100.0% |
| steps | float64 | 97.6% |
| logtype | object | 100.0% |
| manualvaluesspecified | object | 100.0% |
| heartratezones | object | 99.3% |
| activezoneminutes | object | 100.0% |
| elevationgain | float64 | 90.2% |
| hasgps | bool | 100.0% |
| shouldfetchdetails | bool | 100.0% |
| hasactivezoneminutes | bool | 100.0% |

---

## glucose
- rows: 837130
- unique id count: 42
- day_in_study range: [1, 90]

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| day_in_study | int64 | 100.0% |
| timestamp | object | 100.0% |
| glucose_value | float64 | 99.9% |

---

## heart_rate
- rows: 16756842
- unique id count: 13
- day_in_study range: [1.0, 1004.0]

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | float64 | 100.0% |
| is_weekend | object | 100.0% |
| day_in_study | float64 | 100.0% |
| timestamp | object | 100.0% |
| bpm | float64 | 100.0% |
| confidence | float64 | 100.0% |

---

## heart_rate_variability_details
- rows: 436262
- unique id count: 40
- day_in_study range: [1, 1004]

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| day_in_study | int64 | 100.0% |
| timestamp | object | 100.0% |
| rmssd | float64 | 100.0% |
| coverage | float64 | 100.0% |
| low_frequency | float64 | 100.0% |
| high_frequency | float64 | 100.0% |

---

## height_and_weight
- rows: 42
- unique id count: 42
- day_in_study range: n/a (no 'day_in_study' column)

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| height_2022 | float64 | 50.0% |
| weight_2022 | float64 | 57.1% |
| height_2024 | float64 | 19.0% |
| weight_2024 | float64 | 26.2% |

---

## hormones_and_selfreport
- rows: 5659
- unique id count: 42
- day_in_study range: [1, 1004]

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| day_in_study | int64 | 100.0% |
| phase | object | 100.0% |
| lh | float64 | 94.3% |
| estrogen | float64 | 94.3% |
| pdg | float64 | 32.9% |
| flow_volume | object | 56.4% |
| flow_color | object | 56.4% |
| appetite | object | 58.8% |
| exerciselevel | object | 58.8% |
| headaches | object | 58.8% |
| cramps | object | 58.8% |
| sorebreasts | object | 58.8% |
| fatigue | object | 58.9% |
| sleepissue | object | 58.8% |
| moodswing | object | 58.7% |
| stress | object | 58.8% |
| foodcravings | object | 58.8% |
| indigestion | object | 58.8% |
| bloating | object | 58.8% |

---

## respiratory_rate_summary
- rows: 6301
- unique id count: 40
- day_in_study range: [1, 1004]

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| day_in_study | int64 | 100.0% |
| timestamp | object | 100.0% |
| full_sleep_breathing_rate | float64 | 100.0% |
| full_sleep_standard_deviation | float64 | 100.0% |
| full_sleep_signal_to_noise | float64 | 100.0% |
| deep_sleep_breathing_rate | float64 | 100.0% |
| deep_sleep_standard_deviation | float64 | 100.0% |
| deep_sleep_signal_to_noise | float64 | 100.0% |
| light_sleep_breathing_rate | float64 | 100.0% |
| light_sleep_standard_deviation | float64 | 100.0% |
| light_sleep_signal_to_noise | float64 | 100.0% |
| rem_sleep_breathing_rate | float64 | 100.0% |
| rem_sleep_standard_deviation | float64 | 100.0% |
| rem_sleep_signal_to_noise | float64 | 100.0% |

---

## resting_heart_rate
- rows: 13737
- unique id count: 42
- day_in_study range: [1, 1004]

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| day_in_study | int64 | 100.0% |
| value | float64 | 100.0% |
| error | float64 | 100.0% |

---

## sleep
- rows: 14765
- unique id count: 42
- day_in_study range: n/a (no 'day_in_study' column)

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| sleep_start_day_in_study | int64 | 100.0% |
| sleep_start_timestamp | object | 100.0% |
| sleep_end_day_in_study | int64 | 100.0% |
| sleep_end_timestamp | object | 100.0% |
| duration | int64 | 100.0% |
| minutestofallasleep | int64 | 100.0% |
| minutesasleep | int64 | 100.0% |
| minutesawake | int64 | 100.0% |
| minutesafterwakeup | int64 | 100.0% |
| timeinbed | int64 | 100.0% |
| efficiency | int64 | 100.0% |
| type | object | 100.0% |
| infocode | int64 | 100.0% |
| levels | object | 100.0% |
| mainsleep | bool | 100.0% |

---

## sleep_score
- rows: 5308
- unique id count: 42
- day_in_study range: [1, 1004]

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| day_in_study | int64 | 100.0% |
| timestamp | object | 100.0% |
| overall_score | int64 | 100.0% |
| composition_score | float64 | 70.0% |
| revitalization_score | int64 | 100.0% |
| duration_score | float64 | 70.0% |
| deep_sleep_in_minutes | float64 | 99.9% |
| resting_heart_rate | int64 | 100.0% |
| restlessness | float64 | 100.0% |

---

## steps
- rows: 7666949
- unique id count: 42
- day_in_study range: [1, 1004]

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| day_in_study | int64 | 100.0% |
| timestamp | object | 100.0% |
| steps | int64 | 100.0% |

---

## stress_score
- rows: 7932
- unique id count: 36
- day_in_study range: [1, 1004]

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| day_in_study | int64 | 100.0% |
| timestamp | object | 100.0% |
| stress_score | int64 | 100.0% |
| sleep_points | int64 | 100.0% |
| max_sleep_points | int64 | 100.0% |
| responsiveness_points | int64 | 100.0% |
| max_responsiveness_points | int64 | 100.0% |
| exertion_points | int64 | 100.0% |
| max_exertion_points | int64 | 100.0% |
| status | object | 100.0% |
| calculation_failed | bool | 100.0% |

---

## subject-info
- rows: 42
- unique id count: 42
- day_in_study range: n/a (no 'day_in_study' column)

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| birth_year | int64 | 100.0% |
| gender | object | 100.0% |
| ethnicity | object | 100.0% |
| education | object | 100.0% |
| sexually_active | object | 100.0% |
| self_report_menstrual_health_literacy | object | 97.6% |
| age_of_first_menarche | int64 | 100.0% |

---

## time_in_heart_rate_zones
- rows: 5553
- unique id count: 42
- day_in_study range: [1, 1004]

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| day_in_study | int64 | 100.0% |
| in_default_zone_3 | float64 | 100.0% |
| in_default_zone_2 | float64 | 100.0% |
| in_default_zone_1 | float64 | 100.0% |
| below_default_zone_1 | float64 | 100.0% |

---

## wrist_temperature
- rows: 6856019
- unique id count: 42
- day_in_study range: [1, 1004]

| column | dtype | non-null % |
|---|---|---|
| id | int64 | 100.0% |
| study_interval | int64 | 100.0% |
| is_weekend | bool | 100.0% |
| day_in_study | int64 | 100.0% |
| timestamp | object | 100.0% |
| temperature_diff_from_baseline | float64 | 100.0% |

---
