# Exploratory Data Analysis Report

Read-only investigation. No modeling table built, no columns dropped here.

## 1. Label: Class Balance

Total rows: 5659; unlabeled (NaN phase) dropped from analysis: 1.

### Overall
| phase | count | pct |
|---|---|---|
| Menstrual | 1079 | 19.1% |
| Follicular | 1386 | 24.5% |
| Fertility | 1281 | 22.6% |
| Luteal | 1912 | 33.8% |

### Per participant (id x phase counts)
| id | Menstrual | Follicular | Fertility | Luteal | total |
|---|---|---|---|---|---|
| 1 | 17 | 15 | 20 | 38 | 90 |
| 2 | 8 | 23 | 20 | 39 | 90 |
| 3 | 19 | 16 | 19 | 36 | 90 |
| 4 | 17 | 17 | 20 | 35 | 89 |
| 6 | 26 | 27 | 18 | 19 | 90 |
| 7 | 23 | 10 | 21 | 36 | 90 |
| 8 | 19 | 19 | 21 | 31 | 90 |
| 9 | 28 | 62 | 45 | 55 | 190 |
| 10 | 24 | 45 | 37 | 69 | 175 |
| 11 | 10 | 36 | 19 | 25 | 90 |
| 12 | 42 | 43 | 42 | 65 | 192 |
| 13 | 34 | 55 | 41 | 59 | 189 |
| 14 | 32 | 40 | 45 | 73 | 190 |
| 15 | 23 | 8 | 19 | 40 | 90 |
| 16 | 10 | 27 | 18 | 35 | 90 |
| 18 | 41 | 32 | 56 | 56 | 185 |
| 19 | 19 | 20 | 18 | 33 | 90 |
| 20 | 55 | 28 | 49 | 59 | 191 |
| 22 | 45 | 46 | 47 | 52 | 190 |
| 23 | 19 | 20 | 15 | 36 | 90 |
| 24 | 11 | 32 | 21 | 26 | 90 |
| 26 | 56 | 19 | 48 | 87 | 210 |
| 27 | 42 | 11 | 39 | 100 | 192 |
| 29 | 13 | 11 | 17 | 19 | 60 |
| 30 | 18 | 74 | 29 | 69 | 190 |
| 32 | 25 | 39 | 19 | 41 | 124 |
| 33 | 25 | 88 | 42 | 39 | 194 |
| 34 | 18 | 31 | 19 | 22 | 90 |
| 37 | 13 | 32 | 19 | 26 | 90 |
| 38 | 40 | 45 | 41 | 66 | 192 |
| 39 | 11 | 31 | 22 | 26 | 90 |
| 40 | 18 | 23 | 16 | 33 | 90 |
| 41 | 38 | 40 | 57 | 56 | 191 |
| 42 | 47 | 44 | 59 | 54 | 204 |
| 43 | 15 | 48 | 32 | 91 | 186 |
| 44 | 22 | 17 | 21 | 30 | 90 |
| 45 | 12 | 22 | 18 | 38 | 90 |
| 46 | 15 | 20 | 21 | 34 | 90 |
| 47 | 57 | 48 | 53 | 35 | 193 |
| 48 | 18 | 67 | 39 | 66 | 190 |
| 49 | 10 | 11 | 6 | 11 | 38 |
| 50 | 44 | 44 | 53 | 52 | 193 |

### Sample participant phase timelines
- `reports/eda/phase_timeline_id_26.png`
- `reports/eda/phase_timeline_id_33.png`
- `reports/eda/phase_timeline_id_42.png`

## 2. Categorical Symptom Columns

### cramps
| value | count |
|---|---|
| nan | 2332 |
| Not at all | 1576 |
| Very Low/Little | 937 |
| Moderate | 320 |
| Low | 295 |
| High | 140 |
| Very High | 59 |

### moodswing
| value | count |
|---|---|
| nan | 2339 |
| Not at all | 1032 |
| Very Low/Little | 780 |
| Moderate | 612 |
| Low | 544 |
| High | 268 |
| Very High | 84 |

### appetite
| value | count |
|---|---|
| nan | 2329 |
| Moderate | 1748 |
| Low | 758 |
| High | 580 |
| Very Low | 151 |
| Very High | 89 |
| Not at all | 4 |

### headaches
| value | count |
|---|---|
| nan | 2331 |
| Not at all | 1088 |
| Very Low/Little | 886 |
| Low | 504 |
| Moderate | 502 |
| High | 254 |
| Very High | 86 |
| 2 | 4 |
| 3 | 2 |
| 5 | 1 |
| 4 | 1 |

### sorebreasts
| value | count |
|---|---|
| nan | 2332 |
| Not at all | 1718 |
| Very Low/Little | 882 |
| Low | 350 |
| Moderate | 291 |
| High | 58 |
| Very High | 28 |

### fatigue
| value | count |
|---|---|
| nan | 2328 |
| Moderate | 936 |
| High | 685 |
| Low | 552 |
| Very Low/Little | 473 |
| Not at all | 444 |
| Very High | 241 |

### sleepissue
| value | count |
|---|---|
| nan | 2330 |
| Low | 767 |
| Moderate | 688 |
| Not at all | 662 |
| Very Low/Little | 649 |
| High | 362 |
| Very High | 201 |

### stress
| value | count |
|---|---|
| nan | 2332 |
| Moderate | 1048 |
| High | 610 |
| Low | 567 |
| Not at all | 436 |
| Very Low/Little | 369 |
| Very High | 290 |
| 2 | 4 |
| 3 | 2 |
| 1 | 1 |

### foodcravings
| value | count |
|---|---|
| nan | 2332 |
| Not at all | 922 |
| Low | 642 |
| Moderate | 624 |
| Very Low/Little | 595 |
| High | 391 |
| Very High | 153 |

### indigestion
| value | count |
|---|---|
| nan | 2334 |
| Not at all | 1183 |
| Very Low/Little | 724 |
| Low | 569 |
| Moderate | 528 |
| High | 238 |
| Very High | 83 |

### bloating
| value | count |
|---|---|
| nan | 2331 |
| Not at all | 1094 |
| Very Low/Little | 663 |
| Moderate | 586 |
| Low | 558 |
| High | 335 |
| Very High | 92 |

### exerciselevel
| value | count |
|---|---|
| nan | 2329 |
| Low | 1153 |
| Moderate | 1093 |
| Very Low | 673 |
| High | 347 |
| Very High | 58 |
| Not at all | 6 |

## 3. Join Granularity (rows per id x day key)

| table | day column | rows | unique (id,day) keys | avg rows/key | median | max | % keys already 1-row |
|---|---|---|---|---|---|---|---|
| active_minutes | day_in_study | 5552 | 5481 | 1.0 | 1 | 5 | 98.9% |
| active_zone_minutes | day_in_study | 154482 | 4542 | 34.0 | 19 | 496 | 5.8% |
| altitude | day_in_study | 90878 | 4998 | 18.2 | 12 | 270 | 6.1% |
| calories | day_in_study | 20166975 | 5655 | 3566.2 | 1440 | 36000 | 0.0% |
| computed_temperature | sleep_start_day_in_study | 5575 | 4523 | 1.2 | 1 | 21 | 84.7% |
| demographic_vo2_max | day_in_study | 11482 | 5492 | 2.1 | 1 | 15 | 64.7% |
| distance | day_in_study | 7666949 | 5534 | 1385.4 | 692 | 9898 | 0.0% |
| estimated_oxygen_variation | day_in_study | 3070312 | 5457 | 562.6 | 562 | 5040 | 0.0% |
| exercise | start_day_in_study | 7282 | 1744 | 4.2 | 3 | 28 | 26.4% |
| glucose | day_in_study | 837130 | 3109 | 269.3 | 288 | 576 | 0.0% |
| heart_rate | day_in_study | 16756842 | 1482 | 11306.9 | 11862 | 25518 | 0.0% |
| heart_rate_variability_details | day_in_study | 436262 | 4839 | 90.2 | 90 | 535 | 0.1% |
| height_and_weight | (static, no day column) | 42 | n/a | 1.0 | 1 | 1 | n/a |
| hormones_and_selfreport | day_in_study | 5659 | 5659 | 1.0 | 1 | 1 | 100.0% |
| respiratory_rate_summary | day_in_study | 6301 | 4739 | 1.3 | 1 | 20 | 83.9% |
| resting_heart_rate | day_in_study | 13737 | 5659 | 2.4 | 1 | 20 | 65.4% |
| sleep | sleep_start_day_in_study | 14765 | 5126 | 2.9 | 2 | 25 | 45.9% |
| sleep_score | day_in_study | 5308 | 5078 | 1.0 | 1 | 5 | 95.6% |
| steps | day_in_study | 7666949 | 5534 | 1385.4 | 692 | 9898 | 0.0% |
| stress_score | day_in_study | 7932 | 4239 | 1.9 | 2 | 5 | 43.3% |
| subject-info | (static, no day column) | 42 | n/a | 1.0 | 1 | 1 | n/a |
| time_in_heart_rate_zones | day_in_study | 5553 | 5450 | 1.0 | 1 | 5 | 98.7% |
| wrist_temperature | day_in_study | 6856019 | 5138 | 1334.4 | 1395 | 2880 | 0.0% |

## 4. Value Sanity Checks

_Plausible ranges are approximate literature-based bounds for flagging outliers, not clinical cutoffs._

| signal | table.column | min | median | max | plausible range | n implausible | % implausible |
|---|---|---|---|---|---|---|---|
| resting HR (bpm) | resting_heart_rate.value | 0.0 | 69.8 | 89.3 | [25, 180] | 1380 | 10.0% |
| nightly temperature | computed_temperature.nightly_temperature | 25.7 | 33.9 | 36.4 | [-15, 115] | 0 | 0.0% |
| glucose (mg/dL) | glucose.glucose_value | 2.2 | 6.1 | 253.0 | [20, 600] | 801423 | 95.9% |
| HRV rmssd (ms) | heart_rate_variability_details.rmssd | 0.0 | 46.4 | 894.3 | [0, 300] | 201 | 0.0% |
| respiratory rate (breaths/min) | respiratory_rate_summary.full_sleep_breathing_rate | 0.0 | 15.6 | 29.8 | [5, 40] | 137 | 2.2% |
| sleep minutes asleep | sleep.minutesasleep | 56.0 | 413.0 | 1125.0 | [0, 900] | 61 | 0.5% |

## 5. Coverage After Join (onto the label spine)

Label spine: 5658 labeled (id, day_in_study) rows.

| signal | non-null days | labeled days | coverage % |
|---|---|---|---|
| resting HR (bpm) | 5658 | 5658 | 100.0% |
| nightly temperature | 4507 | 5658 | 79.7% |
| glucose (mg/dL) | 3108 | 5658 | 54.9% |
| HRV rmssd (ms) | 4838 | 5658 | 85.5% |
| respiratory rate (breaths/min) | 4738 | 5658 | 83.7% |
| sleep minutes asleep | 5038 | 5658 | 89.0% |

## 6. Signal Check: mean +/- std by phase

### resting HR (bpm)
| phase | mean | std | n |
|---|---|---|---|
| Menstrual | 57.04 | 24.85 | 1079 |
| Follicular | 59.59 | 21.60 | 1386 |
| Fertility | 58.48 | 24.56 | 1281 |
| Luteal | 58.02 | 26.12 | 1912 |

### nightly temperature
| phase | mean | std | n |
|---|---|---|---|
| Menstrual | 33.82 | 0.94 | 854 |
| Follicular | 33.63 | 0.80 | 1124 |
| Fertility | 33.74 | 0.91 | 1026 |
| Luteal | 33.93 | 0.94 | 1503 |

### HRV rmssd (ms)
| phase | mean | std | n |
|---|---|---|---|
| Menstrual | 58.90 | 29.94 | 912 |
| Follicular | 56.93 | 28.43 | 1205 |
| Fertility | 53.71 | 28.58 | 1089 |
| Luteal | 52.44 | 27.58 | 1632 |

### glucose (mg/dL)
| phase | mean | std | n |
|---|---|---|---|
| Menstrual | 10.32 | 21.49 | 557 |
| Follicular | 14.15 | 28.53 | 811 |
| Fertility | 11.38 | 23.80 | 709 |
| Luteal | 9.86 | 20.33 | 1031 |
