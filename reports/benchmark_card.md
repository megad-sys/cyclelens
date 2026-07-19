# Benchmark Card

## Task
4-class daily cycle-phase prediction: given one day's wearable + self-report features for a participant (see reports/feature_dictionary.md), predict which hormone-validated menstrual-cycle phase that day falls in.

## Label mapping
| label | phase |
|---|---|
| 0 | Menstrual |
| 1 | Follicular |
| 2 | Fertility |
| 3 | Luteal |

## Feature list (40 features)
`appetite`, `exerciselevel`, `headaches`, `cramps`, `sorebreasts`, `fatigue`, `sleepissue`, `moodswing`, `stress`, `foodcravings`, `indigestion`, `bloating`, `resting_hr`, `nightly_temperature`, `nightly_temperature_std`, `hrv_rmssd`, `hrv_high_frequency`, `hrv_low_frequency`, `respiratory_rate`, `sleep_minutesasleep`, `sleep_efficiency`, `sleep_minutes_to_fall_asleep`, `sleep_time_in_bed`, `sleep_score_overall`, `sleep_score_deep_minutes`, `sleep_score_restlessness`, `steps_total`, `active_minutes_sedentary`, `active_minutes_lightly`, `active_minutes_moderately`, `active_minutes_very`, `stress_score`, `glucose_mean`, `glucose_std`, `glucose_cv`, `demographic_vo2_max`, `age_of_first_menarche`, `bmi`, `resting_hr_roll3`, `nightly_temperature_roll3`

## Split methodology
Participant-grouped (sklearn `GroupShuffleSplit` on `id`, seed=42), target 70/15/15 train/val/test. To keep all four classes represented in every split, participants are first bucketed by their own majority (most frequent) phase, and the group split is run independently within each bucket before recombining. No participant ID appears in more than one split (buckets with fewer than 4 participants use a deterministic seeded shuffle+slice instead of GroupShuffleSplit, which requires enough groups per side to split meaningfully).

## Split sizes
| split | n participants | n participant-days |
|---|---|---|
| train | 28 | 3622 |
| val | 7 | 1150 |
| test | 7 | 886 |

## Class balance per split
| split | Menstrual | Follicular | Fertility | Luteal |
|---|---|---|---|---|
| train | 694 (19.2%) | 880 (24.3%) | 807 (22.3%) | 1241 (34.3%) |
| val | 214 (18.6%) | 275 (23.9%) | 266 (23.1%) | 395 (34.3%) |
| test | 171 (19.3%) | 231 (26.1%) | 208 (23.5%) | 276 (31.2%) |

## Evaluation metric
- **Primary**: macro-F1 (unweighted mean of per-class F1; treats all 4 phases equally regardless of their class imbalance).
- **Secondary**: per-class F1, balanced accuracy.

## Baselines the model must beat
1. **Global majority class**: fit the single most frequent label on TRAIN, predict it for every row of the evaluation split.
2. **Personal-majority**: for each evaluation day, predict that participant's own most common phase computed leave-one-out over their OTHER days within the same split (excluding the day being predicted). This isolates how much signal comes purely from knowing a person's typical phase distribution, independent of that day's biosignals.
3. **Calendar heuristic (day-within-cycle)**: NOT implemented in this stage -- see Future Work below.

## Future work: calendar heuristic baseline
A day-within-cycle heuristic (e.g. bucket days by time-since-last-menstruation-onset, predict the historically most common phase for that bucket) was not added here because it cannot be derived cleanly from the current tables: there is no explicit cycle-number or cycle-start column, `day_in_study` is a fixed calendar index (not reset per cycle), and study_interval has a real multi-month gap between Interval 1 (2022) and Interval 2 (2024) that would break a naive day-since-onset count for participants who span both. A robust version would need to: (1) detect menstruation-onset days per participant as `phase == Menstrual` where the previous LABELED day's phase != Menstrual, (2) compute day-within-cycle relative to the most recent onset, resetting across the Interval 1/2 gap, and (3) bin day-within-cycle and take the historical majority phase per bin from TRAIN only. That is real feature-engineering work, not a naive baseline, so it is left for a follow-up stage rather than forced in here.

## Correctness invariant
Every split is at the PARTICIPANT level: train/val/test share zero participant ids. Enforced by `tests/test_splits.py`.