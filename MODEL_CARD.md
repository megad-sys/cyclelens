# Model Card: cycle-phase-benchmark LightGBM baseline

## Model details

- **Type**: LightGBM gradient-boosted trees, multiclass objective (4 classes).
- **Regularization**: `num_leaves=15`, `min_child_samples=40`, `lambda_l1=1.0`, `lambda_l2=1.0`,
  `feature_fraction=0.7`, `bagging_fraction=0.7`, `bagging_freq=1`, `learning_rate=0.03`, up to
  1000 boosting rounds with early stopping (patience 50) on a held-out validation split. Tuned
  specifically to reduce overfitting on ~28 training participants.
- **Class imbalance**: handled via `sklearn.utils.class_weight.compute_sample_weight("balanced", ...)`
  sample weights computed from the training split.
- **Inputs**: 54 daily features per (participant, day) — wearable signals (Fitbit heart rate,
  HRV, temperature, respiratory rate, sleep, steps, stress), CGM glucose, self-reported symptoms
  (ordinal-encoded), static per-participant traits (BMI, age at menarche, VO2max), and
  per-participant z-scored (`_pz`) twins of the continuous physiological signals. Full definitions
  in `reports/feature_dictionary.md`.
- **Output**: one of 4 classes — `Menstrual`, `Follicular`, `Fertility`, `Luteal` — plus a
  probability distribution over all four.
- **Code**: `src/train.py` (training), `src/evaluate.py` (evaluation), `src/crossval.py`
  (cross-validation), `src/explain.py` (SHAP explainability). Seed 42 throughout; deterministic.

## Intended use

A **research baseline and reusable benchmark reference point**, not a finished product: intended
for researchers to reproduce, extend, and try to beat, using the frozen participant-grouped split
in `data/processed/splits.json` and the metric protocol in `reports/benchmark_card.md`. The
`api/main.py` inference layer exists to make the model's behavior legible in a demo (predicted
phase, probabilities, top SHAP drivers, one grounded plain-language sentence), not as a deployable
health product.

**Out of scope**: clinical decision-making, diagnosis, contraceptive or fertility guidance, or any
use where an incorrect phase prediction could inform a medical or reproductive decision.

## Training data

[mcPHASES](https://physionet.org/content/mcphases) (PhysioNet, restricted access, DUA required):
42 participants, Fitbit + Dexcom CGM + Mira fertility-hormone device + daily self-report, spanning
two study intervals (Jan–Apr 2022, Jul–Oct 2024). Labels are hormone-validated (derived from LH,
estrogen, and PDG measurements), not self-reported guesses. See `reports/eda.md` for full
exploratory findings (class balance, missingness, value-sanity checks) and
`reports/feature_dictionary.md` for the exact feature pipeline, including two data-quality fixes
applied before modeling: a glucose mmol/L↔mg/dL unit-mixing correction, and sensor-dropout 0→NaN
sentinel handling for resting heart rate and respiratory rate. The dataset itself is **not**
redistributed in this repository (`data/raw/` is gitignored).

## Evaluation

From `reports/eval_test.json` (frozen held-out test split, 7 participants, 886 participant-days)
and `reports/crossval.json` (5-fold participant-grouped cross-validation, all 42 participants):

| approach | test macro-F1 | 5-fold CV mean ± std macro-F1 |
|---|---|---|
| **model** | **0.368** | **0.366 ± 0.029** |
| personal-majority baseline | 0.280 | 0.233 ± 0.038 |
| global-majority baseline | 0.119 | 0.126 ± 0.007 |

The model beats the personal-majority baseline (the harder of the two, since it already captures
each participant's typical phase distribution) in every one of the 5 cross-validation folds. The
close agreement between the single frozen test split and the 5-fold mean indicates this is a
stable estimate, not a lucky split. Per-class performance is weakest on `Fertility` (F1 0.238 on
test) — the shortest phase, most often confused with the adjacent `Follicular` phase (see
`reports/confusion_test.png`).

Explainability: `reports/shap_importance.json` / `shap_global.png` / `shap_by_class.png`, produced
by `src/explain.py`'s SHAP TreeExplainer over the test split.

## Ethical considerations

- **Small, non-diverse training population**: 42 participants from a single (Canadian) cohort.
  Predictions have not been validated on other populations, age ranges, cycle-length distributions,
  or hormonal conditions (e.g. PCOS, perimenopause) and should not be assumed to generalize.
- **A disclosed confound**: `demographic_vo2_max` (a relatively stable personal trait, not a daily
  physiological signal) ranks among the top-6 global SHAP drivers. This suggests the model is
  partly learning participant identity via a static trait rather than purely the day's hormonal
  signature — disclosed here rather than presented as a clean physiological signal. See
  Limitations in `README.md`.
- **Self-report symptom features** encode subjective, self-reported experience (cramps, mood,
  fatigue, etc.) rather than objective measurement; treating model output as authoritative over a
  person's own reported experience would be inappropriate.
- **No fairness/subgroup analysis** was performed (e.g. by age, BMI, or cycle regularity) given
  the small sample size — this is a known gap, not an implicit claim of fairness.

## Not a medical device

**This model is a research prototype. It is not a medical device, has not been clinically
validated, and must not be used for diagnosis, contraception, fertility planning, or any other
medical or reproductive decision.** The inference API (`api/main.py`) and any conversational
output it generates (`/explain`) always append the disclaimer *"Research prototype, not medical
advice."* — this card is the canonical statement of that same constraint at the model level.

## Caveats and recommendations

- Reproduce results only via the frozen split (`data/processed/splits.json`) and the documented
  metric protocol — comparing against a different split invalidates the comparison.
- Treat the 5-fold CV mean ± std (0.366 ± 0.029), not the single test-split number, as the
  headline generalization estimate.
- Before extending this model (e.g. pooling in more participants, per Future Work in `README.md`),
  re-run `src/eda.py` on the expanded raw data — data-quality issues found here (glucose units,
  sensor sentinels) are dataset-specific and may not hold for new sources.
