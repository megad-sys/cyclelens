# CLAUDE.md — Project rules (read this first, every session)

## What we're building
An open, reproducible ML **benchmark + baseline model** for **wearable-based menstrual
cycle-phase prediction** using the mcPHASES dataset (4 classes: menstruation,
late_follicular, ovulation, luteal), evaluated against hormone-validated labels.
Plus a thin FastAPI + demo layer. The deliverable is a *reusable scientific asset*,
not just a prototype.

(Fallback problem if PhysioNet DUA isn't ready: NHANES 2017-2020 menopause-stage
classification. Same architecture; only the loader + label logic change.)

## How we work — INCREMENTAL, one stage at a time
- The human will paste ONE stage at a time, each with an explicit acceptance test.
- Do ONLY that stage. Create ONLY the files named in it. When the acceptance test
  passes, print its output and STOP. Do NOT start the next stage on your own.
- Keep files small and named exactly as specified so edits stay surgical.

## Engineering rules
- Priorities: correctness, reproducibility, no data leakage, clean tests — NOT speed
  or model size.
- seed=42 everywhere. Deterministic scripts.
- Every script: argparse CLI, logs the shape/counts of what it produces, and fails
  loudly (assert) on malformed input. No silent except/pass.
- Tests use pytest under tests/, run on tiny SYNTHETIC fixtures in <10s. Never
  test against the full real dataset.
- Stack: pandas, pyarrow, pyreadstat, numpy, scikit-learn, lightgbm, shap,
  matplotlib, fastapi, uvicorn, pydantic, openai, tavily-python, python-dotenv,
  pytest. Pin versions in requirements.txt.

## Two NON-NEGOTIABLE correctness rules
1. **Split by participant, never by day.** Train/val/test must share zero
   participant ids (GroupShuffleSplit). This is enforced by a test — keep it green.
2. **Keep the whole pytest suite green.** Run it after every stage; if red, fix
   before moving on.

## Compute & cost discipline (usage limit)
- Raw data lives at ./data/raw/ and is NOT redistributed (PhysioNet DUA).
- You WRITE code and unit-test it on synthetic fixtures — that's the main loop.
- Heavy/long runs (full-data profiling, model training, full-test SHAP) are meant to
  be run by the human in Google Colab. Write those scripts; don't execute long jobs
  yourself.
- Quick sanity checks on real data (row counts, label distribution, a single pipeline
  run to confirm sane output) ARE fine and encouraged — just don't loop on them.

## Repo hygiene
- .gitignore must exclude data/raw/, *.parquet, model checkpoints, and .env.
- Never stage raw data or secrets. git status should show no data files.
