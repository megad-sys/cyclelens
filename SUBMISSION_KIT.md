# CycleLens — Submission Kit
_Scripts + description for the 4 deliverables. Read at a natural pace ≈ 1 min each._

---

## Brief project description

**CycleLens — an open benchmark and explainable model for wearable-based menstrual cycle-phase prediction.**

Women's hormonal health is one of the least-represented areas in AI. CycleLens predicts which of the four menstrual cycle phases (menstruation, follicular, fertility/ovulation, luteal) a person is in — from wearable signals alone (temperature, heart-rate variability, resting heart rate, sleep), with no hormone test. We built it on the hormone-validated mcPHASES dataset as a **reusable scientific asset**: a standardized benchmark with participant-grouped, leakage-free train/validation/test splits; a reproducible baseline model that beats naive baselines in every cross-validation fold (0.37 macro-F1, 5-fold participant-grouped CV); SHAP explainability; and a live demo app with a medically-grounded insight layer. Everything — dataset benchmark, model checkpoint, evaluation protocol, and code — is open on GitHub for researchers to reproduce and extend.

- **Code:** https://github.com/megad-sys/cyclelens
- **Backend API:** https://cyclelens.onrender.com (wake with `/health` before demo — free-tier cold start)
- **Live demo:** [Lovable frontend URL — publish from the connected Cycle Insight project]

---

## 1. Team intro video (1 min) — template

Keep it fast and warm. Each person: name → one line of background → their contribution.

> "Hi, we're [team name]. We built CycleLens, an open AI benchmark for women's hormonal health.
> I'm [Name], [background] — I worked on [e.g. the data pipeline and benchmark].
> This is [Name], [background] — [e.g. model training and evaluation].
> [Name] — [e.g. the frontend and demo app].
> [Name] — [e.g. backend, deployment, and the explainability layer].
> Together we turned a raw wearable dataset into a reusable scientific tool in one weekend."

_Fill in real names and contributions. If solo/pair, just cover the roles you played._

---

## 2. Pitch video (1 min) — script

> "Women are more than half the world, yet female physiology is one of the least-studied areas in AI. The menstrual cycle shapes how women feel day to day — but today, phase is guessed from a calendar, which breaks the moment a cycle is irregular, like with PCOS, stress, or perimenopause. The accurate answer, hormone testing, is impractical to do every day.
>
> But here's the thing: the smartwatch already on your wrist passively picks up signals that shift with your hormones — temperature, heart-rate variability, resting heart rate, sleep.
>
> CycleLens predicts your cycle phase from those wearable signals alone — validated against real hormone measurements. And it's not just an app: it's open scientific infrastructure. A standardized benchmark, a reproducible baseline that beats naive baselines in every cross-validation fold, an evaluation protocol future researchers can build on, and an explainable model that tells you not just your phase, but why.
>
> We're turning static, once-a-year hormonal care into something continuous — and giving the research community a foundation to compound on."

---

## 3. Demo video (1 min) — script + what to click

**Before recording:** open the backend URL `/health` first to wake it (free-tier cold start), then open the app. Use **sample days only** — not the sliders.

> "This is CycleLens. It predicts menstrual cycle phase from a smartwatch — no hormone test.
>
> [Select a sample day] I'll pick a real day from our held-out test set — one the model has never seen. Its true phase is **Luteal**.
>
> [Click Predict] The model predicts **Luteal** — correct. These bars are its confidence across the four phases.
>
> [Point at drivers] And it's explainable: it shows *why* — elevated temperature and suppressed heart-rate variability, which is textbook post-ovulation physiology.
>
> [Insight box] Below, a plain-language explanation, grounded in a real medical source with a citation — not an AI guess.
>
> [Optional: pick another sample day, e.g. Menstrual, predict again]
>
> Everything behind this — the dataset benchmark, the model, the evaluation code — is open on GitHub for any researcher to reproduce and extend."

---

## Recording logistics (fast)

- **Screen record:** QuickTime (Mac: File → New Screen Recording) or Loom.
- **Warm the backend** right before recording the demo so there's no cold-start delay.
- **Rehearse once** with a timer — 1 minute is tight; cut words if you run over.
- Record demo in **sample-day mode**; don't touch the sliders on camera.
- Submit: code repo link + 3 video files + this description, before 9:00 AM ET.
