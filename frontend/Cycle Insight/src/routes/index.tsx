import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";

export const Route = createFileRoute("/")({
  component: CycleLens,
});

// ============================================================
// API contract
// ============================================================
const API_BASE_URL = "https://cyclelens.onrender.com";
const USE_MOCK = false;

type PhaseLabel = "Menstrual" | "Follicular" | "Fertility" | "Luteal";

type Features = {
  nightly_temperature: number;
  hrv_rmssd: number;
  resting_hr: number;
  sleep_score_overall: number;
  respiratory_rate: number;
  glucose_mean: number;
  steps_total: number;
  cramps: number;
};

type Driver = { feature: string; direction: "up" | "down"; weight: number };

type PredictResponse = {
  phase_label: PhaseLabel;
  probabilities: Record<PhaseLabel, number>;
  top_drivers: Driver[];
};

type ExplainResponse = {
  sentence: string;
  source_title: string | null;
  source_url: string | null;
};
type AskResponse = { answer: string; source_url: string };

// ---------- Mock backend ----------
function mockPredict(features: Features): PredictResponse {
  // Simple heuristics so predictions vary with sliders.
  const t = features.nightly_temperature - 34.0; // deviation from an approximate population baseline
  const hrv = features.hrv_rmssd;
  const rhr = features.resting_hr;
  const cramps = features.cramps;

  let probs: Record<PhaseLabel, number>;
  if (cramps >= 2) probs = { Menstrual: 0.62, Follicular: 0.18, Fertility: 0.06, Luteal: 0.14 };
  else if (t >= 0.2 && hrv < 60) probs = { Menstrual: 0.06, Follicular: 0.18, Fertility: 0.18, Luteal: 0.58 };
  else if (t < 0 && hrv >= 65) probs = { Menstrual: 0.08, Follicular: 0.52, Fertility: 0.28, Luteal: 0.12 };
  else if (t >= 0 && t < 0.2 && rhr <= 62) probs = { Menstrual: 0.08, Follicular: 0.22, Fertility: 0.52, Luteal: 0.18 };
  else probs = { Menstrual: 0.15, Follicular: 0.3, Fertility: 0.25, Luteal: 0.3 };

  const label = (Object.entries(probs) as [PhaseLabel, number][]).sort((a, b) => b[1] - a[1])[0][0];

  const drivers: Driver[] = [
    { feature: "nightly temp", direction: t >= 0 ? "up" : "down", weight: Math.min(0.4, Math.abs(t) + 0.05) },
    { feature: "HRV rmssd", direction: hrv < 60 ? "down" : "up", weight: 0.22 },
    { feature: "resting HR", direction: rhr >= 62 ? "up" : "down", weight: 0.14 },
  ];
  return { phase_label: label, probabilities: probs, top_drivers: drivers };
}

function mockExplain(res: PredictResponse): ExplainResponse {
  const map: Record<PhaseLabel, string> = {
    Menstrual:
      "Higher cramps and lower nightly temperature align with early follicular bleeding days.",
    Follicular:
      "Cooler nightly temperature and higher HRV are consistent with the follicular phase.",
    Fertility:
      "A rising nightly temperature with elevated HRV and lower resting HR is characteristic of the peri-ovulatory window.",
    Luteal:
      "Elevated nightly temperature and suppressed HRV are consistent with the post-ovulatory luteal phase.",
  };
  return {
    sentence: map[res.phase_label],
    source_title: "ACOG — the menstrual cycle",
    source_url: "https://www.acog.org",
  };
}

function mockAsk(question: string): AskResponse {
  return {
    answer:
      "Based on the current wearable readings, the model puts you closer to the luteal phase than the fertile window. This is a research prototype and not medical advice — you asked: “" +
      question +
      "”.",
    source_url: "https://www.acog.org",
  };
}

async function apiPredictFeatures(features: Record<string, number>): Promise<PredictResponse> {
  if (USE_MOCK) {
    await new Promise((r) => setTimeout(r, 350));
    return mockPredict(features as Features);
  }
  const r = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
  if (!r.ok) throw new Error("Prediction failed");
  return r.json();
}

async function apiPredict(features: Features): Promise<PredictResponse> {
  return apiPredictFeatures(features);
}

async function apiExplain(res: PredictResponse): Promise<ExplainResponse> {
  if (USE_MOCK) {
    await new Promise((r) => setTimeout(r, 250));
    return mockExplain(res);
  }
  const r = await fetch(`${API_BASE_URL}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ phase_label: res.phase_label, top_drivers: res.top_drivers }),
  });
  if (!r.ok) throw new Error("Explain failed");
  return r.json();
}

async function apiAsk(question: string, features: Features): Promise<AskResponse> {
  if (USE_MOCK) {
    await new Promise((r) => setTimeout(r, 400));
    return mockAsk(question);
  }
  const r = await fetch(`${API_BASE_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, features }),
  });
  if (!r.ok) throw new Error("Ask failed");
  return r.json();
}

// ============================================================
// Sample days — 3 real days.
// display: values shown on sliders. full_features: sent to /predict.
// ============================================================
type SampleDay = {
  id: string;
  true_phase: PhaseLabel;
  display: Features;
  full_features: Record<string, number>;
};

// Real held-out test-set days (from data/processed/dataset.parquet).
// display = friendly slider view; full_features = the exact 54-feature row sent to /predict.
const SAMPLE_DAYS: SampleDay[] = [
  {
    id: "s1",
    true_phase: "Menstrual",
    display: {
      nightly_temperature: 34.56,
      hrv_rmssd: 60,
      resting_hr: 63,
      sleep_score_overall: 75,
      respiratory_rate: 18.8,
      glucose_mean: 120,
      steps_total: 3572,
      cramps: 0,
    },
    full_features: {
      appetite: 1, exerciselevel: 2, headaches: 4, cramps: 0, sorebreasts: 0, fatigue: 4,
      sleepissue: 3, moodswing: 5, stress: 1, foodcravings: 1, indigestion: 0, bloating: 4,
      resting_hr: 62.892, nightly_temperature: 34.556, nightly_temperature_std: 1.032,
      hrv_rmssd: 60.198, hrv_high_frequency: 844.83, hrv_low_frequency: 1035.421,
      respiratory_rate: 18.8, sleep_minutesasleep: 501, sleep_efficiency: 94,
      sleep_minutes_to_fall_asleep: 0, sleep_time_in_bed: 569, sleep_score_overall: 75,
      sleep_score_deep_minutes: 61, sleep_score_restlessness: 0.104, steps_total: 3572,
      active_minutes_sedentary: 832, active_minutes_lightly: 169, active_minutes_moderately: 0,
      active_minutes_very: 0, stress_score: 75, glucose_mean: 120.078, glucose_std: 16.654,
      glucose_cv: 0.139, demographic_vo2_max: 42.2, age_of_first_menarche: 11, bmi: 21.967,
      resting_hr_pz: -1.635, nightly_temperature_pz: 1.003, nightly_temperature_std_pz: 1.641,
      hrv_rmssd_pz: -0.721, hrv_high_frequency_pz: -0.783, hrv_low_frequency_pz: -1.018,
      respiratory_rate_pz: 0.328, sleep_minutesasleep_pz: 1.326, sleep_efficiency_pz: -0.059,
      sleep_score_overall_pz: -0.031, steps_total_pz: -0.926, stress_score_pz: 0.321,
      glucose_mean_pz: 0.392, demographic_vo2_max_pz: -0.966, resting_hr_roll3: 62.059,
      nightly_temperature_roll3: 34.565,
    },
  },
  {
    id: "s2",
    true_phase: "Fertility",
    display: {
      nightly_temperature: 33.0,
      hrv_rmssd: 86,
      resting_hr: 68,
      sleep_score_overall: 91,
      respiratory_rate: 14.6,
      glucose_mean: 118,
      steps_total: 2501,
      cramps: 0,
    },
    full_features: {
      appetite: 2, exerciselevel: 3, headaches: 0, cramps: 0, sorebreasts: 0, fatigue: 2,
      sleepissue: 0, moodswing: 1, stress: 3, foodcravings: 0, indigestion: 0, bloating: 0,
      resting_hr: 68.415, nightly_temperature: 32.995, nightly_temperature_std: 0.45,
      hrv_rmssd: 86.42, hrv_high_frequency: 2065.92, hrv_low_frequency: 1754.408,
      respiratory_rate: 14.6, sleep_minutesasleep: 469, sleep_efficiency: 37,
      sleep_minutes_to_fall_asleep: 0, sleep_time_in_bed: 532, sleep_score_overall: 91,
      sleep_score_deep_minutes: 125, sleep_score_restlessness: 0.086, steps_total: 2501,
      active_minutes_sedentary: 723, active_minutes_lightly: 149, active_minutes_moderately: 0,
      active_minutes_very: 0, stress_score: 0, glucose_mean: 118.37, glucose_std: 11.496,
      glucose_cv: 0.097, demographic_vo2_max: 36.567, age_of_first_menarche: 10, bmi: 30.552,
      resting_hr_pz: -0.979, nightly_temperature_pz: 0.242, nightly_temperature_std_pz: -0.776,
      hrv_rmssd_pz: 0.018, hrv_high_frequency_pz: 0.117, hrv_low_frequency_pz: -0.757,
      respiratory_rate_pz: -1.03, sleep_minutesasleep_pz: 0.632, sleep_efficiency_pz: -1.372,
      sleep_score_overall_pz: 1.806, steps_total_pz: -1.057, stress_score_pz: -1.519,
      glucose_mean_pz: 0.245, demographic_vo2_max_pz: -0.932, resting_hr_roll3: 69.48,
      nightly_temperature_roll3: 33.034,
    },
  },
  {
    id: "s3",
    true_phase: "Luteal",
    display: {
      nightly_temperature: 33.23,
      hrv_rmssd: 112,
      resting_hr: 61,
      sleep_score_overall: 79,
      respiratory_rate: 18.8,
      glucose_mean: 128,
      steps_total: 5463,
      cramps: 1,
    },
    full_features: {
      appetite: 3, exerciselevel: 4, headaches: 4, cramps: 1, sorebreasts: 4, fatigue: 4,
      sleepissue: 3, moodswing: 0, stress: 0, foodcravings: 4, indigestion: 1, bloating: 2,
      resting_hr: 61.228, nightly_temperature: 33.228, nightly_temperature_std: 0.559,
      hrv_rmssd: 111.781, hrv_high_frequency: 2384.544, hrv_low_frequency: 2287.055,
      respiratory_rate: 18.8, sleep_minutesasleep: 315, sleep_efficiency: 92,
      sleep_minutes_to_fall_asleep: 0, sleep_time_in_bed: 357, sleep_score_overall: 79,
      sleep_score_deep_minutes: 83, sleep_score_restlessness: 0.101, steps_total: 5463,
      active_minutes_sedentary: 791, active_minutes_lightly: 165, active_minutes_moderately: 13,
      active_minutes_very: 23, stress_score: 72, glucose_mean: 128.305, glucose_std: 13.324,
      glucose_cv: 0.104, demographic_vo2_max: 42.901, age_of_first_menarche: 11, bmi: 21.967,
      resting_hr_pz: -2.048, nightly_temperature_pz: -0.619, nightly_temperature_std_pz: -0.843,
      hrv_rmssd_pz: 1.378, hrv_high_frequency_pz: 1.317, hrv_low_frequency_pz: 1.104,
      respiratory_rate_pz: 0.328, sleep_minutesasleep_pz: -0.719, sleep_efficiency_pz: -0.895,
      sleep_score_overall_pz: 0.506, steps_total_pz: -0.81, stress_score_pz: 0.132,
      glucose_mean_pz: 1.279, demographic_vo2_max_pz: -0.966, resting_hr_roll3: 62.149,
      nightly_temperature_roll3: 33.096,
    },
  },
];

const DEFAULT_FEATURES: Features = {
  nightly_temperature: 34.0,
  hrv_rmssd: 52,
  resting_hr: 61,
  sleep_score_overall: 78,
  respiratory_rate: 15.2,
  glucose_mean: 95,
  steps_total: 7400,
  cramps: 0,
};

// ============================================================
// UI
// ============================================================
const PHASE_COLORS: Record<PhaseLabel, string> = {
  Menstrual: "#CECBF6",
  Follicular: "#7F77DD",
  Fertility: "#AFA9EC",
  Luteal: "#534AB7",
};
const PHASES: PhaseLabel[] = ["Menstrual", "Follicular", "Fertility", "Luteal"];

type DataMode = "sample" | "manual" | "upload";
type View = "predict" | "fertile" | "ask";

function CycleLens() {
  const [dataMode, setDataMode] = useState<DataMode>("sample");
  const [sampleId, setSampleId] = useState<string>(SAMPLE_DAYS[0].id);
  const [features, setFeatures] = useState<Features>(SAMPLE_DAYS[0].display);
  const [view, setView] = useState<View>("predict");

  const [predicting, setPredicting] = useState(false);
  const [predictErr, setPredictErr] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);

  const [explain, setExplain] = useState<ExplainResponse | null>(null);
  const [explainErr, setExplainErr] = useState<string | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);

  const [uploadedFeatures, setUploadedFeatures] = useState<Record<string, number> | null>(null);
  const [uploadErr, setUploadErr] = useState<string | null>(null);

  const activeSample = useMemo(
    () => SAMPLE_DAYS.find((s) => s.id === sampleId) ?? null,
    [sampleId],
  );

  // When sample changes, sync displayed features.
  useEffect(() => {
    if (dataMode === "sample" && activeSample) {
      setFeatures(activeSample.display);
    }
  }, [dataMode, activeSample]);

  async function runPrediction(payload: Record<string, number>) {
    setPredicting(true);
    setPredictErr(null);
    setExplain(null);
    setExplainErr(null);
    try {
      const res = await apiPredictFeatures(payload);
      setPrediction(res);
      setExplainLoading(true);
      try {
        const ex = await apiExplain(res);
        setExplain(ex);
      } catch (e) {
        setExplainErr(e instanceof Error ? e.message : "Could not load insight.");
      } finally {
        setExplainLoading(false);
      }
    } catch (e) {
      setPredictErr(e instanceof Error ? e.message : "Something went wrong.");
      setPrediction(null);
    } finally {
      setPredicting(false);
    }
  }

  async function handlePredict() {
    const payload: Record<string, number> =
      dataMode === "sample" && activeSample
        ? ({ ...features, ...activeSample.full_features } as Record<string, number>)
        : dataMode === "upload" && uploadedFeatures
          ? uploadedFeatures
          : features;
    await runPrediction(payload);
  }

  async function handleUploadFile(file: File) {
    setUploadErr(null);
    setPredictErr(null);
    setPrediction(null);
    setExplain(null);
    setExplainErr(null);
    try {
      const text = await file.text();
      const parsed = JSON.parse(text) as Record<string, unknown>;
      const featuresPayload =
        parsed.features && typeof parsed.features === "object" && !Array.isArray(parsed.features)
          ? (parsed.features as Record<string, number>)
          : (parsed as Record<string, number>);
      setUploadedFeatures(featuresPayload);
      await runPrediction(featuresPayload);
    } catch (e) {
      const message =
        e instanceof SyntaxError
          ? "Invalid JSON file."
          : e instanceof Error
            ? e.message
            : "Upload failed.";
      setUploadErr(message);
      setUploadedFeatures(null);
    }
  }

  const activeFeatures: Record<string, number> =
    dataMode === "upload" && uploadedFeatures
      ? uploadedFeatures
      : dataMode === "sample" && activeSample
        ? ({ ...features, ...activeSample.full_features } as Record<string, number>)
        : features;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto max-w-4xl px-5 py-10 sm:py-14">
        <Header />

        <main className="mt-10 space-y-6">
          <DataSelector
            dataMode={dataMode}
            setDataMode={setDataMode}
            sampleId={sampleId}
            setSampleId={setSampleId}
            activeSample={activeSample}
            uploadErr={uploadErr}
            onUploadFile={handleUploadFile}
          />

          <ViewSelector view={view} setView={setView} />

          <InputPanel
            features={features}
            setFeatures={setFeatures}
            readOnly={dataMode === "sample" || dataMode === "upload"}
            hidden={dataMode === "upload"}
          />

          {dataMode !== "upload" && (
          <button
            type="button"
            onClick={handlePredict}
            disabled={predicting}
            className="w-full rounded-xl px-5 py-3.5 text-sm font-semibold text-white shadow-sm transition-all disabled:opacity-60 disabled:cursor-not-allowed hover:brightness-110 active:brightness-95"
            style={{ backgroundColor: "var(--brand-primary)" }}
          >
            {predicting
              ? "Analyzing…"
              : view === "ask"
                ? "Update readings"
                : view === "fertile"
                  ? "Estimate fertile window"
                  : "Predict phase"}
          </button>
          )}

          <ResultCard
            view={view}
            predicting={predicting}
            predictErr={predictErr}
            prediction={prediction}
            explain={explain}
            explainErr={explainErr}
            explainLoading={explainLoading}
            features={activeFeatures as Features}
            activeSample={dataMode === "sample" ? activeSample : null}
            dataMode={dataMode}
          />
        </main>

        <Footer />
      </div>
    </div>
  );
}

function Header() {
  return (
    <header className="text-center sm:text-left">
      <div className="inline-flex items-center gap-2">
        <div
          className="h-9 w-9 rounded-xl"
          style={{
            background:
              "conic-gradient(from 220deg, #CECBF6, #7F77DD, #534AB7, #AFA9EC, #CECBF6)",
          }}
          aria-hidden
        />
        <span className="text-sm font-semibold tracking-widest uppercase text-muted-foreground">
          CycleLens
        </span>
      </div>
      <h1
        className="mt-4 text-3xl sm:text-5xl leading-tight tracking-tight text-foreground"
        style={{ fontFamily: "var(--font-display)", fontWeight: 600 }}
      >
        Read your cycle phase from your wearable
        <span className="block text-muted-foreground italic">— no hormone test.</span>
      </h1>
    </header>
  );
}

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <section
      className={`rounded-xl border border-border bg-card p-5 sm:p-6 shadow-[0_1px_0_rgba(83,74,183,0.04),0_8px_24px_-16px_rgba(83,74,183,0.15)] ${className}`}
    >
      {children}
    </section>
  );
}

function SectionLabel({ n, children }: { n: number; children: React.ReactNode }) {
  return (
    <div className="mb-4 flex items-center gap-2">
      <span
        className="inline-flex h-6 w-6 items-center justify-center rounded-full text-[11px] font-semibold text-white"
        style={{ backgroundColor: "var(--brand-primary)" }}
      >
        {n}
      </span>
      <h2 className="text-sm font-semibold tracking-wide text-foreground">{children}</h2>
    </div>
  );
}

function DataSelector({
  dataMode,
  setDataMode,
  sampleId,
  setSampleId,
  activeSample,
  uploadErr,
  onUploadFile,
}: {
  dataMode: DataMode;
  setDataMode: (m: DataMode) => void;
  sampleId: string;
  setSampleId: (id: string) => void;
  activeSample: SampleDay | null;
  uploadErr: string | null;
  onUploadFile: (file: File) => void;
}) {
  return (
    <Card>
      <SectionLabel n={1}>Choose your data</SectionLabel>
      <div className="inline-flex rounded-lg bg-muted p-1 text-sm">
        {(["sample", "manual", "upload"] as DataMode[]).map((m) => {
          const active = m === dataMode;
          return (
            <button
              key={m}
              type="button"
              onClick={() => setDataMode(m)}
              className={`rounded-md px-3.5 py-1.5 font-medium transition-colors ${
                active
                  ? "bg-card text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {m === "sample"
                ? "Use a sample day"
                : m === "manual"
                  ? "Enter readings"
                  : "Upload a day"}
            </button>
          );
        })}
      </div>

      {dataMode === "sample" && (
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <select
            value={sampleId}
            onChange={(e) => setSampleId(e.target.value)}
            className="rounded-lg border border-border bg-card px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-[var(--brand-primary)]"
          >
            {SAMPLE_DAYS.map((s, i) => (
              <option key={s.id} value={s.id}>
                Day {i + 1} — {s.true_phase}
              </option>
            ))}
          </select>
          {activeSample && (
            <span
              className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium"
              style={{
                backgroundColor: "color-mix(in oklab, var(--brand-primary) 10%, white)",
                color: "var(--brand-primary)",
              }}
            >
              <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: PHASE_COLORS[activeSample.true_phase] }} />
              true phase: {activeSample.true_phase}
            </span>
          )}
        </div>
      )}
      {dataMode === "manual" && (
        <p className="mt-3 text-sm text-muted-foreground">
          Adjust the sliders below with your own wearable readings.
        </p>
      )}
      {dataMode === "upload" && (
        <div className="mt-4 space-y-2">
          <p className="text-sm text-muted-foreground">
            Upload a JSON file with a feature object or a <code className="text-xs">{"{ \"features\": { ... } }"}</code> wrapper.
          </p>
          <input
            type="file"
            accept=".json,application/json"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onUploadFile(file);
            }}
            className="block w-full text-sm text-foreground file:mr-3 file:rounded-lg file:border-0 file:bg-muted file:px-3 file:py-2 file:text-sm file:font-medium file:text-foreground hover:file:bg-muted/80"
          />
          {uploadErr && (
            <p className="text-sm text-destructive">{uploadErr}</p>
          )}
        </div>
      )}
    </Card>
  );
}

function ViewSelector({ view, setView }: { view: View; setView: (v: View) => void }) {
  const opts: { id: View; title: string; sub: string }[] = [
    { id: "predict", title: "Predict phase", sub: "Classify the 4 cycle phases" },
    { id: "fertile", title: "Fertile window", sub: "Highlight ovulation likelihood" },
    { id: "ask", title: "Ask assistant", sub: "Chat about the reading" },
  ];
  return (
    <Card>
      <SectionLabel n={2}>What do you want to know?</SectionLabel>
      <div className="grid gap-3 sm:grid-cols-3">
        {opts.map((o) => {
          const active = o.id === view;
          return (
            <button
              key={o.id}
              type="button"
              onClick={() => setView(o.id)}
              className={`rounded-xl border p-4 text-left transition-all ${
                active
                  ? "border-transparent shadow-sm"
                  : "border-border bg-card hover:border-[var(--brand-primary)]/40"
              }`}
              style={
                active
                  ? {
                      backgroundColor:
                        "color-mix(in oklab, var(--brand-primary) 8%, white)",
                      borderColor: "var(--brand-primary)",
                    }
                  : undefined
              }
            >
              <div className="text-sm font-semibold text-foreground">{o.title}</div>
              <div className="mt-1 text-xs text-muted-foreground">{o.sub}</div>
            </button>
          );
        })}
      </div>
    </Card>
  );
}

type SliderDef = {
  key: keyof Features;
  label: string;
  unit?: string;
  min: number;
  max: number;
  step: number;
  decimals: number;
};

const SLIDERS: SliderDef[] = [
  { key: "nightly_temperature", label: "Nightly temperature", unit: "°C", min: 33, max: 35, step: 0.05, decimals: 2 },
  { key: "hrv_rmssd", label: "HRV (rmssd)", unit: "ms", min: 20, max: 120, step: 1, decimals: 0 },
  { key: "resting_hr", label: "Resting heart rate", unit: "bpm", min: 45, max: 90, step: 1, decimals: 0 },
  { key: "sleep_score_overall", label: "Sleep score", min: 30, max: 100, step: 1, decimals: 0 },
  { key: "respiratory_rate", label: "Respiratory rate", unit: "/min", min: 10, max: 22, step: 0.1, decimals: 1 },
  { key: "glucose_mean", label: "Daily glucose mean", unit: "mg/dL", min: 70, max: 140, step: 1, decimals: 0 },
  { key: "steps_total", label: "Daily steps", min: 0, max: 20000, step: 100, decimals: 0 },
  { key: "cramps", label: "Cramps", min: 0, max: 3, step: 1, decimals: 0 },
];

function InputPanel({
  features,
  setFeatures,
  readOnly,
  hidden = false,
}: {
  features: Features;
  setFeatures: (f: Features) => void;
  readOnly: boolean;
  hidden?: boolean;
}) {
  if (hidden) return null;
  return (
    <Card>
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-wide text-foreground">Readings</h2>
        {readOnly && (
          <span className="text-xs text-muted-foreground">Sample values — switch to manual to edit</span>
        )}
      </div>
      <div className="grid gap-x-8 gap-y-5 sm:grid-cols-2">
        {SLIDERS.map((s) => (
          <SliderRow
            key={s.key}
            def={s}
            value={features[s.key]}
            onChange={(v) => setFeatures({ ...features, [s.key]: v })}
            disabled={readOnly}
          />
        ))}
      </div>
    </Card>
  );
}

function SliderRow({
  def,
  value,
  onChange,
  disabled,
}: {
  def: SliderDef;
  value: number;
  onChange: (v: number) => void;
  disabled: boolean;
}) {
  const pct = ((value - def.min) / (def.max - def.min)) * 100;
  const display =
    def.decimals === 0 ? Math.round(value).toLocaleString() : value.toFixed(def.decimals);
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <label className="text-sm font-medium text-foreground">{def.label}</label>
        <span className="text-sm tabular-nums text-foreground">
          {display}
          {def.unit && <span className="ml-1 text-muted-foreground">{def.unit}</span>}
        </span>
      </div>
      <input
        type="range"
        min={def.min}
        max={def.max}
        step={def.step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="mt-2 h-1.5 w-full cursor-pointer appearance-none rounded-full outline-none disabled:cursor-not-allowed disabled:opacity-70"
        style={{
          background: `linear-gradient(to right, var(--brand-primary) 0%, var(--brand-primary) ${pct}%, var(--muted) ${pct}%, var(--muted) 100%)`,
        }}
      />
    </div>
  );
}

function ResultCard({
  view,
  predicting,
  predictErr,
  prediction,
  explain,
  explainErr,
  explainLoading,
  features,
  activeSample,
  dataMode,
}: {
  view: View;
  predicting: boolean;
  predictErr: string | null;
  prediction: PredictResponse | null;
  explain: ExplainResponse | null;
  explainErr: string | null;
  explainLoading: boolean;
  features: Features;
  activeSample: SampleDay | null;
  dataMode: DataMode;
}) {
  if (view === "ask") {
    return <AssistantCard features={features} />;
  }

  return (
    <Card>
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-wide text-foreground">
          {view === "fertile" ? "Fertile window estimate" : "Prediction"}
        </h2>
        {activeSample && prediction && (
          <span className="text-xs text-muted-foreground">
            true phase:{" "}
            <span
              className={
                prediction.phase_label === activeSample.true_phase
                  ? "font-semibold text-[color:var(--brand-teal)]"
                  : "font-semibold text-foreground"
              }
            >
              {activeSample.true_phase}
            </span>
          </span>
        )}
      </div>

      {predicting && <SkeletonResult />}

      {!predicting && predictErr && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">
          Prediction failed — {predictErr}. Try again.
        </div>
      )}

      {!predicting && !predictErr && !prediction && (
        <EmptyResult view={view} dataMode={dataMode} />
      )}

      {!predicting && !predictErr && prediction && (
        <>
          {view === "predict" ? (
            <PredictView prediction={prediction} />
          ) : (
            <FertileView prediction={prediction} />
          )}

          <InsightBox
            loading={explainLoading}
            explain={explain}
            err={explainErr}
          />
        </>
      )}
    </Card>
  );
}

function EmptyResult({ view, dataMode }: { view: View; dataMode: DataMode }) {
  if (dataMode === "upload") {
    return (
      <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
        Upload a JSON file above to see phase prediction, drivers, and insight.
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
      Adjust readings and press{" "}
      <span className="font-medium text-foreground">
        {view === "fertile" ? "Estimate fertile window" : "Predict phase"}
      </span>{" "}
      to see results.
    </div>
  );
}

function SkeletonResult() {
  return (
    <div className="animate-pulse space-y-4">
      <div className="h-8 w-40 rounded-full bg-muted" />
      <div className="space-y-2">
        {PHASES.map((p) => (
          <div key={p} className="h-4 w-full rounded-full bg-muted" />
        ))}
      </div>
      <div className="h-16 w-full rounded-lg bg-muted" />
    </div>
  );
}

function PredictView({ prediction }: { prediction: PredictResponse }) {
  const topProb = prediction.probabilities[prediction.phase_label];
  return (
    <div>
      <div className="flex items-center gap-3">
        <span
          className="inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-sm font-semibold text-white"
          style={{ backgroundColor: PHASE_COLORS[prediction.phase_label] }}
        >
          {prediction.phase_label}
        </span>
        <span className="text-sm text-muted-foreground">
          {(topProb * 100).toFixed(0)}% likelihood
        </span>
      </div>

      <div className="mt-5 space-y-2.5">
        {PHASES.map((p) => (
          <ProbBar key={p} label={p} value={prediction.probabilities[p]} color={PHASE_COLORS[p]} />
        ))}
      </div>

      <DriverRow drivers={prediction.top_drivers} />
    </div>
  );
}

function FertileView({ prediction }: { prediction: PredictResponse }) {
  const p = prediction.probabilities.Fertility;
  return (
    <div>
      <div className="flex items-baseline gap-3">
        <span
          className="text-5xl font-semibold tabular-nums tracking-tight"
          style={{ fontFamily: "var(--font-display)", color: "var(--brand-primary)" }}
        >
          {(p * 100).toFixed(0)}%
        </span>
        <span className="text-sm text-muted-foreground">fertile-window likelihood</span>
      </div>
      <div className="mt-2 text-sm text-muted-foreground">
        Most likely phase overall:{" "}
        <span className="font-medium text-foreground">{prediction.phase_label}</span>
      </div>

      <div className="mt-5 space-y-2.5">
        {PHASES.map((ph) => (
          <ProbBar key={ph} label={ph} value={prediction.probabilities[ph]} color={PHASE_COLORS[ph]} />
        ))}
      </div>

      <DriverRow drivers={prediction.top_drivers} />
    </div>
  );
}

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="tabular-nums text-foreground">{(value * 100).toFixed(0)}%</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${value * 100}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

function DriverRow({ drivers }: { drivers: Driver[] }) {
  return (
    <div className="mt-6">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Top drivers
      </div>
      <div className="flex flex-wrap gap-2">
        {drivers.map((d, i) => (
          <span
            key={i}
            className="inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium"
            style={{
              backgroundColor: "color-mix(in oklab, var(--brand-teal) 10%, white)",
              color: "var(--brand-teal)",
            }}
          >
            <span aria-hidden>{d.direction === "up" ? "↑" : "↓"}</span>
            {d.feature}
          </span>
        ))}
      </div>
    </div>
  );
}

function InsightBox({
  loading,
  explain,
  err,
}: {
  loading: boolean;
  explain: ExplainResponse | null;
  err: string | null;
}) {
  return (
    <div
      className="mt-6 rounded-xl p-4"
      style={{
        backgroundColor: "color-mix(in oklab, var(--brand-primary) 6%, white)",
        border: "1px solid color-mix(in oklab, var(--brand-primary) 15%, white)",
      }}
    >
      <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Insight
      </div>
      {loading && <div className="mt-2 h-4 w-3/4 animate-pulse rounded bg-muted" />}
      {!loading && err && (
        <div className="mt-2 text-sm text-destructive">Could not load insight — {err}</div>
      )}
      {!loading && !err && explain && (
        <>
          <p className="mt-1.5 text-sm leading-relaxed text-foreground">{explain.sentence}</p>
          {explain.source_url && explain.source_title && (
            <a
              href={explain.source_url}
              target="_blank"
              rel="noreferrer noopener"
              className="mt-2 inline-block text-xs font-medium underline underline-offset-2"
              style={{ color: "var(--brand-primary)" }}
            >
              Source: {explain.source_title}
            </a>
          )}
        </>
      )}
    </div>
  );
}

type ChatMsg = { role: "user" | "assistant"; text: string; source?: string; error?: boolean };

function AssistantCard({ features }: { features: Features }) {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const q = input.trim();
    if (!q || pending) return;
    setMessages((m) => [...m, { role: "user", text: q }]);
    setInput("");
    setPending(true);
    try {
      const res = await apiAsk(q, features);
      setMessages((m) => [...m, { role: "assistant", text: res.answer, source: res.source_url }]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          error: true,
          text:
            err instanceof Error
              ? `Something went wrong — ${err.message}. Try again.`
              : "Something went wrong. Try again.",
        },
      ]);
    } finally {
      setPending(false);
    }
  }

  return (
    <Card>
      <h2 className="text-sm font-semibold tracking-wide text-foreground">Ask assistant</h2>
      <p className="mt-1 text-xs text-muted-foreground">
        Ask about the current readings. Answers cite public sources.
      </p>

      <div className="mt-4 space-y-3">
        {messages.length === 0 && (
          <div className="rounded-lg border border-dashed border-border p-4 text-sm text-muted-foreground">
            Try: <span className="italic">“Am I likely ovulating today?”</span>
          </div>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[85%] rounded-xl px-3.5 py-2.5 text-sm ${
                m.role === "user"
                  ? "text-white"
                  : m.error
                    ? "bg-destructive/5 text-destructive border border-destructive/20"
                    : "bg-muted text-foreground"
              }`}
              style={m.role === "user" ? { backgroundColor: "var(--brand-primary)" } : undefined}
            >
              <p className="leading-relaxed">{m.text}</p>
              {m.source && (
                <a
                  href={m.source}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="mt-1.5 inline-block text-xs underline underline-offset-2 opacity-80"
                >
                  Source
                </a>
              )}
            </div>
          </div>
        ))}
        {pending && (
          <div className="flex justify-start">
            <div className="rounded-xl bg-muted px-3.5 py-2.5 text-sm text-muted-foreground">
              Thinking…
            </div>
          </div>
        )}
      </div>

      <form onSubmit={submit} className="mt-4 flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your reading…"
          className="flex-1 rounded-lg border border-border bg-card px-3.5 py-2.5 text-sm outline-none focus:border-[var(--brand-primary)] focus:ring-2 focus:ring-[var(--brand-primary)]/20"
        />
        <button
          type="submit"
          disabled={pending || !input.trim()}
          className="rounded-lg px-4 py-2.5 text-sm font-semibold text-white disabled:opacity-60"
          style={{ backgroundColor: "var(--brand-primary)" }}
        >
          Ask
        </button>
      </form>
    </Card>
  );
}

function Footer() {
  return (
    <footer className="mt-12 border-t border-border pt-5 text-center text-xs leading-relaxed text-muted-foreground">
      Research prototype · not a medical device · trained on 42 participants (mcPHASES) · not
      validated for clinical use.
    </footer>
  );
}

// silence unused warning for DEFAULT_FEATURES (reference for API contract shape)
void DEFAULT_FEATURES;
