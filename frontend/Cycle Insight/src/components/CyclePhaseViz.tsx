/**
 * Lightweight CSS 3D cycle ring — wearable signal orbiting the four phases.
 * No WebGL/Three.js; keeps bundle small for demo landing pages.
 */

const PHASES = [
  { label: "Menstrual", color: "#534AB7" },
  { label: "Follicular", color: "#7F77DD" },
  { label: "Fertility", color: "#AFA9EC" },
  { label: "Luteal", color: "#CECBF6" },
] as const;

const RING_GRADIENT =
  "conic-gradient(from 210deg, #534AB7 0% 25%, #7F77DD 25% 50%, #AFA9EC 50% 75%, #CECBF6 75% 100%)";

export function CyclePhaseViz() {
  return (
    <div
      className="relative mx-auto flex w-full max-w-[280px] flex-col items-center"
      aria-label="Illustration: wearable signals mapping to cycle phases"
      role="img"
    >
      <div
        className="relative mx-auto h-[190px] w-[190px]"
        style={{ perspective: "820px" }}
      >
        <div
          className="absolute inset-0 cycle-viz-ring"
          style={{
            transformStyle: "preserve-3d",
            background: RING_GRADIENT,
            borderRadius: "9999px",
            boxShadow:
              "0 20px 44px -14px rgba(83, 74, 183, 0.5), inset 0 0 0 11px rgba(255,255,255,0.88)",
          }}
        />
        <div className="absolute inset-0 cycle-viz-orbit" style={{ transformStyle: "preserve-3d" }}>
          <div className="cycle-viz-signal">
            <span className="cycle-viz-signal-glow" />
            <span className="cycle-viz-signal-dot" />
          </div>
        </div>
        <div className="absolute left-1/2 top-1/2 z-10 flex h-14 w-14 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-2xl border border-white/80 bg-white/90 shadow-lg backdrop-blur-sm">
          <svg viewBox="0 0 24 24" className="h-7 w-7 text-[var(--brand-primary)]" fill="none" aria-hidden>
            <rect x="5" y="3" width="14" height="18" rx="4" stroke="currentColor" strokeWidth="1.5" />
            <path d="M9 7h6M9 11h4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </div>
      </div>

      <div className="mt-2 flex flex-wrap justify-center gap-x-3 gap-y-1">
        {PHASES.map((p) => (
          <span key={p.label} className="inline-flex items-center gap-1 text-[10px] text-muted-foreground">
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: p.color }} />
            {p.label}
          </span>
        ))}
      </div>
      <p className="mt-2 text-center text-xs text-muted-foreground">
        Signals from your wearable orbit the four cycle phases
      </p>

      <style>{`
        .cycle-viz-ring {
          animation: cycle-viz-ring-spin 22s linear infinite;
        }
        .cycle-viz-orbit {
          animation: cycle-viz-ring-spin 22s linear infinite;
        }
        .cycle-viz-signal {
          position: absolute;
          left: 50%;
          top: 50%;
          width: 12px;
          height: 12px;
          margin-left: -6px;
          margin-top: -6px;
          animation: cycle-viz-signal-orbit 5.5s linear infinite;
        }
        .cycle-viz-signal-glow {
          position: absolute;
          inset: -4px;
          border-radius: 9999px;
          background: var(--brand-teal);
          opacity: 0.45;
          filter: blur(2px);
        }
        .cycle-viz-signal-dot {
          position: relative;
          display: block;
          width: 12px;
          height: 12px;
          border-radius: 9999px;
          border: 2px solid white;
          background: var(--brand-teal);
          box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        @keyframes cycle-viz-ring-spin {
          from { transform: rotateX(62deg) rotateZ(0deg); }
          to { transform: rotateX(62deg) rotateZ(360deg); }
        }
        @keyframes cycle-viz-signal-orbit {
          from { transform: rotateZ(0deg) translateX(78px); }
          to { transform: rotateZ(360deg) translateX(78px); }
        }
      `}</style>
    </div>
  );
}
