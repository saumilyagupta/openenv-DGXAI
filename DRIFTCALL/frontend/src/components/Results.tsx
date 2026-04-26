import { useId } from "react";

import { METRICS, RESULTS } from "../data/content";

import "./Results.css";

interface CurveProps {
  data: readonly number[];
  stages: { name: string; len: number }[];
  yMin?: number;
  yMax?: number;
}

function CurveSvg({ data, stages, yMin = 0, yMax = 1 }: CurveProps): JSX.Element {
  const id = useId();
  const W = 800;
  const H = 260;
  const pad = 28;
  const yRange = yMax - yMin;

  const xs = data.map((_, i) => pad + (i / Math.max(data.length - 1, 1)) * (W - pad * 2));
  const ys = data.map((v) => {
    const norm = (v - yMin) / yRange;
    return H - pad - Math.max(0, Math.min(1, norm)) * (H - pad * 2);
  });
  const path = xs
    .map((x, i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${ys[i].toFixed(1)}`)
    .join(" ");
  const fill = `${path} L${(W - pad).toFixed(1)} ${(H - pad).toFixed(1)} L${pad.toFixed(1)} ${(H - pad).toFixed(1)} Z`;

  // Stage boundaries based on real point counts.
  const total = data.length;
  let acc = 0;
  const stageEdges = stages.map((s) => {
    acc += s.len;
    return { name: s.name, edge: acc };
  });

  // Y-axis ticks at 0, 0.25, 0.5, 0.75, 1
  const yTicks = [0, 0.25, 0.5, 0.75, 1].filter((y) => y >= yMin && y <= yMax);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="results__curve" preserveAspectRatio="none">
      <defs>
        <linearGradient id={`grad-${id}`} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="var(--saffron)" stopOpacity="0.4" />
          <stop offset="100%" stopColor="var(--saffron)" stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* horizontal grid + y-tick labels */}
      {yTicks.map((y) => {
        const py = H - pad - ((y - yMin) / yRange) * (H - pad * 2);
        return (
          <g key={y}>
            <line
              x1={pad}
              x2={W - pad}
              y1={py}
              y2={py}
              stroke="var(--ink-edge)"
              strokeWidth="1"
            />
            <text
              x={pad - 6}
              y={py + 3}
              fill="var(--ash-deep)"
              fontFamily="var(--font-mono)"
              fontSize="9"
              textAnchor="end"
            >
              {y.toFixed(2)}
            </text>
          </g>
        );
      })}

      {/* stage boundaries — vertical dashed line + label */}
      {stageEdges.slice(0, -1).map((s) => {
        const x = pad + (s.edge / total) * (W - pad * 2);
        return (
          <g key={s.name}>
            <line
              x1={x}
              x2={x}
              y1={pad}
              y2={H - pad}
              stroke="var(--saffron)"
              strokeOpacity="0.35"
              strokeDasharray="2 4"
            />
          </g>
        );
      })}

      {/* stage labels along the top */}
      {stages.map((s, i) => {
        const startEdge = i === 0 ? 0 : stageEdges[i - 1].edge;
        const x = pad + ((startEdge + s.len / 2) / total) * (W - pad * 2);
        return (
          <text
            key={s.name + i}
            x={x}
            y={pad - 8}
            fill="var(--ash)"
            fontFamily="var(--font-mono)"
            fontSize="9"
            letterSpacing="0.18em"
            textAnchor="middle"
          >
            {`STAGE ${i + 1} · ${s.name.toUpperCase()}`}
          </text>
        );
      })}

      <path d={fill} fill={`url(#grad-${id})`} />
      <path d={path} fill="none" stroke="var(--saffron)" strokeWidth="1.6" />
    </svg>
  );
}

interface MultilineProps {
  series: { label: string; color: string; data: readonly number[] }[];
  yMax: number;
}

function MultilineSvg({ series, yMax }: MultilineProps): JSX.Element {
  const W = 800;
  const H = 200;
  const pad = 28;
  const len = Math.max(...series.map((s) => s.data.length));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="results__curve" preserveAspectRatio="none">
      {[0.25, 0.5, 0.75, 1].map((y) => {
        const py = H - pad - (y / 1) * (H - pad * 2);
        return (
          <line
            key={y}
            x1={pad}
            x2={W - pad}
            y1={py}
            y2={py}
            stroke="var(--ink-edge)"
            strokeWidth="1"
          />
        );
      })}
      {series.map((s) => {
        const path = s.data
          .map((v, i) => {
            const x = pad + (i / Math.max(s.data.length - 1, 1)) * (W - pad * 2);
            const y = H - pad - Math.min(v / yMax, 1) * (H - pad * 2);
            return `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
          })
          .join(" ");
        return (
          <path
            key={s.label}
            d={path}
            fill="none"
            stroke={s.color}
            strokeWidth="1.4"
            opacity="0.9"
          />
        );
      })}
      <text
        x={pad}
        y={pad - 10}
        fill="var(--ash)"
        fontFamily="var(--font-mono)"
        fontSize="9"
        letterSpacing="0.15em"
      >
        {`240 STEPS · ${series.length} SERIES · max(y)=${yMax.toFixed(2)}`}
      </text>
      {series.map((s, i) => (
        <g key={s.label} transform={`translate(${W - pad - 90}, ${pad + 4 + i * 14})`}>
          <line x1="0" x2="14" y1="0" y2="0" stroke={s.color} strokeWidth="1.6" />
          <text
            x="20"
            y="3"
            fill="var(--paper-soft)"
            fontFamily="var(--font-mono)"
            fontSize="10"
            letterSpacing="-0.01em"
          >
            {s.label}
          </text>
        </g>
      ))}
      {void len}
    </svg>
  );
}

function deltaPct(b: number, t: number): string {
  const d = ((t - b) / Math.max(b, 1e-6)) * 100;
  const sign = d >= 0 ? "+" : "";
  return `${sign}${d.toFixed(0)}%`;
}

export function Results(): JSX.Element {
  const rows = [
    {
      label: "mean reward",
      base: RESULTS.baseline.mean_reward,
      trained: RESULTS.trained.mean_reward,
      better: "higher",
    },
    {
      label: "drift detection rate",
      base: RESULTS.baseline.drift_detection_rate,
      trained: RESULTS.trained.drift_detection_rate,
      better: "higher",
    },
    {
      label: "constraint adherence",
      base: RESULTS.baseline.constraint_adherence,
      trained: RESULTS.trained.constraint_adherence,
      better: "higher",
    },
    {
      label: "avg turns to complete",
      base: RESULTS.baseline.avg_turns_to_complete,
      trained: RESULTS.trained.avg_turns_to_complete,
      better: "lower",
    },
  ] as const;

  return (
    <section className="section results" id="results">
      <div className="shell results__shell">
        <header className="results__header">
          <span className="eyebrow">§04 — results</span>
          <h2 className="results__title">
            Before / after,
            <br />
            <em>same 50 seeds.</em>
          </h2>
          <p className="results__sub">
            Held-out evaluation set. Same prompts, same seeds, same drift
            schedule. Only the LoRA changes. Numbers are placeholder shapes
            until the live training run finishes —{" "}
            <a className="inline" href="#resources">
              wandb dashboard
            </a>{" "}
            has the live curves.
          </p>
        </header>

        <div className="results__grid">
          <div className="results__chart">
            <header className="results__chart-head">
              <span className="kicker">reward_mean · 240 steps · live data</span>
              <span className="mono results__chart-y">y ∈ [0, 0.4]</span>
            </header>
            <CurveSvg
              data={METRICS.reward_mean}
              stages={METRICS.stages}
              yMin={0}
              yMax={0.4}
            />
            <footer className="results__chart-foot mono">
              <span>step 0</span>
              <span>
                {METRICS.reward_mean.length > 0
                  ? `step ${METRICS.reward_mean.length - 1}`
                  : "step –"}
              </span>
            </footer>
          </div>

          <div className="results__chart">
            <header className="results__chart-head">
              <span className="kicker">5 reward components · per step</span>
              <span className="mono results__chart-y">R₁..R₅</span>
            </header>
            <MultilineSvg
              yMax={0.4}
              series={[
                { label: "R1 task", color: "var(--saffron)", data: METRICS.r1 },
                { label: "R2 drift", color: "var(--rasa-teal)", data: METRICS.r2 },
                { label: "R3 cnstr", color: "var(--saffron-soft)", data: METRICS.r3 },
                { label: "R4 fmt", color: "var(--paper-soft)", data: METRICS.r4 },
                { label: "R5 hack", color: "var(--ash)", data: METRICS.r5 },
              ]}
            />
          </div>

          <table className="results__table">
            <thead>
              <tr>
                <th scope="col">metric</th>
                <th scope="col" className="results__th-base">
                  base
                </th>
                <th scope="col" className="results__th-trained">
                  trained
                </th>
                <th scope="col">Δ</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const isBetter =
                  (r.better === "higher" && r.trained > r.base) ||
                  (r.better === "lower" && r.trained < r.base);
                return (
                  <tr key={r.label}>
                    <th scope="row" className="results__cell-label">
                      {r.label}
                    </th>
                    <td className="results__cell-base mono">{r.base.toFixed(2)}</td>
                    <td className="results__cell-trained mono">{r.trained.toFixed(2)}</td>
                    <td className={`results__cell-delta mono ${isBetter ? "is-better" : "is-worse"}`}>
                      {deltaPct(r.base, r.trained)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
