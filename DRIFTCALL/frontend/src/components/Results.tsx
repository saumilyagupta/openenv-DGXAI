import { useId } from "react";

import { RESULTS, REWARD_CURVE } from "../data/content";

import "./Results.css";

function CurveSvg({ data }: { data: readonly number[] }): JSX.Element {
  const id = useId();
  const W = 800;
  const H = 260;
  const pad = 24;
  const xs = data.map((_, i) => pad + (i / (data.length - 1)) * (W - pad * 2));
  const ys = data.map(
    (v) => H - pad - v * (H - pad * 2),
  );
  const path = xs.map((x, i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${ys[i].toFixed(1)}`).join(" ");
  const fill =
    `${path} L${(W - pad).toFixed(1)} ${(H - pad).toFixed(1)} L${pad.toFixed(1)} ${(H - pad).toFixed(1)} Z`;
  const stages = [
    { label: "stage 1 — no drift", at: 0.20 },
    { label: "stage 2 — single drift", at: 0.50 },
    { label: "stage 3 — compound", at: 0.85 },
  ];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="results__curve" preserveAspectRatio="none">
      <defs>
        <linearGradient id={`grad-${id}`} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="var(--saffron)" stopOpacity="0.32" />
          <stop offset="100%" stopColor="var(--saffron)" stopOpacity="0" />
        </linearGradient>
      </defs>
      {/* baseline grid */}
      {[0.2, 0.4, 0.6, 0.8].map((y) => (
        <line
          key={y}
          x1={pad}
          x2={W - pad}
          y1={H - pad - y * (H - pad * 2)}
          y2={H - pad - y * (H - pad * 2)}
          stroke="var(--ink-edge)"
          strokeWidth="1"
        />
      ))}
      {/* stage markers */}
      {stages.map((s) => {
        const x = pad + s.at * (W - pad * 2);
        return (
          <g key={s.label}>
            <line
              x1={x}
              x2={x}
              y1={pad}
              y2={H - pad}
              stroke="var(--ink-edge-soft)"
              strokeDasharray="2 4"
            />
            <text
              x={x + 4}
              y={pad + 11}
              fill="var(--ash-deep)"
              fontFamily="var(--font-mono)"
              fontSize="9"
              letterSpacing="0.1em"
              textTransform="uppercase"
            >
              {s.label}
            </text>
          </g>
        );
      })}
      <path d={fill} fill={`url(#grad-${id})`} />
      <path d={path} fill="none" stroke="var(--saffron)" strokeWidth="1.6" />
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
              <span className="kicker">reward · 3-stage curriculum</span>
              <span className="mono results__chart-y">
                R<sub>1..5</sub>
              </span>
            </header>
            <CurveSvg data={REWARD_CURVE} />
            <footer className="results__chart-foot mono">
              <span>step 0</span>
              <span>step 290</span>
            </footer>
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
