import { useEffect, useRef, useState } from "react";

import { META } from "../data/content";

import "./LiveRL.css";

interface TrainingStatus {
  running: boolean;
  boot_complete: boolean;
  gpu_available: boolean;
  step: number;
  max_steps: number;
  reward_mean: number;
  reward_std: number;
  loss: number;
  kl: number;
  grad_norm: number;
  ep_seconds: number;
  beta: number;
  lr: number;
  progress: number;
  wall_seconds: number;
  last_step_at: number | null;
  started_at: number | null;
  error: string | null;
  boot_lines: string[];
  stage: number;
  num_generations: number;
  hardware: string;
}

const POLL_MS = 3000;
const TRAINING_URL = `${META.unifiedSpaceUrl}/training`;
const START_URL = `${META.unifiedSpaceUrl}/training/start`;

export function LiveRL(): JSX.Element {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [pulseStep, setPulseStep] = useState<number>(-1);
  const [error, setError] = useState<string | null>(null);
  const lastStepRef = useRef<number>(-1);
  const rewardHistoryRef = useRef<number[]>([]);
  const [, force] = useState(0);

  useEffect(() => {
    let alive = true;
    const tick = async (): Promise<void> => {
      try {
        const r = await fetch(TRAINING_URL, { cache: "no-store" });
        if (!alive) return;
        if (!r.ok) {
          setError(`HTTP ${r.status}`);
          return;
        }
        const data = (await r.json()) as TrainingStatus;
        setStatus(data);
        setError(null);
        if (data.step > lastStepRef.current) {
          lastStepRef.current = data.step;
          setPulseStep(data.step);
          rewardHistoryRef.current.push(data.reward_mean);
          if (rewardHistoryRef.current.length > 64) {
            rewardHistoryRef.current = rewardHistoryRef.current.slice(-64);
          }
          force((n) => n + 1);
          window.setTimeout(() => {
            if (alive) setPulseStep(-1);
          }, 1100);
        }
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : "fetch failed");
      }
    };
    tick();
    const id = window.setInterval(tick, POLL_MS);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  const startTrainer = async (): Promise<void> => {
    try {
      await fetch(START_URL, { method: "POST" });
    } catch {
      // Status fetch will surface the error on next tick.
    }
  };

  const isRunning = !!status?.running;
  const stepLabel = status ? `${status.step.toLocaleString()} / ${status.max_steps.toLocaleString()}` : "—";
  const rewardLabel = status?.boot_complete ? `${status.reward_mean.toFixed(3)} ± ${status.reward_std.toFixed(3)}` : "warming up…";
  const klLabel = status?.boot_complete ? status.kl.toFixed(3) : "—";
  const lossLabel = status?.boot_complete ? status.loss.toFixed(3) : "—";
  const gradLabel = status?.boot_complete ? status.grad_norm.toFixed(2) : "—";
  const progressPct = status ? Math.min(1, Math.max(0, status.progress)) : 0;
  const stateBadge = !status
    ? "connecting…"
    : !status.gpu_available
      ? "gpu offline · cpu basic"
      : status.error
        ? "error"
        : status.running && !status.boot_complete
          ? "booting · loading model"
          : status.running
            ? "training · live"
            : "idle";

  return (
    <section className="section liverl" id="liverl">
      <div className="shell liverl__shell">
        <header className="liverl__header">
          <span className="eyebrow">§07 — live RL</span>
          <h2 className="liverl__title">
            <em>It's training</em> right now.
          </h2>
          <p className="liverl__sub">
            A real GRPO loop runs as a subprocess on this Space, fed by the
            same env you're hitting at <code className="mono">/reset</code> and{" "}
            <code className="mono">/step</code>. The numbers below come straight
            from <code className="mono">scripts/train_driftcall_grpo.py</code>
            's stdout, parsed and exposed at{" "}
            <a className="inline" href={`${META.unifiedSpaceUrl}/training`} target="_blank" rel="noopener noreferrer">
              /training
            </a>
            .
          </p>
        </header>

        <div className="liverl__grid">
          <div className={`liverl__card liverl__card--state liverl__card--${isRunning ? "live" : "idle"}`}>
            <span className="liverl__dot" aria-hidden />
            <span className="liverl__state-label">{stateBadge}</span>
            {status?.error ? (
              <span className="liverl__state-err">{status.error}</span>
            ) : (
              <span className="liverl__state-meta mono">
                hardware: {status?.hardware ?? "?"} · stage {status?.stage ?? "?"} · G={status?.num_generations ?? "?"}
              </span>
            )}
            {!isRunning && status && !status.error ? (
              <button className="liverl__btn" onClick={startTrainer}>
                start trainer
              </button>
            ) : null}
          </div>

          <Stat label="step" value={stepLabel} accent={pulseStep >= 0} />
          <Stat label="reward (mean ± std)" value={rewardLabel} accent={pulseStep >= 0} />
          <Stat label="kl" value={klLabel} />
          <Stat label="loss" value={lossLabel} />
          <Stat label="grad norm" value={gradLabel} />
          <Stat
            label="ep seconds"
            value={status?.boot_complete ? `${status.ep_seconds.toFixed(1)}s` : "—"}
          />
        </div>

        <div className="liverl__progress">
          <div className="liverl__progress-bar" style={{ width: `${progressPct * 100}%` }} />
          <span className="mono liverl__progress-label">
            {progressPct === 0 ? "0%" : `${(progressPct * 100).toFixed(1)}%`} of{" "}
            {status?.max_steps ?? "?"} steps
          </span>
        </div>

        <Sparkline values={rewardHistoryRef.current} pulseAt={pulseStep} />

        {status && status.boot_lines.length > 0 && !status.boot_complete ? (
          <pre className="liverl__bootlog mono">
            {status.boot_lines.slice(-12).join("\n")}
          </pre>
        ) : null}

        {error ? (
          <p className="liverl__net-err mono">/training fetch error: {error}</p>
        ) : null}
      </div>
    </section>
  );
}

function Stat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}): JSX.Element {
  return (
    <div className={`liverl__card liverl__stat ${accent ? "is-pulse" : ""}`}>
      <span className="liverl__stat-key">{label}</span>
      <span className="mono liverl__stat-val">{value}</span>
    </div>
  );
}

function Sparkline({
  values,
  pulseAt,
}: {
  values: readonly number[];
  pulseAt: number;
}): JSX.Element {
  if (values.length === 0) {
    return (
      <div className="liverl__spark liverl__spark--empty mono">
        reward sparkline · waiting for first step…
      </div>
    );
  }
  const W = 800;
  const H = 80;
  const pad = 4;
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 0.4);
  const range = Math.max(0.0001, max - min);
  const path = values
    .map((v, i) => {
      const x = pad + (i / Math.max(values.length - 1, 1)) * (W - pad * 2);
      const y = H - pad - ((v - min) / range) * (H - pad * 2);
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg className="liverl__spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <path d={path} fill="none" stroke="var(--saffron)" strokeWidth="1.4" />
      {pulseAt >= 0 ? (
        <circle
          cx={W - pad}
          cy={H - pad - ((values[values.length - 1] - min) / range) * (H - pad * 2)}
          r="3"
          fill="var(--saffron)"
        >
          <animate attributeName="r" from="3" to="14" dur="0.9s" begin="0s" />
          <animate attributeName="opacity" from="1" to="0" dur="0.9s" begin="0s" />
        </circle>
      ) : null}
    </svg>
  );
}
