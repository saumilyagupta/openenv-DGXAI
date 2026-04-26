import { REWARDS, DRIFT_PATTERNS } from "../data/content";

import "./RewardGrid.css";

export function RewardGrid(): JSX.Element {
  return (
    <section className="section reward" id="rewards">
      <div className="shell reward__shell">
        <header className="reward__header">
          <span className="eyebrow">§02 — reward function</span>
          <h2 className="reward__title">
            Five components.
            <br />
            <em>Zero LLM judges.</em>
          </h2>
          <p className="reward__sub">
            Every bit of reward traces to a deterministic check against the
            episode trace and the (possibly drifted) JSON schema. Calibrated
            with a Brier penalty against the agent&apos;s own confidence; an
            <em> uncertain floor </em> at 0.50 prevents pathological
            high-confidence wrong answers from gaming the score. Source:{" "}
            <code className="mono">cells/step_08_rewards.py</code>.
          </p>
        </header>

        <ol className="reward__grid">
          {REWARDS.map((r, idx) => (
            <li className="reward__card" key={r.id} style={{ animationDelay: `${idx * 80}ms` }}>
              <div className="reward__card-head">
                <span className="reward__id">{r.id}</span>
                <span className="reward__weight mono">w = {r.weight.toFixed(2)}</span>
              </div>
              <h3 className="reward__name">
                <span className="mono">{r.name}</span>
              </h3>
              <p className="reward__blurb">{r.blurb}</p>
              <code className="reward__impl mono">{r.impl}</code>
            </li>
          ))}
        </ol>

        {/* Pipeline strip — shows the calibration chain. */}
        <div className="reward__pipeline" aria-label="reward pipeline">
          <span className="mono reward__pipe-step">combine_quality</span>
          <span className="reward__pipe-arrow" aria-hidden>→</span>
          <span className="mono reward__pipe-step">brier_penalty</span>
          <span className="reward__pipe-arrow" aria-hidden>→</span>
          <span className="mono reward__pipe-step">apply_uncertain_floor</span>
          <span className="reward__pipe-arrow" aria-hidden>→</span>
          <span className="mono reward__pipe-step reward__pipe-step--final">
            final_reward
          </span>
        </div>

        {/* Wall of drift — 20 patterns enumerated. */}
        <div className="reward__drift">
          <header className="reward__drift-head">
            <span className="kicker">drift catalogue</span>
            <span className="mono reward__drift-count">{DRIFT_PATTERNS.length} / 20</span>
          </header>
          <ul className="reward__drift-list">
            {DRIFT_PATTERNS.map((p, i) => (
              <li key={p}>
                <span className="mono reward__drift-num">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className="reward__drift-name">{p.replace(/_/g, " ")}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
