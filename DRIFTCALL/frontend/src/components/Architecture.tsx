import { VENDORS } from "../data/content";

import "./Architecture.css";

/**
 * Architecture diagram — drawn as an annotated SVG instead of a generic
 * flowchart. Boxes are typographic plates; lines are precise; the whole thing
 * reads like a fold-out from a paper, not a Figma export.
 */
export function Architecture(): JSX.Element {
  return (
    <section className="section arch" id="architecture">
      <div className="shell arch__shell">
        <header className="arch__header">
          <span className="eyebrow">§06 — architecture</span>
          <h2 className="arch__title">
            <em>Three</em> deployable artefacts.
            <br />
            One canonical source.
          </h2>
          <p className="arch__sub">
            The repo at the root is the source of truth. Each deploy target —
            env Space, demo Space, inference client — is regenerated from it on
            every push via <code className="mono">deploy/build_all.sh</code>.
            The trained LoRA stays on HF Hub; the Spaces stay small.
          </p>
        </header>

        {/* SVG schematic */}
        <div className="arch__diagram" aria-label="DriftCall deployment topology">
          <svg viewBox="0 0 1200 640" preserveAspectRatio="xMidYMid meet">
            <defs>
              <marker
                id="arrow"
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="7"
                markerHeight="7"
                orient="auto"
              >
                <path d="M0,0 L10,5 L0,10 z" fill="var(--saffron)" />
              </marker>
              <pattern
                id="dots"
                x="0"
                y="0"
                width="14"
                height="14"
                patternUnits="userSpaceOnUse"
              >
                <circle cx="1" cy="1" r="0.6" fill="var(--ink-edge)" />
              </pattern>
            </defs>

            <rect x="0" y="0" width="1200" height="640" fill="url(#dots)" opacity="0.5" />

            {/* Top: trained LoRA on Hub */}
            <g className="arch__node arch__node--accent">
              <rect x="430" y="50" width="340" height="86" />
              <text x="450" y="80" className="arch__node-kicker">
                01 · MODEL ARTEFACT
              </text>
              <text x="450" y="110" className="arch__node-title">
                DGXAI/gemma-3n-e2b-driftcall-lora
              </text>
              <text x="450" y="128" className="arch__node-sub">
                pushed by cells.step_24_deploy_hf · adapter only · 84.6 MB
              </text>
            </g>

            {/* Center: env_space */}
            <g className="arch__node">
              <rect x="80" y="240" width="380" height="160" />
              <text x="100" y="270" className="arch__node-kicker">
                02 · OPENENV SPACE
              </text>
              <text x="100" y="306" className="arch__node-title">
                DGXAI/driftcall-env
              </text>
              <text x="100" y="330" className="arch__node-line">
                /reset · /step · /state · /close · /healthz
              </text>
              <text x="100" y="352" className="arch__node-line">
                cells/step_10_env · DriftCallEnv
              </text>
              <text x="100" y="374" className="arch__node-line">
                cells/step_08_rewards · 5 components
              </text>
              <text x="100" y="386" className="arch__node-foot">
                docker · cpu basic · &lt; 2 GB
              </text>
            </g>

            {/* Center: demo_space */}
            <g className="arch__node">
              <rect x="740" y="240" width="380" height="160" />
              <text x="760" y="270" className="arch__node-kicker">
                03 · DEMO SPACE
              </text>
              <text x="760" y="306" className="arch__node-title">
                DGXAI/driftcall-demo
              </text>
              <text x="760" y="330" className="arch__node-line">
                gradio · mic → asr → env → lora → tts → speaker
              </text>
              <text x="760" y="352" className="arch__node-line">
                kokoro tts · faster-whisper asr
              </text>
              <text x="760" y="374" className="arch__node-line">
                base ↔ trained toggle
              </text>
              <text x="760" y="386" className="arch__node-foot">
                gradio sdk · zerogpu / a10g
              </text>
            </g>

            {/* Bottom: inference */}
            <g className="arch__node arch__node--ghost">
              <rect x="430" y="500" width="340" height="100" />
              <text x="450" y="530" className="arch__node-kicker">
                04 · OPENENV GYM CLIENT
              </text>
              <text x="450" y="558" className="arch__node-title">
                deploy/inference/run.py
              </text>
              <text x="450" y="582" className="arch__node-line">
                DriftCallGymClient + GemmaPolicy
              </text>
            </g>

            {/* Edges */}
            <path
              className="arch__edge"
              d="M520 136 C 520 200, 270 220, 270 240"
              fill="none"
              markerEnd="url(#arrow)"
            />
            <path
              className="arch__edge"
              d="M680 136 C 680 200, 930 220, 930 240"
              fill="none"
              markerEnd="url(#arrow)"
            />
            <path
              className="arch__edge"
              d="M270 400 C 270 460, 520 470, 600 500"
              fill="none"
              markerEnd="url(#arrow)"
            />
            <path
              className="arch__edge arch__edge--soft"
              d="M740 320 C 600 320, 460 320, 460 320"
              fill="none"
              strokeDasharray="4 6"
            />

            <text x="600" y="318" className="arch__edge-label">
              shared cells/ + data/
            </text>
            <text x="375" y="180" className="arch__edge-label">
              lora pulled at runtime
            </text>
            <text x="828" y="180" className="arch__edge-label">
              lora pulled at runtime
            </text>
          </svg>
        </div>

        {/* Vendor strip */}
        <div className="arch__vendors">
          <span className="kicker">vendor surface · 5 mock APIs</span>
          <ul>
            {VENDORS.map((v) => (
              <li key={v.name}>
                <span className="mono arch__vendor-glyph">{v.glyph}</span>
                <span className="arch__vendor-name">{v.name}</span>
                <span className="arch__vendor-role">{v.role}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
