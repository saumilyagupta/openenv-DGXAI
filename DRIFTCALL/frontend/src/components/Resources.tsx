import { META } from "../data/content";

import "./Resources.css";

const TILES = [
  {
    label: "openenv env space",
    title: "DGXAI/driftcall-env",
    desc: "FastAPI · /reset /step /state /close · bearer auth · OpenEnv v1.0 manifest.",
    href: `https://huggingface.co/spaces/${META.envSpace}`,
    suffix: "huggingface.co/spaces",
    accent: true,
  },
  {
    label: "demo space",
    title: "DGXAI/driftcall-demo",
    desc: "voice UI · gradio · base ↔ trained toggle · 20 drift patterns selectable.",
    href: `https://huggingface.co/spaces/${META.demoSpace}`,
    suffix: "huggingface.co/spaces",
    accent: false,
  },
  {
    label: "trained adapter",
    title: META.loraRepo,
    desc: "lora · 84.6 MB · adapter-only · pulled at runtime by both Spaces.",
    href: `https://huggingface.co/${META.loraRepo}`,
    suffix: "huggingface.co",
    accent: false,
  },
  {
    label: "source",
    title: "saumilyagupta/openenv-DGXAI · branch google/gemma-3n-E4B-it",
    desc: "monorepo · cells/ · data/ · scripts/train_driftcall_grpo.py · deploy/",
    href: META.github,
    suffix: "github.com",
    accent: false,
  },
] as const;

export function Resources(): JSX.Element {
  return (
    <section className="section resources" id="resources">
      <div className="shell resources__shell">
        <header className="resources__header">
          <span className="eyebrow">§06 — resources</span>
          <h2 className="resources__title">
            Where everything lives.
          </h2>
        </header>

        <ul className="resources__grid">
          {TILES.map((t) => (
            <li key={t.title}>
              <a
                className={`resources__tile${t.accent ? " resources__tile--accent" : ""}`}
                href={t.href}
                target="_blank"
                rel="noopener noreferrer"
              >
                <span className="resources__suffix mono">{t.suffix} ↗</span>
                <span className="resources__label">{t.label}</span>
                <span className="resources__title-text">{t.title}</span>
                <span className="resources__desc">{t.desc}</span>
              </a>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
