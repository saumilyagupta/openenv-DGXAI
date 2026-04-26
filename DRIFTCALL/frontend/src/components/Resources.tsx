import { META } from "../data/content";

import "./Resources.css";

const TILES = [
  {
    label: "unified space",
    title: "saumilyajj/driftcall",
    desc: "site + openenv api + demo + manifest, all under one origin · /reset /step /demo /lora /source",
    href: `https://huggingface.co/spaces/${META.envSpace}`,
    suffix: "huggingface.co/spaces",
    accent: true,
  },
  {
    label: "trained adapter",
    title: META.loraRepo,
    desc: "lora · 84.6 MB · adapter-only · 240 GRPO steps · 3-stage curriculum.",
    href: `https://huggingface.co/${META.loraRepo}`,
    suffix: "huggingface.co",
    accent: false,
  },
  {
    label: "openenv api docs",
    title: "live swagger UI",
    desc: "fastapi auto-docs · POST /reset /step · GET /state · POST /close.",
    href: `${META.unifiedSpaceUrl}/docs`,
    suffix: "saumilyajj-driftcall.hf.space",
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
          <span className="eyebrow">§09 — resources</span>
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
