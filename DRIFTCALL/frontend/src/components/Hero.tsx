import { useEffect, useRef } from "react";

import { META } from "../data/content";
import { Pretext } from "./Pretext";

import "./Hero.css";

export function Hero(): JSX.Element {
  const waveRef = useRef<SVGPathElement | null>(null);

  // Animate the bottom waveform — pure SVG path morph driven by sin.
  useEffect(() => {
    const path = waveRef.current;
    if (!path) return;
    let t = 0;
    let raf = 0;
    const W = 1600;
    const H = 100;
    const points = 240;

    const tick = (): void => {
      t += 0.012;
      const segments: string[] = [];
      for (let i = 0; i <= points; i++) {
        const x = (i / points) * W;
        const y =
          H / 2 +
          Math.sin(i * 0.04 + t) * 14 +
          Math.sin(i * 0.013 - t * 1.4) * 22 +
          Math.sin(i * 0.27 + t * 0.7) * 4;
        segments.push(`${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`);
      }
      path.setAttribute("d", segments.join(" "));
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <header className="hero">
      {/* Decorative Devanagari watermark behind the title. */}
      <span className="hero__devanagari" aria-hidden>
        {META.devanagari}
      </span>

      <div className="shell hero__shell">
        <div className="hero__top">
          <span className="kicker">DGX × OpenEnv · Hackathon 2026</span>
          <span className="mono hero__coord">28.6° N, 77.2° E</span>
        </div>

        <h1 className="hero__title">
          <Pretext stagger className="hero__brand">
            DriftCall
          </Pretext>
          <span className="hero__slash">/</span>
          <em className="hero__sub">
            voice concierge under
            <br />
            <span className="hero__sub-em">schema drift.</span>
          </em>
        </h1>

        <div className="hero__meta">
          <p className="hero__lede">
            An OpenEnv-compliant RL environment where a voice-first agent must
            book the cab, hold the room, settle the payment — in Hindi, Tamil,
            Kannada, Hinglish — while the vendor APIs <em>mutate mid-episode</em>.
            Five reward components. No LLM judges. Deterministic.
          </p>

          <ul className="hero__chips" aria-label="quick facts">
            <li>
              <span className="mono hero__chip-key">model</span>
              <span className="mono hero__chip-val">gemma-3n-E2B + LoRA</span>
            </li>
            <li>
              <span className="mono hero__chip-key">trainer</span>
              <span className="mono hero__chip-val">native GRPO · g=2</span>
            </li>
            <li>
              <span className="mono hero__chip-key">curriculum</span>
              <span className="mono hero__chip-val">3 stages · drift→compound</span>
            </li>
            <li>
              <span className="mono hero__chip-key">eval</span>
              <span className="mono hero__chip-val">held-out · 200-ep probe</span>
            </li>
          </ul>

          <div className="hero__cta">
            <a className="hero__btn hero__btn--primary" href="#demo">
              <span>live demo</span>
              <span aria-hidden>→</span>
            </a>
            <a
              className="hero__btn hero__btn--ghost"
              href={`https://huggingface.co/spaces/${META.envSpace}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              <span>openenv gym</span>
              <span aria-hidden>↗</span>
            </a>
            <a className="hero__btn hero__btn--ghost" href={META.github} target="_blank" rel="noopener noreferrer">
              <span>source</span>
              <span aria-hidden>↗</span>
            </a>
          </div>
        </div>
      </div>

      {/* Bottom voice waveform — running animation. */}
      <svg
        className="hero__wave"
        viewBox="0 0 1600 100"
        preserveAspectRatio="none"
        aria-hidden
      >
        <path ref={waveRef} d="" />
      </svg>
    </header>
  );
}
