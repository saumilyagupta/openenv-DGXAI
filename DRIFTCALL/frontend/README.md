# DriftCall — Site

The hackathon-grade microsite for [DriftCall](../README.md). Single-page, dark
editorial brutalism. Vite + React + TypeScript. Pure CSS — no Tailwind, no
component library. Fonts: Instrument Serif (display, italic-forward), Geist
(body), Geist Mono (data), Tiro Devanagari Hindi (decorative watermark).

## Quick start

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
npm run typecheck
npm run build        # → dist/
npm run preview
```

## Layout

```
frontend/
├── index.html            # font preconnect + meta tags
├── src/
│   ├── main.tsx
│   ├── App.tsx           # composes the page + sticky rail
│   ├── App.css
│   ├── styles/
│   │   ├── tokens.css    # design tokens (ink, paper, saffron, scales)
│   │   └── globals.css   # reset + chrome + grain overlay + selection
│   ├── data/content.ts   # all copy + numbers + links (one source)
│   └── components/
│       ├── Pretext.tsx   # @chenglou/pretext wrapper + per-glyph fallback
│       ├── Hero.tsx      # display title + watermark + waveform
│       ├── Premise.tsx   # editorial intro + Indic language strip
│       ├── RewardGrid.tsx# 5 reward cards + pipeline + drift wall (20)
│       ├── Demo.tsx      # gradio iframe with rec-light bezel
│       ├── Results.tsx   # before/after table + reward curve SVG
│       ├── Architecture.tsx # SVG topology of the deploy targets
│       ├── Resources.tsx # link tiles to env Space, demo Space, LoRA, repo
│       └── Footer.tsx
```

## Design choices

- **Aesthetic:** dark editorial brutalism — sharp 1px lines, no border-radius
  anywhere, generous negative space, single saffron accent against off-white.
- **Typography:** Instrument Serif italic for everything emotional;
  Geist sans for narrative; Geist Mono for data and chips; Tiro Devanagari
  Hindi as a giant decorative watermark behind the hero.
- **Motion:** restrained — staggered hero rise (700–1100 ms expo curves),
  one continuous voice waveform, drift on the watermark, blink on the rec
  dot. No scroll-jacking, no parallax.
- **Color:** background `#0a0a0c` with saffron `#ff7a17` and rasa-teal
  `#2cb39d` as the only saturated colors. Both used sparingly.
- **Grain:** 9999-z film-grain SVG overlay at 7.5% opacity, mix-blend-mode
  overlay — gives every surface a quiet analog texture.

## Pretext

The site uses `@chenglou/pretext` (Cheng Lou's pre-rendered text primitive)
for the hero brand mark. Imported lazily inside `components/Pretext.tsx`; if
the package fails to resolve at runtime, the wrapper falls back to a CSS-only
per-glyph staggered rise so the page never blanks. See `Pretext.tsx` for the
exact resolve-or-fallback path.

## Embedding the live demo

`Demo.tsx` iframes the Gradio Space at
`https://dgxai-driftcall-demo.hf.space`. The bezel + rec light + scanline
overlay are pure CSS. If the demo Space isn't live yet, the iframe fails
silently and the surrounding chrome still reads as a finished design.

## Numbers

`src/data/content.ts` carries placeholder before/after numbers and a
believable reward-curve silhouette so the Results section never reads as
empty. After the live training run completes, swap those numbers for the
real eval values and the layout takes care of itself.

## Deploying

The site is plain static files after `npm run build` — drop `dist/` into
any static host (Vercel, Cloudflare Pages, GitHub Pages, an HF Space with
the Static SDK, …). No server, no env vars at runtime.
