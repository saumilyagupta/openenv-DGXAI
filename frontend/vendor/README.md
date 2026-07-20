# vendor/

Third-party source vendored into the DriftCall frontend so the build never
depends on the public npm registry. Each subfolder is a clean clone of an
upstream project; nothing here is patched.

## pretext/

Cloned from <https://github.com/chenglou/pretext> at v0.0.6 (MIT).

`@chenglou/pretext` is a pure JS/TS text measurement and layout library —
it does the work of `getBoundingClientRect()` without ever touching the
DOM, by re-implementing line-breaking + bidi against the canvas font
engine. We use it in `src/components/Pretext.tsx` for the hero brand
telemetry readout (real `width`, `height`, `lineCount` values displayed
under the title).

The Vite alias in `../vite.config.ts` resolves the bare specifier
`@chenglou/pretext` to `./pretext/src/layout.ts` — Vite handles the
TypeScript + ESM interop, so we don't need to build the upstream
`dist/` ourselves and we don't need to install it from npm.

To refresh:

```bash
cd vendor
rm -rf pretext
git clone --depth=1 https://github.com/chenglou/pretext.git
rm -rf pretext/.git
```

The clone keeps the upstream `package.json`, README, RESEARCH.md, and the
full `src/` tree. We only import from `src/layout.ts`; the `scripts/`,
`benchmarks/`, `corpora/` etc. are unused at build time but kept here as
reference for anyone studying how the measurement primitives work.
