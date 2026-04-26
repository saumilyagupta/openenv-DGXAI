import { fileURLToPath, URL } from "node:url";

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `@chenglou/pretext` is vendored under ./vendor/pretext (cloned from
// https://github.com/chenglou/pretext) so the build never depends on the
// public npm registry. The alias resolves the bare specifier to the
// vendored TypeScript source; Vite handles the TS + ESM interop.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@chenglou/pretext": fileURLToPath(
        new URL("./vendor/pretext/src/layout.ts", import.meta.url),
      ),
    },
  },
  optimizeDeps: {
    exclude: ["@chenglou/pretext"],
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    target: "es2022",
  },
});
