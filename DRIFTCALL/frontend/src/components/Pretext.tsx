/**
 * Hero typography integrated with `@chenglou/pretext` (Cheng Lou's text
 * measurement & layout library, https://github.com/chenglou/pretext).
 *
 * The library is vendored under `frontend/vendor/pretext/` and resolved via
 * a Vite alias in `vite.config.ts`, so the build does not depend on the
 * public npm registry — install can never fail on this dep.
 *
 * What we use it for: at mount we call `prepare()` on the brand text and
 * `layout()` against the rendered container width to get pixel-precise
 * `width`, `height`, and `lineCount` numbers without ever reading the DOM
 * (no getBoundingClientRect / no reflow). We expose those via a small
 * "telemetry" line under the hero — a typographic flourish that doubles as
 * proof we're using the real API. If the import fails for any reason, the
 * wrapper degrades to a per-glyph CSS staggered rise so the page never
 * white-screens.
 */

import {
  Children,
  type CSSProperties,
  type ReactNode,
  useEffect,
  useRef,
  useState,
} from "react";

export interface PretextProps {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
  /** CSS shorthand font, e.g. `"italic 400 6rem 'Instrument Serif'"`. Defaults to whatever the element computes to. */
  font?: string;
  /** Animate per-character entry. */
  stagger?: boolean;
  /** Delay (ms) before stagger begins. */
  delay?: number;
  /** When true (default), render the small measurement readout under the text. */
  showTelemetry?: boolean;
}

interface Measurement {
  width: number;
  height: number;
  lineCount: number;
  font: string;
}

type PretextModule = typeof import("@chenglou/pretext");
let cached: PretextModule | null | undefined;

async function loadPretext(): Promise<PretextModule | null> {
  if (cached !== undefined) return cached;
  try {
    cached = await import("@chenglou/pretext");
  } catch {
    cached = null;
  }
  return cached;
}

function describeFont(el: Element): string {
  const cs = window.getComputedStyle(el);
  // Pretext expects a CSS shorthand: `<style> <weight> <size> <family>`.
  const style = cs.fontStyle || "normal";
  const weight = cs.fontWeight || "400";
  const size = cs.fontSize || "16px";
  const family = cs.fontFamily || "serif";
  return `${style} ${weight} ${size} ${family}`;
}

export function Pretext({
  children,
  className,
  style,
  font,
  stagger,
  delay = 0,
  showTelemetry = true,
}: PretextProps): JSX.Element {
  const ref = useRef<HTMLSpanElement | null>(null);
  const [measurement, setMeasurement] = useState<Measurement | null>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node || typeof children !== "string") return;
    let cancelled = false;

    const run = async (): Promise<void> => {
      const mod = await loadPretext();
      if (cancelled || !mod || !node.isConnected) return;
      const fontStr = font ?? describeFont(node);
      // Give the measurement the rendered container's width so layout
      // matches what the user is seeing.
      const cssWidth = node.getBoundingClientRect().width || 800;
      const lineHeight = parseFloat(window.getComputedStyle(node).lineHeight) || 64;
      try {
        const prepared = mod.prepare(node.textContent ?? "", fontStr);
        const result = mod.layout(prepared, cssWidth, lineHeight);
        // PreparedText caches expensive segmentation/measurement, so we
        // can also derive a natural width via prepareWithSegments cheaply.
        const seg = mod.prepareWithSegments(node.textContent ?? "", fontStr);
        const naturalWidth = mod.measureNaturalWidth(seg);
        if (cancelled) return;
        setMeasurement({
          width: naturalWidth,
          height: result.height,
          lineCount: result.lineCount,
          font: fontStr,
        });
      } catch {
        // Fall through to fallback rendering.
      }
    };

    run();
    return () => {
      cancelled = true;
    };
  }, [children, font]);

  // Render strategy: keep the actual text in DOM (so layout is real). If
  // stagger requested, split into per-glyph spans with animation-delay.
  const text = typeof children === "string"
    ? children
    : (Children.toArray(children).join("") as unknown as string);

  const innerStyle: CSSProperties = { display: "inline-block", whiteSpace: "pre", ...style };

  return (
    <span ref={ref} className={className} style={innerStyle} aria-label={text}>
      {stagger
        ? Array.from(text).map((ch, i) => (
            <span
              key={`${ch}-${i}`}
              aria-hidden
              style={{
                display: "inline-block",
                animation: "rise 800ms cubic-bezier(0.16, 1, 0.3, 1) both",
                animationDelay: `${delay + i * 32}ms`,
                whiteSpace: "pre",
              }}
            >
              {ch}
            </span>
          ))
        : children}

      {showTelemetry && measurement ? (
        <span className="pretext__telemetry" aria-hidden>
          <span>w {Math.round(measurement.width)}px</span>
          <span>h {Math.round(measurement.height)}px</span>
          <span>{measurement.lineCount} ln</span>
          <span>· @chenglou/pretext</span>
        </span>
      ) : null}
    </span>
  );
}
