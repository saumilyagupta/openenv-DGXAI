/**
 * Thin wrapper around `@chenglou/pretext` (https://github.com/chenglou/pretext).
 *
 * Pretext is a pre-rendering text primitive — it lays out the string before
 * paint so character-level animations (stagger, kern-tween) can hook in
 * without layout thrash. We use it for hero typography only.
 *
 * If the upstream package fails to resolve at runtime, we fall back to a
 * plain <span> with CSS-driven per-glyph rise, so the page never blanks.
 */

import {
  Children,
  type CSSProperties,
  type ComponentType,
  type ReactNode,
  useEffect,
  useState,
} from "react";

export interface PretextProps {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
  /** Animate per-character entry. */
  stagger?: boolean;
  /** Delay (ms) before stagger begins. */
  delay?: number;
}

type UpstreamComp = ComponentType<{
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
}>;

// Singleton cache so we only resolve the dynamic import once across mounts.
let cached: UpstreamComp | null | undefined;

async function resolveUpstream(): Promise<UpstreamComp | null> {
  if (cached !== undefined) return cached;
  try {
    const mod = (await import("@chenglou/pretext")) as Record<string, unknown>;
    const candidate = (mod.Pretext ?? mod.default) as UpstreamComp | undefined;
    cached = candidate ?? null;
  } catch {
    cached = null;
  }
  return cached;
}

export function Pretext({ children, className, style, stagger, delay = 0 }: PretextProps): ReactNode {
  const [Up, setUp] = useState<UpstreamComp | null>(cached ?? null);

  useEffect(() => {
    let live = true;
    resolveUpstream().then((c) => {
      if (live && c) setUp(() => c);
    });
    return () => {
      live = false;
    };
  }, []);

  if (Up) {
    return (
      <Up className={className} style={style}>
        {children}
      </Up>
    );
  }

  // Fallback: per-glyph staggered rise via CSS animation-delay.
  if (!stagger || typeof children !== "string") {
    return (
      <span className={className} style={style}>
        {children}
      </span>
    );
  }
  const text = Children.toArray(children).join("");
  return (
    <span className={className} style={style} aria-label={text}>
      {Array.from(text).map((ch, i) => (
        <span
          key={`${ch}-${i}`}
          aria-hidden
          style={{
            display: "inline-block",
            animation: "rise 700ms cubic-bezier(0.16, 1, 0.3, 1) both",
            animationDelay: `${delay + i * 28}ms`,
            whiteSpace: "pre",
          }}
        >
          {ch}
        </span>
      ))}
    </span>
  );
}
