import { Architecture } from "./components/Architecture";
import { BeforeAfter } from "./components/BeforeAfter";
import { Demo } from "./components/Demo";
import { Footer } from "./components/Footer";
import { Future } from "./components/Future";
import { Hero } from "./components/Hero";
import { LiveRL } from "./components/LiveRL";
import { Premise } from "./components/Premise";
import { Results } from "./components/Results";
import { Resources } from "./components/Resources";
import { RewardGrid } from "./components/RewardGrid";

import "./App.css";

const NAV = [
  { id: "premise", label: "premise" },
  { id: "rewards", label: "reward" },
  { id: "demo", label: "demo" },
  { id: "liverl", label: "live RL" },
  { id: "results", label: "results" },
  { id: "architecture", label: "arch" },
  { id: "shift", label: "before / after" },
  { id: "future", label: "future" },
  { id: "resources", label: "links" },
] as const;

export function App(): JSX.Element {
  return (
    <>
      {/* Sticky vertical rail (desktop) — index marks for the spread. */}
      <nav className="rail" aria-label="section index">
        <span className="rail__brand mono">drift / call</span>
        <ol className="rail__list">
          {NAV.map((n, i) => (
            <li key={n.id}>
              <a href={`#${n.id}`} className="rail__link">
                <span className="rail__num mono">{String(i + 1).padStart(2, "0")}</span>
                <span className="rail__label">{n.label}</span>
              </a>
            </li>
          ))}
        </ol>
        <span className="rail__foot mono">v0.1.0</span>
      </nav>

      <main className="main">
        <Hero />
        <Premise />
        <RewardGrid />
        <Demo />
        <LiveRL />
        <Results />
        <Architecture />
        <BeforeAfter />
        <Future />
        <Resources />
      </main>

      <Footer />
    </>
  );
}
