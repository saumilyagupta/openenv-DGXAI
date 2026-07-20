import { LANGUAGES } from "../data/content";

import "./Premise.css";

export function Premise(): JSX.Element {
  return (
    <section className="section premise" id="premise">
      <div className="shell premise__shell">
        <header className="premise__header">
          <span className="eyebrow">§01 — premise</span>
          <h2 className="premise__title">
            Production APIs
            <br />
            don&apos;t hold still.
          </h2>
        </header>

        <div className="premise__columns">
          <p className="premise__lede">
            <span className="premise__drop">A</span>n agent that books a flight on
            Tuesday confidently fires the same JSON payload on Thursday and gets
            <em> 422 </em>back. The endpoint moved. <em>price</em> is now
            <em> total</em>. <em>seat_class</em> is split into <em>cabin </em>
            and <em>fare_brand</em>. The cancel window shrank from 24h to 6h.
            Auth tokens rotate every 90 minutes now.
          </p>

          <p className="premise__body">
            Every benchmark in the open assumes static schemas, English-only
            briefs, and a friendly oracle in the loop. Real concierge work is
            the opposite. Tasks arrive in Hindi mixed with Tamil mixed with
            English numerals. Vendors deprecate fields without changelog. The
            agent has to <strong>notice the drift</strong>, retry against the
            new shape, and keep its promise to the user — &ldquo;under ₹800,
            before 9 pm, vegetarian, no haldi&rdquo; — through the whole thing.
          </p>

          <p className="premise__body">
            DriftCall is an environment built around that gap. It speaks five
            ways. It mutates schemas mid-episode. It scores reward
            deterministically — no LLM judges anywhere in the pipeline — across
            five independent components. And it ships as an OpenEnv-compliant
            Space, so any agent that talks the protocol can train against it.
          </p>
        </div>

        {/* Language strip — five Indic scripts as decorative ledger. */}
        <ul className="premise__langs" aria-label="languages exercised">
          {LANGUAGES.map((l, idx) => (
            <li key={l.code}>
              <span className="mono premise__lang-num">
                {String(idx + 1).padStart(2, "0")}
              </span>
              <span className="premise__lang-script">{l.script}</span>
              <span className="mono premise__lang-name">{l.name}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
