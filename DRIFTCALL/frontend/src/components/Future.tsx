import "./Future.css";

interface Direction {
  id: string;
  num: string;
  kicker: string;
  title: string;
  pull: string;
  body: string[];
  signals: { label: string; detail: string }[];
  variant: "dark" | "light";
  anchor: "left" | "right";
}

const DIRECTIONS: Direction[] = [
  {
    id: "emergency",
    num: "01",
    kicker: "public safety",
    title: "Emergency assistance, in any language",
    pull: "If someone shouts “Bachao” in Hindi or “Help me” in English, the same primitive that routes a cab booking should route an ambulance.",
    body: [
      "Distress detection at two boundaries — sight and sound. Camera spots a closed-fist gesture or a hand sign held against a window; mic hears panicked shouting in any of the five Indic languages we already train on. The same drift-aware action loop then reaches into a different vendor surface: emergency services rather than payments.",
      "Why DriftCall is the right substrate for this: emergency endpoints drift constantly. Police WhatsApp numbers move between districts. Ambulance dispatch APIs change shape state-by-state. The agent already trains against schema mutation, so the same model handles the policy churn that has historically killed every “one-tap SOS” project.",
    ],
    signals: [
      { label: "vision", detail: "hand gestures · written cues (Bachao, Help me)" },
      { label: "audio", detail: "shout detection · panic keyword spotting" },
      { label: "action", detail: "112 dispatch · GPS share · live caller bridge" },
      { label: "fallback", detail: "SMS to emergency contacts when bandwidth dies" },
    ],
    variant: "dark",
    anchor: "left",
  },
  {
    id: "education",
    num: "02",
    kicker: "multilingual teaching",
    title: "A teacher who switches language at the right moment",
    pull: "A topic in Tamil for the student who thinks in Tamil. A worked example in Hindi for the kid sitting next to her. The same concept, the same accuracy, no translation lag.",
    body: [
      "The schema-drift training we did for concierge work is, structurally, the same problem teachers solve all day: the same idea expressed under shifting representation. Move the pedagogy: instead of vendor APIs as the surface, the textbook + curriculum + student model become the surface, and the agent’s job is to keep the explanation invariant while the language and example layer change.",
      "What this looks like in practice: a student says “I don’t get it” in Hinglish; the model re-explains in their preferred mix; the teacher sees a transcript and a confidence score; the explanation grounds against the curriculum, not the model’s priors. Five-language coverage already exists in the env briefs — the rewards just need re-keying for pedagogical correctness.",
    ],
    signals: [
      { label: "input", detail: "voice · text · live transcription" },
      { label: "stack", detail: "the cells/ env, retargeted to a curriculum schema" },
      { label: "reward", detail: "concept retention · constraint adherence · idiom fit" },
      { label: "scope", detail: "K-12 first · vocational + adult upskilling next" },
    ],
    variant: "light",
    anchor: "right",
  },
  {
    id: "platform",
    num: "03",
    kicker: "the platform thesis",
    title: "The plumbing layer for an Indic voice revolution",
    pull: "NVIDIA built the hardware that the AI revolution runs on. India’s multilingual voice revolution will run on a layer too — deterministic rewards, drift-aware agents, vernacular ground truth.",
    body: [
      "Every vertical that wants to reach the next 800M Indians will need the same primitives: speech recognition that does not collapse on code-switching, action grounding that survives schema mutation, evaluation that does not silently leak the answer to an LLM judge. DriftCall ships those primitives as an OpenEnv-compliant gym. Other teams can train their domain-specific agents against it.",
      "The pitch is not “we will build every product on top.” The pitch is: build the substrate so well that every health-tech, ed-tech, fin-tech, and gov-tech team building voice agents in India and the diaspora reaches for it before they reach for English-only baselines. The trained adapter on HF Hub is a starting weight; the env on this same Space is the ground.",
    ],
    signals: [
      { label: "surface", detail: "OpenEnv v1.0 · /reset /step /state /close" },
      { label: "weights", detail: "DGXAI/gemma-3n-e2b-driftcall-lora (apache-2.0)" },
      { label: "moat", detail: "20-pattern drift catalogue · 5-language briefs · deterministic rewards" },
      { label: "scale-out", detail: "Indic → SEA → LATAM — one vendor surface at a time" },
    ],
    variant: "dark",
    anchor: "left",
  },
];

export function Future(): JSX.Element {
  return (
    <section className="section future" id="future">
      {/* Stencil wordmark — the section's spinal column. */}
      <div className="future__wordmark" aria-hidden>
        <span>FUT</span>
        <span>URE</span>
      </div>

      {/* Saffron diagonal that threads all three cards. */}
      <div className="future__diagonal" aria-hidden />

      <div className="shell future__shell">
        <header className="future__header">
          <span className="eyebrow">§07 — future work</span>
          <h2 className="future__title">
            What the same primitive
            <br />
            <em>can become</em> next.
          </h2>
          <p className="future__sub">
            DriftCall is, mechanically, a deterministic agent that holds an
            invariant intent through a mutating environment. Concierge
            booking is one instance of that. Emergency response is another.
            Multilingual teaching is a third. The substrate generalises.
          </p>
        </header>

        <ol className="future__stack">
          {DIRECTIONS.map((d, i) => (
            <li
              className="future__slab"
              data-variant={d.variant}
              data-anchor={d.anchor}
              data-index={i}
              key={d.id}
              id={`future-${d.id}`}
            >
              {/* Stacked-print shadow plate behind the slab. */}
              <span className="future__plate" aria-hidden />

              <div className="future__slab-inner">
                <aside className="future__numeral" aria-hidden>
                  <span className="future__numeral-digit">{d.num}</span>
                  <span className="future__numeral-rule" />
                </aside>

                <div className="future__col">
                  <span className="future__kicker mono">
                    <span className="future__kicker-tick" aria-hidden />
                    {d.kicker}
                  </span>
                  <h3 className="future__slab-title">{d.title}</h3>
                  <p className="future__pull">{d.pull}</p>
                  <div className="future__paras">
                    {d.body.map((p, j) => (
                      <p className="future__para" key={j}>
                        {p}
                      </p>
                    ))}
                  </div>
                </div>

                <ul className="future__signals" aria-label="signals & surface">
                  {d.signals.map((s, k) => (
                    <li key={s.label} style={{ ["--sig-i" as string]: k }}>
                      <span className="mono future__sig-key">{s.label}</span>
                      <span className="future__sig-val">{s.detail}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </li>
          ))}
        </ol>

        <footer className="future__footer">
          <span className="mono future__footer-rule" aria-hidden />
          <p className="future__footer-line">
            <em>three directions, one substrate.</em>
            <span className="mono"> the env runs already · the rest is a vendor surface away.</span>
          </p>
        </footer>
      </div>
    </section>
  );
}
