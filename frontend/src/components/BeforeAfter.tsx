import "./BeforeAfter.css";

interface Panel {
  scene: string;
  body: string;
  meta: string;
}

interface Pair {
  id: string;
  num: string;
  context: string;
  before: Panel;
  after: Panel;
}

const PAIRS: Pair[] = [
  {
    id: "classroom",
    num: "01",
    context: "education / live notes",
    before: {
      scene: "“ma’am bola ‘was’, but now she’s writing ‘is being’… wait, kya likhna hai?”",
      body: "A 14-year-old in Pune misses three lines of a derivation because the teacher code-switched Hindi↔English mid-sentence and the auto-transcription tool collapsed both halves to garbage English.",
      meta: "language: hinglish · device: notes app · result: silently dropped tokens",
    },
    after: {
      scene: "transcript locks both halves: “verb hai ‘was’, tense imperfect → derive: dy/dx = …”",
      body: "Open the same lecture after class — both halves of every sentence are intact, the math is grounded against the textbook, the teacher’s code-switch is annotated, not erased.",
      meta: "stack: cells/ env retargeted to curriculum · reward: concept retention + idiom fit",
    },
  },
  {
    id: "cab",
    num: "02",
    context: "concierge / mobility",
    before: {
      scene: "ENTER PICK-UP · SET DROP · APPLY PROMO — “kuch samajh nahi aaya beta.”",
      body: "A 62-year-old in Delhi tries to book a cab to her son’s flat. Yesterday’s app update buried the language toggle in a settings sub-menu she can’t find. She cancels and calls a family member.",
      meta: "language: hindi · friction: schema mutation in mobile UI · outcome: drop-off",
    },
    after: {
      scene: "“beta, Vasant Kunj le chalo” — agent fills the slots, confirms back in a Hindi voice.",
      body: "One sentence. The drift-aware action loop transcribes, resolves the destination, calls cab.book against whichever vendor surface survived the last app refactor.",
      meta: "primitive: drift-aware action loop · surface: cab.book v3 · confirmation: TTS hindi",
    },
  },
  {
    id: "emergency",
    num: "03",
    context: "public safety / 112",
    before: {
      scene: "“mujhe lagi hai… arre koi help…” — 112 IVR replies in formal English.",
      body: "A migrant worker in Bengaluru dislocates a shoulder on-site. He dials 112; the IVR offers Kannada or English. He hangs up and limps to the nearest auto-stand alone.",
      meta: "language: bhojpuri-hindi · vendor: 112 IVR · drift: state-by-state schema · outcome: bypass",
    },
    after: {
      scene: "“mujhe lagi hai” → camera sees the slung arm · GPS shared · dispatch routed in his dialect.",
      body: "Voice primitive routes intent regardless of language; the same drift-aware loop reaches the right state’s emergency endpoint; an SMS reaches his contact when bandwidth dies mid-call.",
      meta: "input: voice + vision · fallback: SMS to emergency contacts · routing: state-aware",
    },
  },
  {
    id: "shopkeeper",
    num: "04",
    context: "fintech / point-of-sale",
    before: {
      scene: "“QR ಸ್ಕ್ಯಾನ್ ಬಟನ್ ಎಲ್ಲಿಗೆ ಹೋಯಿತು?” — yesterday it was right there.",
      body: "A Bengaluru shopkeeper opens his payments app at the till. A new release moved the QR scan button three taps deep into a “Tools” drawer. The customer queue grows; helpline supports English + Hindi only.",
      meta: "language: kannada · drift: in-app navigation schema · outcome: queue lost",
    },
    after: {
      scene: "“QR scan beku” — agent surfaces the scanner, reads the amount, confirms back in Kannada.",
      body: "He speaks one phrase. The agent recognises Kannada, walks the new schema for him, returns him to where he was without ever leaving the till.",
      meta: "language: kannada (asr + tts) · primitive: schema recovery · result: stayed on-task",
    },
  },
  {
    id: "teacher",
    num: "05",
    context: "education / pedagogy",
    before: {
      scene: "“இது quadratic equation, பாரு” — board says Tamil, textbook says English.",
      body: "A teacher in Madurai works through a quadratic on the board in Tamil; the textbook beside her is in English; students flip between the two and lose the thread on every sign change.",
      meta: "language: tamil + english · friction: pedagogy / textbook gap · outcome: comprehension drag",
    },
    after: {
      scene: "agent re-renders the worked example in each student’s preferred mix, anchored to the diagram.",
      body: "A tablet in front of every student shows the same problem in their language blend, grounded against the curriculum — not the model’s priors. The teacher is the source of truth; the agent is the translator.",
      meta: "stack: cells/ env + curriculum schema · grounding: textbook-anchored · personal: per-student",
    },
  },
];

export function BeforeAfter(): JSX.Element {
  return (
    <section className="section beforeafter" id="shift">
      <div className="ba__bg" aria-hidden>
        <span className="ba__bg-mark ba__bg-mark--left">NOW</span>
        <span className="ba__bg-mark ba__bg-mark--right">NEXT</span>
      </div>

      <div className="shell ba__shell">
        <header className="ba__header">
          <span className="eyebrow">§07 — before / after</span>
          <h2 className="ba__title">
            Five rooms in India.
            <br />
            <em>Same primitive,</em> different ending.
          </h2>
          <p className="ba__sub">
            Drift-aware Indic voice agents are not a benchmark — they are a
            change in what a Tuesday afternoon feels like for the next 800M
            people. Five real moments, before and after the same substrate
            ships.
          </p>
          <div className="ba__legend">
            <span className="ba__legend-cell ba__legend-cell--before">
              <span className="mono ba__legend-tag">/now</span>
              <span className="ba__legend-text">today, English-only floors and frozen schemas.</span>
            </span>
            <span className="ba__legend-arrow mono" aria-hidden>→</span>
            <span className="ba__legend-cell ba__legend-cell--after">
              <span className="mono ba__legend-tag">/next</span>
              <span className="ba__legend-text">DriftCall’s primitives, in five vendor surfaces.</span>
            </span>
          </div>
        </header>

        <ol className="ba__pairs">
          {PAIRS.map((p, i) => (
            <li
              className="ba__pair"
              key={p.id}
              id={`shift-${p.id}`}
              style={{ ["--pair-i" as string]: i }}
            >
              {/* the central hinge — book-binding clamps + counter */}
              <div className="ba__hinge" aria-hidden>
                <span className="ba__hinge-clamp ba__hinge-clamp--top" />
                <span className="ba__hinge-counter mono">
                  <span className="ba__hinge-num">{p.num}</span>
                  <span className="ba__hinge-arrow">→</span>
                </span>
                <span className="ba__hinge-clamp ba__hinge-clamp--btm" />
              </div>

              {/* BEFORE panel */}
              <article className="ba__panel ba__panel--before">
                <header className="ba__panel-head">
                  <span className="mono ba__panel-tag">/now · {p.num}</span>
                  <span className="mono ba__panel-ctx">{p.context}</span>
                </header>
                <p className="ba__scene">{p.before.scene}</p>
                <p className="ba__body">{p.before.body}</p>
                <p className="mono ba__meta">{p.before.meta}</p>
                <span className="ba__panel-grain" aria-hidden />
              </article>

              {/* AFTER panel */}
              <article className="ba__panel ba__panel--after">
                <header className="ba__panel-head">
                  <span className="mono ba__panel-tag">/next · {p.num}</span>
                  <span className="mono ba__panel-ctx">{p.context}</span>
                </header>
                <p className="ba__scene">{p.after.scene}</p>
                <p className="ba__body">{p.after.body}</p>
                <p className="mono ba__meta">{p.after.meta}</p>
                <span className="ba__panel-flare" aria-hidden />
              </article>
            </li>
          ))}
        </ol>

        <footer className="ba__footer">
          <span className="mono ba__footer-mark" aria-hidden>↳</span>
          <p className="ba__footer-line">
            Five rooms become a million. The substrate is the same one
            <em> training right now </em> on this Space.
          </p>
        </footer>
      </div>
    </section>
  );
}
