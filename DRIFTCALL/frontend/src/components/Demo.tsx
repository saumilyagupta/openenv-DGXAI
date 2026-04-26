import { META } from "../data/content";

import "./Demo.css";

export function Demo(): JSX.Element {
  const spaceUrl = `https://huggingface.co/spaces/${META.demoSpace}`;
  const embedUrl = `https://${META.demoSpace.replace("/", "-").toLowerCase()}.hf.space`;

  return (
    <section className="section demo" id="demo">
      <div className="shell demo__shell">
        <header className="demo__header">
          <div>
            <span className="eyebrow">§03 — live demo</span>
            <h2 className="demo__title">
              Speak to it.
              <br />
              <em>Watch it adapt.</em>
            </h2>
          </div>
          <p className="demo__sub">
            Press <span className="mono demo__kbd">⏺</span>, ask in any of the
            five languages, and the agent will walk through the full
            tool-calling chain. Use the <em>drift dropdown</em> mid-episode to
            inject one of twenty schema mutations and watch the trace recover.
            Toggle <em>base</em> vs <em>trained</em> to A/B the LoRA.
          </p>
        </header>

        {/* Trace prompts as decorative side rail. */}
        <div className="demo__layout">
          <aside className="demo__prompts" aria-label="example prompts">
            <span className="kicker">try one</span>
            <ul>
              <li>
                <span className="devanagari">9 बजे से पहले एक वेज थाली ₹500 के अंदर मिलनी चाहिए</span>
                <span className="mono demo__prompt-tag">restaurant · drift: tax_added</span>
              </li>
              <li>
                <span className="devanagari">कल सुबह 8 बजे की दिल्ली से बेंगलुरु फ्लाइट</span>
                <span className="mono demo__prompt-tag">airline · drift: field_renamed</span>
              </li>
              <li>
                Book me a non-AC cab from Indiranagar to MG Road for ₹250.
                <span className="mono demo__prompt-tag">cab · drift: enum_pruned</span>
              </li>
              <li>
                <span className="devanagari">होटल चाहिए, ₹3000 के अंदर, कल चेक-इन</span>
                <span className="mono demo__prompt-tag">hotel · drift: cancel_window_shrunk</span>
              </li>
            </ul>
            <a
              className="demo__hf-link mono"
              href={spaceUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              open in huggingface ↗
            </a>
          </aside>

          {/* The iframe frame. Bezels, recording-light, scanlines. */}
          <div className="demo__frame" role="region" aria-label="DriftCall live demo">
            <div className="demo__bezel">
              <span className="demo__bezel-dot" aria-hidden />
              <span className="mono demo__bezel-id">{META.demoSpace}</span>
              <span className="mono demo__bezel-rec">
                <span className="demo__bezel-rec-dot" /> rec
              </span>
            </div>
            <iframe
              className="demo__iframe"
              src={embedUrl}
              title="DriftCall Gradio demo"
              loading="lazy"
              allow="microphone; clipboard-read; clipboard-write"
            />
            <div className="demo__scanlines" aria-hidden />
          </div>
        </div>
      </div>
    </section>
  );
}
