import { META } from "../data/content";

import "./Footer.css";

export function Footer(): JSX.Element {
  return (
    <footer className="footer">
      <div className="shell footer__shell">
        <div className="footer__top">
          <span className="footer__brand">
            DriftCall <em className="footer__deva">ड्रिफ़्ट</em>
          </span>
          <span className="mono footer__hack">{META.hackathon}</span>
        </div>

        <div className="footer__grid">
          <p className="footer__about">
            DriftCall is built on Gemma 3n E2B (Unsloth quantised) plus a custom
            native PyTorch GRPO loop. Five reward components, twenty drift
            patterns, five Indic languages, no LLM judges. The repo, the
            adapter, the env, and the demo are all public — the entire pipeline
            is reproducible from a single <code className="mono">build_all.sh</code>.
          </p>

          <ul className="footer__credits">
            <li>
              <span className="footer__credit-key mono">env spec</span>
              <span className="footer__credit-val">DESIGN.md (54 KB) · 14 module docs</span>
            </li>
            <li>
              <span className="footer__credit-key mono">trainer</span>
              <span className="footer__credit-val">scripts/train_driftcall_grpo.py</span>
            </li>
            <li>
              <span className="footer__credit-key mono">eval</span>
              <span className="footer__credit-val">cells/step_18..20 · 50-ep + 200-probe</span>
            </li>
            <li>
              <span className="footer__credit-key mono">demo</span>
              <span className="footer__credit-val">demo/app_gradio.py · 28 KB</span>
            </li>
          </ul>
        </div>

        <div className="footer__rule" />

        <div className="footer__bottom">
          <span className="mono">© 2026 · DriftCall · apache-2.0</span>
          <span className="mono">type: instrument serif × geist · drift: 0.000</span>
        </div>
      </div>
    </footer>
  );
}
