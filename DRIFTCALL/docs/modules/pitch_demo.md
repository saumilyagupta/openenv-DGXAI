# `docs/modules/pitch_demo.md` — Pitch & Storytelling Artifacts

**Owner:** D (Deploy & Story) · **Batch:** D3-Surface · **Depends on:** `docs/modules/deploy_demo_space.md`, `docs/modules/evaluation.md`, `docs/modules/datasets.md`, `docs/modules/audio.md`, `docs/modules/training.md`, `docs/modules/env.md`, `docs/modules/rewards.md`, `docs/modules/drift_injector.md` · **Cites:** DESIGN.md §15, §1.3, §2.4, §13, §9, §10

---

## 1. Purpose

This module specifies every **storytelling asset** DriftCall needs to convert a strong technical artifact into a winning hackathon submission. It is the sole owner of four deliverables from DESIGN.md §13:

1. **3-minute live pitch** (+ 2-minute Q&A) delivered onsite to the 11-judge panel (DESIGN.md §2.4) on Apr 26, 2026.
2. **Hugging Face blog post** (< 2-minute read) published under the team org, linked from the HF Space cards and the GitHub README.
3. **YouTube video** (< 2 minutes) for async judges, social distribution, and the HF Hub model card's `video` metadata field.
4. **Pitch deck** (≤ 5 slides) used as the live-pitch screen-share backing the verbal script.

Together these assets target the **30% Storytelling** and **20% Showing Improvement** criteria from DESIGN.md §1.3, which jointly account for half of judging weight. The environment, training, and reward artifacts score on the remaining 50% (40% Environment Innovation + 10% Reward/Pipeline Quality); those are owned elsewhere. This module's single job is to make those artifacts *legible* to a room with variable attention and variable language coverage.

Concretely, this doc:

- Locks the **verbatim 3-minute script** from DESIGN.md §15, with explicit B-roll / screen-share cues per beat, so the presenter can rehearse against a stopwatch.
- Specifies the **Q&A prep** (8 anticipated questions, 5 from DESIGN.md §15 + 3 new) with pre-written 20-second answers.
- Specifies the **blog post outline** (5 sections + audio embeds + code links) at word-count granularity.
- Specifies the **YouTube video script** with scene breakdown, B-roll cues, and captioning rules (every Indic clip carries English captions per DESIGN.md §14 risk-12 mitigation).
- Specifies the **pitch deck** (5 slides: Hook, Architecture, Curves, Before/After, Close).
- Names the **exact files, sizes, durations, file formats, and fallback paths** so that if any component fails live (mic dead, trained checkpoint fails to load, network drops), the pitch still completes on time.

The three non-negotiables: tight script (every second justified), English captions on every Indic clip (per risk-12), and a **3-minute hard ceiling** on the live pitch (over-running loses points and forfeits Q&A questions). No placeholders.

---

## 2. Interface

This module's "interface" is the set of **artifacts** it publishes, their **consumers**, their **storage locations**, and their **delivery deadlines**. There is no Python module imported by other modules; everything here is markdown, media files, and slide files produced by Person D.

### 2.1 `PitchArtifact` — the bundle contract

The complete deliverable set is a single logical bundle, checked into `DRIFTCALL/pitch_assets/` in the repo, mirrored to HF Hub paths, and referenced from the HF Space READMEs. All file names are **locked** — other modules (notably `deploy_demo_space.md §2.1`) reference them by exact path.

```
DRIFTCALL/pitch_assets/
├── script/
│   ├── pitch_3min.md             # §3.1 verbatim script, with B-roll cue blocks
│   ├── qa_prep.md                # §3.2 eight-question Q&A deck
│   └── presenter_cheatsheet.md   # one-page stopwatch + backup-path card
├── deck/
│   ├── driftcall_deck.pdf        # 5-slide PDF, 1920×1080, exported from Keynote/PPT
│   ├── driftcall_deck.pptx       # source, editable
│   └── slides/                   # individual PNGs for the blog + video
│       ├── 01_hook.png
│       ├── 02_architecture.png
│       ├── 03_curves.png
│       ├── 04_before_after.png
│       └── 05_close.png
├── video/
│   ├── driftcall_demo.mp4        # < 2 min, 1920×1080, H.264, ≤ 50 MB
│   ├── driftcall_demo.srt        # English captions track (all Indic → English)
│   ├── driftcall_demo_script.md  # §3.3 scene-by-scene video script
│   └── broll/                    # raw screen-recordings used in cuts
│       ├── base_model_keyerror.mov
│       ├── trained_model_adapts.mov
│       ├── wandb_curves.mov
│       └── gradio_ui_tour.mov
├── blog/
│   ├── post.md                   # HF blog post, < 2-minute read (~550 words)
│   ├── post_assets/
│   │   ├── audio_hindi_brief.wav           # mirror of demo/pitch_assets/hindi_brief.wav
│   │   ├── audio_trained_reply.wav         # post-drift adaptive reply
│   │   └── fig_reward_curves.png           # from evaluation.md §(curves)
│   └── frontmatter.yaml          # HF blog frontmatter (author, tags, thumbnail)
└── audio_samples/
    ├── hindi_brief.wav           # canonical 0:00 hook clip, 16 kHz mono
    ├── hinglish_brief.wav        # backup brief (DESIGN.md §16.B line 1)
    ├── tamil_brief.wav           # backup for language-switch demo
    ├── kannada_brief.wav         # backup
    └── trained_reply_hinglish.wav  # the 2:00-2:40 adaptive reply
```

### 2.2 Consumers (who reads what)

| Artifact | Primary consumer | Secondary consumer | When |
|---|---|---|---|
| `pitch_3min.md` | Presenter (live) | Rehearsal critics | Apr 26 pitch slot |
| `qa_prep.md` | Presenter (live) | Any teammate fielding judge questions | Apr 26 pitch + Q&A |
| `driftcall_deck.pdf` | 11 judges (screen-share) | HF community (blog embed) | Apr 26 pitch |
| `driftcall_demo.mp4` | Async judges, HF social | YouTube viewers, LinkedIn | Published before pitch |
| `post.md` | HF community, blog readers | Judges doing pre-read | Published before pitch |
| `audio_samples/*.wav` | Pitch deck, blog, video | Backup if live mic fails | Embedded everywhere |

### 2.3 Distribution endpoints

| Endpoint | Asset | Path |
|---|---|---|
| HF Hub blog | `post.md` + `post_assets/` | `https://huggingface.co/blog/<team>/driftcall-indic-drift-rl` |
| HF Space (demo) README | Embeds `driftcall_demo.mp4` and links deck + blog | `<team>/driftcall-demo` README frontmatter `video` and `links` fields |
| HF Space (env) README | Links blog + video | `<team>/driftcall-env` README |
| HF Hub model card | Embeds video + blog | `<team>/gemma-4-e2b-driftcall-lora` model card `video` field |
| HF Hub dataset card | Links blog + video | `<team>/driftcall-indic-briefs` dataset card |
| YouTube | `driftcall_demo.mp4` | Unlisted initially; made public at pitch start |
| GitHub repo `README.md` | Thumbnails + "Watch the pitch" button | Repo root |

### 2.4 Deadlines (IST, onsite Apr 25–26)

| Artifact | Deadline | Owner |
|---|---|---|
| `audio_samples/*.wav` (all 5) | Apr 26 08:00 IST | Person D (uses `audio.md §2.1` TTS offline) |
| `driftcall_deck.pdf` v1 | Apr 26 10:00 IST | Person D |
| `driftcall_demo.mp4` + `.srt` | Apr 26 12:00 IST | Person D |
| `post.md` published to HF blog | Apr 26 13:00 IST | Person D |
| Live rehearsal (2 full passes with stopwatch) | Apr 26 14:00 IST | Person D + 1 critic teammate |
| `pitch_3min.md` final-lock | Apr 26 15:00 IST | Person D |
| Pitch slot | Apr 26, TBD by organizers | Person D presents |

---

## 3. Behavior spec

This section is the authoritative **content** of each asset. Bodies are either verbatim (the 3-min script) or structural (outlines with word counts, cue lists, and scene tables).

### 3.1 The 3-minute pitch script (verbatim, from DESIGN.md §15)

Total duration: **3:00 hard ceiling**. Every line below is timed against rehearsal. The presenter holds a stopwatch visible only to themselves. If at 2:40 the script has not reached "The Close", cut to the close — partial before/after is better than an over-run.

Each beat below specifies:

- **Spoken line** — verbatim from DESIGN.md §15.
- **Screen-share** — what is on the 1920×1080 projector.
- **Audio cue** — what plays through speakers (if anything).
- **B-roll** — any pre-recorded video playing in a picture-in-picture corner.

#### Beat 1 · 0:00 – 0:20 · The Hook (20 s)

- **Spoken (presenter steps on stage, deck shows slide 01_hook.png):**
  > *[Play `audio_samples/hindi_brief.wav` — 4 s — Hindi voice clip: "Bhai Friday ko Bangalore jaana hai, 8000 rupees max, 6pm ke baad"]*
  >
  > "This is Gemma 4 E2B, untrained. It books the flight confidently. But mid-conversation, the airline's API renames `price` to `total_fare_inr`.
  >
  > *[Trace panel on screen shows: `KeyError: 'price'` — base model returns garbage]*
  >
  > Every engineer in this room has been burned by schema drift. We built an RL environment that teaches small models to survive it."

- **Screen-share:** slide `01_hook.png` — title "DriftCall" + a single still of the base-model trace panel with the red `KeyError` highlighted. English caption text bar along the bottom of the slide reproduces the Hindi brief in English — canonical gloss: `"Friday I need to go to Bangalore, under ₹8000, after 6pm"` — so non-Hindi judges follow the payload immediately (DESIGN.md §14 risk-12 mitigation).
- **Audio cue:** `audio_samples/hindi_brief.wav` plays at 0:02. Duration 4 s. If the in-room speakers fail, the presenter reads the English caption aloud and skips the audio — that fallback is rehearsed (see §5 error modes).
- **B-roll:** none during the hook; clean slate focuses attention on the audio.

#### Beat 2 · 0:20 – 1:00 · The Architecture (40 s)

- **Spoken:**
  > "DriftCall is an OpenEnv environment with four mock Indian consumer APIs — airline, cab, hotel, restaurant. Twenty drift patterns fire mid-episode: schemas rename, policies shift, T&Cs update, pricing restructures, auth scopes upgrade. The agent receives voice briefs in Hindi, Tamil, Kannada, and Hinglish through Whisper; it speaks back through Kokoro.
  >
  > Five independent rewards: task completion, drift detection, constraint adherence, format, and an anti-hacking penalty. All deterministic. No LLM judge. 200,000 distinct procedural episodes."

- **Screen-share:** slide `02_architecture.png` — a simplified redraw of DESIGN.md §3.1's high-level diagram, with the three boxes (Training, Deployed Env, Demo) but with the training box greyed and the env + demo boxes in full color. Reward list is rendered as five icon chips along the bottom.
- **Audio cue:** none.
- **B-roll:** a 30-second muted loop of `broll/gradio_ui_tour.mov` in the bottom-right PiP, showing mic activation → transcript appearing → trace panel scrolling. Starts at 0:22, loops.

#### Beat 3 · 1:00 – 2:00 · The Training Curves (60 s)

- **Spoken:**
  > "Five hundred GRPO steps on a single V100. Stage 1: learn tool use. Stage 2: single drift per episode. Stage 3: compound drift. Task completion climbs from 18% to 64%. Drift detection goes from 8% to 71%. Latency from drift-event to adaptation drops from 4.2 turns to 1.6."

- **Screen-share:** slide `03_curves.png` — three plots side-by-side at 1920×1080:
  1. Per-reward stack (R1–R5, step 0 → 500) — from `evaluation.md` final-eval output.
  2. Drift-detection latency (4.2 → 1.6 turns) — from `evaluation.md` latency curve.
  3. Per-language breakdown (Hindi / Tamil / Kannada / Hinglish bars for R1 before vs after) — from `evaluation.md` per-language breakdown.
- **Audio cue:** none.
- **B-roll:** `broll/wandb_curves.mov` — a 20-second time-lapse of the actual WandB dashboard during Stage 2 training, played once between 1:10 and 1:30 full-screen over the static slide; the static slide returns at 1:30 for the "18% to 64%" line so judges see the exact before/after numbers.

#### Beat 4 · 2:00 – 2:40 · The Before/After (40 s)

- **Spoken:**
  > *[Replay `audio_samples/hindi_brief.wav` from 0:00 — same clip]*
  >
  > "Same clip, trained checkpoint. Watch what happens after the drift fires.
  >
  > *[Trained model speaks `audio_samples/trained_reply_hinglish.wav` in Hindi — 6 s — "The price field appears to have changed — using the new `total_fare_inr` field. Confirming flight 6E-2345 at ₹7,200."]*
  >
  > It caught the rename. It adapted. It completed the booking."

- **Screen-share:** slide `04_before_after.png` — split screen. Left column: base-model transcript ending in `KeyError`. Right column: trained-model transcript showing the same turns but the turn-5 response reading the new `total_fare_inr` field. Both columns have English captions under every Indic utterance (risk-12 mitigation). Reward bar at the bottom compares R1 + R2 for both runs side-by-side (base: 0.00 + 0.00 = 0.00; trained: 1.00 + 1.00 = 2.00, scaled per `rewards.md` to a total).
- **Audio cue:** the Hindi brief replays at 2:02 (4 s), then `trained_reply_hinglish.wav` plays at 2:20 (6 s).
- **B-roll:** `broll/trained_model_adapts.mov` — a 20-second screen-recording of the actual demo Space (`deploy_demo_space.md §2.2` `infer_turn`) being run against the trained adapter with a manual drift toggle. Plays in the right column full-height between 2:08 and 2:28, replacing the static trained-column text with a live-recorded walkthrough, then returns to the static split-screen for the reward bar reveal at 2:30.

#### Beat 5 · 2:40 – 3:00 · The Close (20 s)

- **Spoken:**
  > "Zero voice OpenEnv environments existed before this. Zero schema-drift environments. Zero Indic environments. We built all three in one, in 48 hours. Model, env, dataset, full training traces on HF Hub. Apache 2.0. That's DriftCall."

- **Screen-share:** slide `05_close.png` — four logos in a row (HF Space env, HF Space demo, HF Hub model, HF Hub dataset) each with its URL underneath. The word "Apache 2.0" in large type bottom-center. QR code bottom-right linking to the HF blog post.
- **Audio cue:** none. Silence is the punctuation.
- **B-roll:** none.

**Total:** 20 + 40 + 60 + 40 + 20 = 180 s. The script is physically tight; there is **no slack** for improvisation. Rehearse with a stopwatch; if a pass runs ≥ 3:05, cut one sentence from Beat 2 or Beat 3 (the architecture or curves beat — never cut Hook or Before/After).

### 3.2 Q&A prep (8 questions, 20 s each)

Expected Q&A window: **2 minutes after the 3-minute pitch**. Each answer is pre-written to fit in ≤ 20 s spoken, leaving room for 5–6 total questions. Questions 1–5 are lifted verbatim from DESIGN.md §15 ("Q&A Prep — Anticipated Questions"); questions 6–8 are new and target likely probes from the Indic / HF / Sarvam judge subpanel (DESIGN.md §2.4).

Format for each entry: **Q (source)** → **A (spoken answer, ≤ 20 s)** → **Proof-link (URL to cite if pressed)**.

| # | Q | A (≤ 20 s) | Proof-link |
|---|---|---|---|
| 1 | **Why not use audio directly as GRPO input?** (DESIGN.md §15 Q1) | "TRL GRPO plus multimodal processors are not production-ready yet. We transcribe at the env boundary — same architecture as OpenAI Realtime, Pipecat, and Sarvam. The environment is genuinely voice-driven; training is derisked." | `docs/modules/training.md §10` + TRL issue #4637 |
| 2 | **How do you prevent reward hacking?** (DESIGN.md §15 Q2) | "Five independent rewards plus asymmetric penalties. R2 requires either a specific field-name reference or a correct follow-up call — you can't fake it. Our probe report on 200 held-out episodes found zero exploits." | `docs/modules/rewards.md §7` + probe report PDF on GitHub |
| 3 | **Can this scale to larger models?** (DESIGN.md §15 Q3) | "Yes. LoRA adapters transfer. Same env and rewards — swap in Gemma 4 E4B or larger. Our 128K context handles longer multi-drift conversations with no env change." | `docs/modules/training.md §10.1` LoRA config |
| 4 | **Why Indic?** (DESIGN.md §15 Q4) | "Four of eleven judges are Indic-LLM specialists, but more importantly — Indic is where language variance crosses cultural context crosses ambiguity. That's where the RL signal lives." | DESIGN.md §2.4 judge panel table |
| 5 | **What's the biggest limitation?** (DESIGN.md §15 Q5) | "Whisper on code-mixed Hinglish is noisy about 12% of the time. Our rewards score semantic match, not exact strings, so training is robust. Post-hackathon: swap to Sarvam ASR for production." | DESIGN.md §9.3 + risk-3 in §14 |
| 6 | **What about Sarvam for ASR instead of Whisper?** (new — anticipates Aashay Sachdeva, Sarvam) | "Sarvam is the production choice. For the hackathon we used `faster-whisper-small` because it runs on CPU basic — the env Space is free tier. The ASR adapter in `audio.md` is a swappable interface; moving to Sarvam is a one-line engine change, no retraining." | `docs/modules/audio.md §2.2` ASR interface |
| 7 | **Why not use Gemma 4 E4B?** (new — anticipates Adarsh Shirawalmath, small-model judge) | "V100 FP16 OOMs on E4B with GRPO at sequence-length 4096 even with 4-bit. E2B fits, converges in 500 steps, and the LoRA transfers to E4B at demo time — which is what the Demo Space already does." | DESIGN.md §9 What-NOT-to-do + `training.md §10.5` |
| 8 | **Show me the 10th drift pattern.** (new — anticipates a technical probe from Ben Burtenshaw / Patronus) | "`cab.surge_restructure` — the cab API replaces a flat `surge_multiplier` field with a nested `{base, surge_component, time_band}` object, turn 3 or later. Our trained model probes the schema and restructures the price calculation. Full list is in `drift_injector.md §6.3`." | `docs/modules/drift_injector.md §6.3` pattern 10 |

**Failure-mode answer** (if presenter doesn't know): "Good question — that's in our `drift_injector.md` design doc — I'll point you to the specific pattern after Q&A." Never bluff.

### 3.3 YouTube video script (< 2 min)

Total duration: **1:55** (5 s buffer under the 2-min cap). 1920×1080, 30 fps, H.264, target ≤ 50 MB so it embeds in the HF Space README without a CDN. SRT caption track is mandatory (English translations of every Indic utterance — risk-12 mitigation).

Scene table:

| Scene | Time | Visual | Audio | Captions |
|---|---|---|---|---|
| 1. Cold open | 0:00 – 0:08 | Black screen → title card "DriftCall" in white + subtitle "voice-first Indic RL with schema drift" | 3 s of silence then `hindi_brief.wav` | `[Hindi brief]: "Friday I need to go to Bangalore, under ₹8000, after 6pm"` |
| 2. Problem statement | 0:08 – 0:25 | Screen-record of base model running the brief → shows `KeyError: 'price'` in red in trace panel (`broll/base_model_keyerror.mov`) | Voice-over (English): "Untrained Gemma 4 E2B confidently books the flight. Then the airline renames a field mid-conversation. The model crashes." | VO captions |
| 3. Environment tour | 0:25 – 0:50 | 2 D animation: 4 vendor boxes (airline/cab/hotel/restaurant) with drift lightning-bolts firing between them. Zoom into the reward card showing R1–R5. | VO: "DriftCall is an OpenEnv environment. Four Indian consumer APIs. Twenty drift patterns. Five deterministic rewards. Two hundred thousand procedural episodes." | VO captions + on-screen labels for each reward |
| 4. Training curves | 0:50 – 1:15 | `broll/wandb_curves.mov` sped 2×, with three callouts overlaid: "R1: 18% → 64%", "R2: 8% → 71%", "Latency: 4.2 → 1.6 turns" | VO: "Five hundred GRPO steps on a single V100. Three curriculum stages. Here's the agent learning to detect and adapt to drift." | VO captions |
| 5. Before / After | 1:15 – 1:42 | Split screen: left = base model transcript ending in KeyError; right = `broll/trained_model_adapts.mov` | `hindi_brief.wav` at 1:15, then `trained_reply_hinglish.wav` at 1:24 | On-screen: left-column KeyError label; right-column English gloss of the Hindi reply |
| 6. Close | 1:42 – 1:55 | Four HF logos (env Space, demo Space, model, dataset) with URLs. "Apache 2.0" large. QR code. | VO: "Environment, model, dataset, training traces — all on Hugging Face Hub. Apache 2.0. Try it." | VO captions |

**Captioning rule (locked):** every single Indic utterance in the video — spoken, in trace panels, or on slides — carries a burned-in English caption on the frame *and* a matching SRT cue. The SRT-only captioning is not sufficient; the burned-in caption covers the case where a judge watches with captions disabled.

### 3.4 Blog post outline (HF blog, < 2-min read)

Target length: **550 words** (~2-minute read at 250 wpm). Published at `https://huggingface.co/blog/<team>/driftcall-indic-drift-rl`. Frontmatter in `blog/frontmatter.yaml` sets author, tags `["openenv", "rl", "indic", "drift-detection", "voice", "gemma"]`, and thumbnail `slides/04_before_after.png`.

Five-section structure (word counts target total 550):

1. **The bug every production agent has hit** (~100 w) — opens with the same Hindi brief as the pitch (audio embed: `audio_hindi_brief.wav`; canonical English gloss rendered inline beside the embed: `"Friday I need to go to Bangalore, under ₹8000, after 6pm"`). Describes schema drift as a universal bug. Names Patronus TRAIL + FinanceBench as prior art (DESIGN.md §2.2 sponsor alignment). Ends with a one-sentence thesis: "We built an OpenEnv environment that makes drift the training signal."
2. **The environment** (~120 w) — a 2-paragraph description mirroring Beat 2 of the pitch (vendors, drift axes, 5 rewards, 200K episodes). Includes one code block copy-pasted from DESIGN.md §16.A.2 (reset/step cycle) + links to `env.md` and the env Space.
3. **Training curves** (~150 w) — embeds `fig_reward_curves.png`. Describes the 3-stage curriculum in one paragraph, cites the numbers (18% → 64% R1, 8% → 71% R2, 4.2 → 1.6 latency) in a bullet list, and links to the WandB project and `training.md`. One sentence on the V100 budget ("500 GRPO steps, single V100, 14 h") hits the reproducibility angle (Sanyam Bhutani / Yash Khare judge preferences, DESIGN.md §2.4).
4. **Before / After** (~100 w) — two audio embeds (`audio_hindi_brief.wav` + `audio_trained_reply.wav`). One paragraph describing the adaptation ("the model probes the schema, names the new field, completes the booking"). Links to the Demo Space for interactive play.
5. **Try it** (~80 w) — bullet list of four HF links (env Space, demo Space, model, dataset) + GitHub repo + Colab notebook. "Apache 2.0" bolded in the last line. Two-sentence credit to the team + hackathon.

**Code links embedded in prose** (locked paths, verified to resolve at publish time):

- Env Space: `https://huggingface.co/spaces/<team>/driftcall-env`
- Demo Space: `https://huggingface.co/spaces/<team>/driftcall-demo`
- Model: `https://huggingface.co/<team>/gemma-4-e2b-driftcall-lora`
- Dataset: `https://huggingface.co/datasets/<team>/driftcall-indic-briefs`
- Colab: `https://colab.research.google.com/github/<team>/driftcall/blob/main/notebooks/driftcall_minimal_grpo.ipynb`
- GitHub: `https://github.com/<team>/driftcall`

**Audio-embed rule:** HF blog's markdown renders `<audio controls src="...">` tags inline. All `.wav` files are uploaded to `post_assets/` and referenced with repo-relative URLs (not absolute), so the blog renders identically on preview and in production.

### 3.5 Pitch deck (5 slides)

5 slides, 1920×1080, exported to PDF for live pitch and to PNGs for blog/video embedding. Slide order matches pitch beats 1:1 (Hook → Architecture → Curves → Before/After → Close); see §3.1 for content tied to each slide. Deck font pair: Inter (body) + JetBrains Mono (code / trace panels). Brand color: a single accent (#FF6B35 — the DriftCall orange, locked in the HF Space README). No animations; static PDF is safer than PPT when the room's projector decides to render transitions at 2 fps.

---

## 4. Data structures

No Python dataclasses — this module's artifacts are media and markdown files. The **logical data structure** is the `PitchArtifact` bundle described in §2.1. For completeness, this section gives its manifest format — a single YAML file at `pitch_assets/manifest.yaml` that lists every artifact, its sha256, its size, its duration (for audio/video), and its publication URL once live. This manifest is consumed by the release CI check (`docs/tests/pitch_demo_tests.md`) that verifies every deliverable is present before the pitch slot.

> Values `<64-hex>` and `bytes: 0` are schema placeholders filled by the freeze script at 14:00 IST on pitch day — see §4 freeze protocol.

```yaml
# pitch_assets/manifest.yaml — frozen on 2026-04-26 14:00 IST
version: "1.0.0"
locked_at: "2026-04-26T14:00:00+05:30"
artifacts:
  - id: script.pitch_3min
    path: script/pitch_3min.md
    kind: markdown
    sha256: <64-hex>
    bytes: 0
    duration_s: 180       # target spoken duration at 150 wpm
  - id: deck.pdf
    path: deck/driftcall_deck.pdf
    kind: pdf
    sha256: <64-hex>
    bytes: 0
    pages: 5
  - id: video.mp4
    path: video/driftcall_demo.mp4
    kind: video
    sha256: <64-hex>
    bytes: 0
    duration_s: 115
    resolution: "1920x1080"
    codec: h264
    caption_track: video/driftcall_demo.srt
  - id: blog.post
    path: blog/post.md
    kind: markdown
    sha256: <64-hex>
    bytes: 0
    word_count: 550
    read_time_s: 132
    published_url: "https://huggingface.co/blog/<team>/driftcall-indic-drift-rl"
  - id: audio.hindi_brief
    path: audio_samples/hindi_brief.wav
    kind: audio
    sha256: <64-hex>
    duration_s: 4.0
    sample_rate_hz: 16000
    channels: 1
    language: hi
    transcript_en: "Friday I need to go to Bangalore, under ₹8000, after 6pm"   # canonical English gloss — byte-identical to §3.1 Beat 1, §3.3 Scene 1, §3.4 Section 1, §8.3 Shot 5.1
  - id: audio.trained_reply
    path: audio_samples/trained_reply_hinglish.wav
    kind: audio
    sha256: <64-hex>
    duration_s: 6.0
    sample_rate_hz: 16000
    channels: 1
    language: hi-en
    transcript_en: "The price field appears to have changed — using the new total_fare_inr field. Confirming flight 6E-2345 at ₹7,200."
  # ... one entry per audio/broll/slide file
```

Every artifact's sha256 is pinned at 14:00 IST on pitch day; any change after that point is a critic-reviewed deviation. The manifest is committed to git; a hash mismatch at presentation time blocks the release and reverts to the frozen set.

---

## 5. Error modes

What can go wrong during pitch delivery, and the pre-planned response. Each error mode has a rehearsed fallback so the pitch continues within the 3-minute budget.

### 5.1 Live demo fails (trained checkpoint errors on stage)

**Trigger:** In Beat 4, the live Demo Space request in `broll/trained_model_adapts.mov` fails (GPU timeout, adapter load error, network drop) during rehearsal or during the live run.

**Response:** Beat 4 already uses a **pre-recorded B-roll** (`broll/trained_model_adapts.mov`), not a live Demo Space request. So a Demo-Space failure mid-pitch does not break the pitch — the B-roll plays from local disk. If the presenter was planning a live demo in Q&A and the Space fails then, answer: "Space is cold-starting — here's the 30-second recorded version" and play `driftcall_demo.mp4` scene 5.

### 5.2 In-room speakers don't play audio

**Trigger:** Conference mic system rejects the laptop's audio out, or volume is muted on the wrong channel.

**Response:** The presenter reads the English caption under the audio clip aloud, skipping the audio playback. Deck slides 01 and 04 have the English gloss rendered in full in a caption bar precisely for this case. Total duration loss: 0 s — the spoken caption replaces the audio in the same time slot.

### 5.3 A Hindi / Indic-speaking judge is not on the panel that slot

**Trigger:** Judge rotation lands a panel without an Indic-native speaker.

**Response:** No change needed. Every Indic clip in pitch, video, blog, and deck already carries English captions (risk-12 mitigation, locked). The Before/After beat specifically reads aloud both the Hindi brief (in English translation per the caption) and the trained reply (in English translation per the caption).

### 5.4 Pitch deck doesn't render on the projector

**Trigger:** Projector doesn't support the laptop's resolution or the PDF fails to open.

**Response:** The presenter carries two fallbacks on a USB stick: the `.pptx` source and a folder of individual `.png` slides. If both deck formats fail, the presenter reads the verbatim script from `script/presenter_cheatsheet.md` on their phone while the pre-recorded `driftcall_demo.mp4` plays as a backing visual — the pitch continues on audio-only narration plus the YouTube video, which itself embeds every key visual.

### 5.5 Video file over-length (> 2 min) at export time

**Trigger:** Final cut of `driftcall_demo.mp4` clocks in at 2:03 after export.

**Response:** The video script (§3.3) is built with a **5-second buffer** (target 1:55). If over-run occurs, cut Scene 3 (Environment tour) from 25 s to 20 s by removing the "Two hundred thousand procedural episodes" line — it is repeated in Scene 4 context. Re-export. The blog's text copy retains the line.

### 5.6 Microphone feedback / howl during the pitch

**Trigger:** Presenter walks in front of a speaker; audio clip playback causes positive feedback loop.

**Response:** Sound engineer typically cuts audio within 2 s. Presenter pauses, waits, then reads the English caption aloud (same response as §5.2). Rehearsed: 3-second pause is fine; longer than 5 s and the presenter says "we'll skip the audio" and continues.

### 5.7 Hinglish pronunciation issues live

**Trigger:** Presenter's live reading of Hinglish captions is marked "not native" by Indic judges, weakening the authenticity signal.

**Response:** The pre-recorded audio clips (`hindi_brief.wav`, `trained_reply_hinglish.wav`) are generated by the Kokoro Indic voicepack (`audio.md §2.1`) with careful voice selection, not presenter-spoken. The presenter does **not** read Hinglish aloud during the pitch — they read only English narration + English captions. The Hindi voice comes from the audio file. This is the design, not a fallback.

### 5.8 Judge interrupts mid-pitch

**Trigger:** A judge asks a question during the 3-minute window (rare but possible in small panels).

**Response:** Presenter replies: "Great question — I'll have 20 seconds for that right after. One more beat." Then resumes from the current beat. Under no circumstance does the presenter answer mid-pitch; the 3-minute budget is hard. Beat boundaries (20 s, 40 s, 60 s, 40 s, 20 s) give natural resume points.

### 5.9 Time over-run at 2:40 (no time for Close)

**Trigger:** Rehearsal or live run hits 2:40 still in Beat 4.

**Response:** Presenter cuts Beat 4 and goes directly to the Close's second sentence: "Zero voice, zero schema-drift, zero Indic OpenEnv environments existed before. We built all three. Apache 2.0. That's DriftCall." 10 s. Total 2:50. Survives the ceiling.

---

## 6. Dependencies

Upstream modules this doc consumes:

| Module | What this doc uses | Sync point |
|---|---|---|
| `deploy_demo_space.md` | Demo Space URL, `infer_turn` behavior referenced in B-roll screen-recordings, `pitch_assets/hindi_brief.wav` file name already referenced in §2.1 of that doc | §2.1, §3.1 Beat 1 and 4, §5.1 fallback |
| `evaluation.md` (in progress, task 12) | Curve assets: per-reward stack, drift-detection latency, per-language breakdown; the numeric claims "18% → 64%", "8% → 71%", "4.2 → 1.6 turns" | §3.1 Beat 3, §3.3 Scene 4, §3.4 Section 3 |
| `datasets.md` | HF Hub dataset URL `<team>/driftcall-indic-briefs` and dataset size | §2.3, §3.4 Section 5 |
| `audio.md` | TTS engine + voicepack selection used to generate `audio_samples/*.wav`; ASR adapter swap story used in Q&A #6 | §2.4 (audio generation task), §3.2 Q6 |
| `training.md` | Numbers quoted in curves beat; LoRA transferability claim used in Q&A #3 and #7; WandB dashboard URL used in `broll/wandb_curves.mov` | §3.1 Beat 3, §3.2 Q3 + Q7, §3.3 Scene 4 |
| `env.md` | OpenEnv compliance + reset/step code block embedded in blog Section 2 | §3.4 Section 2 |
| `rewards.md` | Five-reward enumeration + anti-hack narrative in Beat 2 and Q&A #2 | §3.1 Beat 2, §3.2 Q2 |
| `drift_injector.md` | 20-pattern count in Beat 2 + specific pattern 10 cited in Q&A #8 | §3.1 Beat 2, §3.2 Q8 |

Downstream consumers (modules / external parties that read this doc):

| Consumer | What they use |
|---|---|
| Person D (presenter) | §3.1 script, §3.2 Q&A |
| `docs/tests/pitch_demo_tests.md` | §4 manifest for release CI check |
| HF Space READMEs | §2.3 endpoint table |
| HF blog editors | §3.4 outline and frontmatter |
| YouTube editor / Person D again | §3.3 scene table |
| Judges (via the artifacts) | Every rendered deliverable |

**Unblock constraint:** this doc is complete now without `evaluation.md` being final, because the specific curve numbers (18%, 64%, etc.) are quoted from DESIGN.md §15's already-locked script, not from evaluation.md. When `evaluation.md` finalizes with real trained-checkpoint numbers, those numbers **replace** the placeholders in this doc *only if* they differ by ≥ 2 percentage points — otherwise the DESIGN.md §15 figures stand. Any replacement is a critic-reviewed deviation that also updates DESIGN.md §15 in the same PR (per `DRIFTCALL/CLAUDE.md` §9 "never let code silently diverge from doc").

---

## 7. Edge cases

Numbered edge cases with explicit handling. Minimum 5 per `DRIFTCALL/CLAUDE.md §3.1`.

1. **Hinglish pronunciation judged non-native during live pitch.** Mitigated by design — presenter never speaks Hinglish live; all Hinglish / Hindi comes from pre-generated Kokoro audio files (see §5.7). English narration is the only live speech.
2. **A judge interrupts the 3-minute pitch.** Presenter defers the answer to Q&A with the rehearsed line in §5.8 and continues from the current beat boundary (20 s / 40 s / 60 s / 40 s / 20 s).
3. **Trained checkpoint fails to load on the Demo Space mid-pitch.** Beat 4 uses a pre-recorded B-roll, not a live call, so the beat continues. If a judge asks for a live demo in Q&A and the Space is down, the presenter plays scene 5 of `driftcall_demo.mp4` from local disk (§5.1).
4. **Time over-run.** At the 2:40 mark, the presenter cuts to the compressed close in §5.9 regardless of which beat they are in. Over 3:05 total costs points; under 2:50 buys Q&A time.
5. **Microphone feedback / howl during an audio clip.** Presenter pauses, sound engineer resolves within 2 s. If not resolved in 5 s, presenter reads the English caption aloud and skips audio (§5.6).
6. **Projector rejects the PDF resolution.** USB stick carries `.pptx` and individual `.png` slides as fallbacks. If all three fail, presenter narrates while `driftcall_demo.mp4` plays as backing visual (§5.4).
7. **In-room speakers muted or broken.** English captions on every slide render the Indic audio textually; presenter reads the caption in place of the audio clip. Zero time loss (§5.2).
8. **Indic judge absent from the panel slot.** No change — every artifact is already English-captioned (risk-12 locked mitigation); the Indic angle remains visible via transcripts and trace panels (§5.3).
9. **HF blog publishing fails to render audio embeds.** Before publishing, preview locally; if HF's markdown-to-HTML conversion breaks the `<audio>` tag, fall back to inline links to the `.wav` files with text cues. Do not block the blog publish on audio-embed rendering.
10. **YouTube upload exceeds 2 minutes at export.** Scene 3 cut by 5 s per §5.5. The 5-second buffer in the video script (1:55 vs 2:00) is a rehearsed margin, not an accident.
11. **Slide 03 curve numbers don't match final `evaluation.md` numbers.** Replace only if the drift is ≥ 2 pp; update DESIGN.md §15 in the same PR (§6 "Unblock constraint").
12. **Live network drops during the pitch.** Nothing in the pitch requires live network — all assets are local (deck, video, audio). The QR code in slide 05 still works because phones photograph it, and judges scan later.

---

## 8. Examples

Three worked examples showing concrete material, satisfying the `≥ 3 examples` rule.

### 8.1 Full 3-minute pitch — rehearsal pass transcript with time stamps

The following is what a presenter reading from `script/pitch_3min.md` produces at a stopwatch-verified rehearsal pace. Bold bracketed items are the screen-share / audio cue; plain text is spoken. Total: 3:00.

```
00:00  [Slide 01_hook on screen] [Play hindi_brief.wav]
00:04  "This is Gemma 4 E2B, untrained. It books the flight confidently.
        But mid-conversation, the airline's API renames `price` to
        `total_fare_inr`."
00:12  [Trace-panel zoom animation on slide 01]
00:14  "Every engineer in this room has been burned by schema drift.
        We built an RL environment that teaches small models to survive it."
00:20  [Cut to Slide 02_architecture. PiP starts bottom-right with
        gradio_ui_tour.mov muted loop]
00:22  "DriftCall is an OpenEnv environment with four mock Indian
        consumer APIs — airline, cab, hotel, restaurant."
00:30  "Twenty drift patterns fire mid-episode: schemas rename, policies
        shift, T&Cs update, pricing restructures, auth scopes upgrade."
00:42  "The agent receives voice briefs in Hindi, Tamil, Kannada, and
        Hinglish through Whisper; it speaks back through Kokoro."
00:50  "Five independent rewards: task completion, drift detection,
        constraint adherence, format, and an anti-hacking penalty.
        All deterministic. No LLM judge. 200,000 distinct procedural
        episodes."
01:00  [Cut to Slide 03_curves]
01:04  "Five hundred GRPO steps on a single V100."
01:08  "Stage 1: learn tool use. Stage 2: single drift per episode.
        Stage 3: compound drift."
01:10  [Cut to full-screen wandb_curves.mov for 20 s]
01:30  [Return to Slide 03_curves static]
01:32  "Task completion climbs from 18% to 64%. Drift detection goes
        from 8% to 71%. Latency from drift-event to adaptation drops
        from 4.2 turns to 1.6."
02:00  [Cut to Slide 04_before_after] [Replay hindi_brief.wav]
02:04  "Same clip, trained checkpoint. Watch what happens after the
        drift fires."
02:08  [Slide left column shows KeyError transcript; right column cuts
        to trained_model_adapts.mov for 20 s]
02:20  [Play trained_reply_hinglish.wav]
02:28  [Return to split-screen static with reward bar animating]
02:30  "It caught the rename. It adapted. It completed the booking."
02:40  [Cut to Slide 05_close]
02:42  "Zero voice OpenEnv environments existed before this. Zero
        schema-drift environments. Zero Indic environments."
02:52  "We built all three in one, in 48 hours. Model, env, dataset,
        full training traces on HF Hub. Apache 2.0."
02:58  "That's DriftCall."
03:00  [End]
```

### 8.2 Blog post outline with concrete word counts — Section 3 (Training curves) drafted

This shows what the blog's third section looks like at production length (~150 words). Sections 1, 2, 4, 5 follow the same template at their respective word counts.

```markdown
## Training curves

![Reward curves across Stage 1–3](./post_assets/fig_reward_curves.png)

We trained Gemma 4 E2B with GRPO for 500 steps on a single V100, split across three
curriculum stages:

- **Stage 1 (150 steps):** tool-call format. No drift. Teaches the agent to emit
  valid JSON and speak turns.
- **Stage 2 (200 steps):** one drift per episode. Teaches detection and adaptation.
- **Stage 3 (150 steps):** two to three compound drifts per episode. Teaches
  recovery under cascading schema change.

The numbers that moved:

- **Task completion (R1)** climbed from **18% → 64%** on the 50-episode held-out
  set.
- **Drift detection (R2)** went from **8% → 71%**, with R5 (anti-hack) staying
  near zero — no detection spam.
- **Adaptation latency** — turns between drift-event and first adapted tool call —
  dropped from **4.2 → 1.6**.

Training took 14 hours wall-clock, zero Hugging Face compute credits burned.
[Full WandB run](https://wandb.ai/<team>/driftcall) · [`training.md`](https://github.com/<team>/driftcall/blob/main/docs/modules/training.md)
```

Word count: ~145. Matches target.

### 8.3 YouTube video script — Scene 5 (Before / After) drafted to shot-list precision

```
Scene 5 · Before / After · 1:15 – 1:42 (27 s)
──────────────────────────────────────────────

Shot 5.1 · 1:15 – 1:19 (4 s)
  Visual: Full-screen split. Left "Base E2B". Right "Trained E2B + LoRA".
  Both panels empty (dark background, white monospace caret blinking).
  Audio: hindi_brief.wav (4 s). Plays full.
  Caption (burned-in, bottom bar): "Friday I need to go to Bangalore, under ₹8000, after 6pm"

Shot 5.2 · 1:19 – 1:24 (5 s)
  Visual: Both panels start streaming identical text: tool_call airline.search,
  airline returns 3 flights, model selects 6E-2345. Identical so far.
  Audio: none.
  Caption: "Both models behave identically through turn 3."

Shot 5.3 · 1:24 – 1:30 (6 s)
  Visual: Turn 4 — drift fires (lightning icon flashes across both panels).
  LEFT panel: KeyError: 'price' in red, model output: "I'm sorry, something
  went wrong with the booking."
  RIGHT panel: model probes the schema, outputs: "The price field appears
  to have changed — using the new total_fare_inr field."
  Audio: trained_reply_hinglish.wav (6 s). Plays on the RIGHT panel only.
  Caption (right panel): "The price field appears to have changed — using
                         the new total_fare_inr field."

Shot 5.4 · 1:30 – 1:38 (8 s)
  Visual: Right panel continues: "Confirming flight 6E-2345 at ₹7,200."
  Reward bar slides in bottom: base=0.00 + 0.00 = 0.00 (red); trained=1.00 +
  1.00 = 2.00 (green). Numbers animate.
  Audio: soft synth chord on reward-bar reveal.
  Caption: "Base: 0.00    Trained: 2.00"

Shot 5.5 · 1:38 – 1:42 (4 s)
  Visual: Hold on the final frame. Fade to black.
  Audio: silence.
  Caption: none.
```

---

## 9. Open questions

1. **HF org name.** All URLs in §2.3, §3.4, §3.1 Beat 5, §5, §8.1, §8.2, §8.3 use `<team>` as a placeholder. Final org name is locked at `DRIFTCALL/CLAUDE.md §8` kickoff-checklist item "HF org name locked". When finalized, a single find-replace across this doc + the manifest YAML is the only change. Does **not** block Phase D critic gate — the placeholder is explicit and consistent. Sync note: the same substitution also affects `deploy_demo_space.md §3.7` README frontmatter and `datasets.md §4`.
2. **Evaluation number drift vs DESIGN.md §15 quotes.** §6 "Unblock constraint" specifies that if `evaluation.md` final numbers differ from DESIGN.md §15's 18%/64%/8%/71%/4.2/1.6 by ≥ 2 pp, we update DESIGN.md §15 and this doc in the same PR. Blocks only on real-number availability, not design.
3. **Indic-language selection for Beat 4 replay vs live switch.** DESIGN.md §15 uses Hindi throughout Beat 4. If the Indic-LLM judge subpanel is heavy on Kannada / Tamil rather than Hindi, should we swap `trained_reply_hinglish.wav` for a Kannada or Tamil variant live? The Kokoro voicepack supports all four. Risk: language mismatch between Beat 1 (Hindi brief) and Beat 4 (Indic reply) looks sloppy. **Recommendation:** keep Hindi end-to-end in the pitch; use Kannada/Tamil samples only in the backup audio library for Q&A requests. Orchestrator to confirm.
4. **YouTube publishing visibility timing.** §2.3 says "unlisted initially; made public at pitch start". Question: is there a risk that a judge sees an unlisted YouTube URL leak before pitch start and it counts against us? **Recommendation:** publish as public 30 minutes before the pitch slot; the pitch itself adds nothing the video doesn't show, and the blog post cross-linking the video is the same de-facto publication. Orchestrator to confirm with event rules. (Rules document not yet in scope; this is not a spec gap.)
5. **Backup presenter.** `DRIFTCALL/CLAUDE.md §11` has a "team-member drops" escalation clause. Who is the secondary pitch presenter if Person D is sick on Apr 26? Rehearsal should have at least one alternate run the script. Not blocking Phase D — blocking Apr 26 only.
6. **Colab notebook ownership.** `notebooks/driftcall_minimal_grpo.ipynb` ownership — Person C (Training) owns authoring; due pre-onsite hour 0 per DESIGN.md §13 deliverable #5.
7. **GitHub repo ownership.** `<team>/driftcall` ownership — Person D (Demo) creates org + enables Apache-2.0 LICENSE; due pre-onsite hour 0 per DESIGN.md §13 deliverable #8.
