# pitch_demo_tests.md — Test Plan for `pitch_assets/` (DELIVERABLE-ARTIFACT validation)

**Owner:** Person B (Rewards & Tests), secondary: Person D (Deploy & Story — authored module)
**Target module:** `DRIFTCALL/docs/modules/pitch_demo.md` (sealed)
**Implements coverage for:** DESIGN.md §15 (3-min script + Q&A), §1.3 (30% Storytelling + 20% Showing-Improvement), §2.4 (11-judge panel), §13 (deliverables), §14 risk-12 (Indic-caption mitigation)
**Frameworks:** `pytest`, `hypothesis`, `mdformat` (markdown lint), `mutagen` (audio duration/format probe), `pypdf` (PDF slide-count + page-size probe), `pyav` / `ffmpeg -i` dry-probe (video metadata; **no actual render**), `pysrt` (SRT parse), stdlib `hashlib`, `yaml`, `pathlib`, `re`, `urllib.parse`, `socket` (HEAD-request URL liveness).
**Status:** DRAFT — pending ≥ 1 fresh critic round (test-plan gate is lighter per `DRIFTCALL/CLAUDE.md §3.2` Batch D4).

---

## 0. Scope & Non-goals

`pitch_demo.md` is **deliverable-artefact-based**, not code-based: its output is a markdown script, a ≤ 5-slide deck, an ≤ 2-min video, a ≤ 550-word blog post, an SRT caption track, a YAML manifest, and five `.wav` audio clips. There is **no Python module** named `pitch_demo`; `pitch_demo.md §2` explicitly says the interface is "the set of artefacts it publishes". Therefore this test plan is a **validation checklist + artefact-hash conformance suite** rather than a code-coverage test plan.

Correctness is measured by:

1. **Artefact presence** — every path in `pitch_demo.md §2.1` exists under `DRIFTCALL/pitch_assets/` at freeze time.
2. **Artefact integrity** — every file's sha256 matches the pin in `pitch_assets/manifest.yaml` (`pitch_demo.md §4`).
3. **Script timing** — every beat cue in the verbatim script (§3.1) resolves to a machine-parseable `MM:SS` boundary that sums to exactly 180 s.
4. **Content invariants** — the canonical English gloss `"Friday I need to go to Bangalore, under ₹8000, after 6pm"` is **byte-identical** across all 5 locations that must quote it (script Beat 1, video Scene 1, blog Section 1, manifest `audio.hindi_brief.transcript_en`, video Shot 5.1 caption). Any drift is a Sev-1 failure.
5. **Caption obligation** — every Indic utterance in the video, deck, and blog carries an English caption within ≤ 3 s of audio start (risk-12 mitigation).
6. **Q&A fitness** — all 8 entries in `qa_prep.md` have answer text that fits inside 20 s at 150 wpm (≤ 50 spoken words).
7. **Blog structural conformance** — exactly 5 sections, 6 code links, all URLs parse and resolve via HEAD request, 550 ± 50 words.
8. **Deck structural conformance** — exactly 5 slides, 1920 × 1080 page size, ≤ 50 MB, PDF exports cleanly via `pypdf`.
9. **Video structural conformance** — scene table sums to ≤ 120 s, SRT parses, every scene has a matching SRT cue, burned-in caption per every Indic utterance.
10. **Placeholder hygiene** — no `<team>`, `TBD`, `TODO`, `pattern X`, `XXX` tokens in published artefacts (unpublished source markdown may retain `<team>` per `pitch_demo.md §9 Open Question 1`).
11. **Fallback paths rehearsable** — every error-mode clause in `pitch_demo.md §5` (5.1–5.9) resolves to a concrete on-disk fallback asset that passes its own integrity check.
12. **Deliverables checklist coverage** — 100% of the 10 items enumerated in DESIGN.md §13 are touched by at least one assertion in this plan.

**Out of scope:** subjective judging of the script's prose quality (enforced by human critic rounds, not pytest); real HF-blog-publishing API calls (stubbed — we only validate the source markdown + frontmatter); real YouTube upload (stubbed — we only validate the mp4 + srt); real projector-resolution negotiation (mocked); live mic-feedback recovery (rehearsal-only, not CI-testable); live judge-interruption handling (rehearsal-only); actual wall-clock stopwatch rehearsal (covered by `pitch.run_rehearsal()` integration test §3.1 with tolerance ±5 s per beat, not a human stopwatch).

**No actual video rendering.** All video-metadata assertions use `ffmpeg -i <file> -f null -` in dry-probe mode (reads headers only), `pyav.open(...).streams.video[0]` for codec/duration/resolution, and `pysrt.open(...)` for the caption track. The `.mp4` and `.wav` files must already exist on disk at freeze time; this test plan does not synthesize media.

**Fixtures contract.** Pinned-sha256 audio/video fixtures (`hindi_brief_wav_pinned_sha256`, `before_trained_mp4_pinned_sha256`, `after_trained_mp4_pinned_sha256`, `final_deck_pdf_pinned_sha256`) are **locally scoped** to `tests/test_pitch_demo.py`. The `probe_report_no_exploits` fixture is **defined in `evaluation_tests.md §5.4`** and imported here and by `risk_book_tests.md` — see `evaluation_tests.md §5.6` cross-plan sharing contract. It is **not** consumed by `training_tests.md` (which tests the probe entry-point via direct mocks in U45, not via a pre-baked `ProbeReport`). Any change to the shared fixture body MUST be mirrored in `tests/conftest.py` in lockstep.

Every test below cites the clause in `docs/modules/pitch_demo.md` it covers via `pitch_demo.md §X.Y` in the docstring.

---

## 1. Unit tests

All unit tests live in `DRIFTCALL/tests/test_pitch_demo.py`. Fixtures (§5) come from `tests/conftest.py` for shared items and from a local `tests/fixtures/pitch_demo/` directory for artefact-pin fixtures. Import lines under test:

```python
import hashlib
from pathlib import Path

import mdformat
import pysrt
import yaml
from mutagen.wave import WAVE
from pypdf import PdfReader

from driftcall.pitch import (
    load_manifest,
    parse_beat_cues,
    spoken_word_count,
    extract_canonical_gloss_sites,
    validate_no_placeholders,
    validate_blog_sections,
    validate_deck_slides,
    validate_video_scenes,
    validate_srt_alignment,
    validate_qa_answer_length,
    run_rehearsal,
)
```

The `driftcall.pitch` helper module is a thin (~200 LOC) validator package — it reads the markdown/yaml/srt/pdf/wav artefacts and returns structured reports. It does **not** mutate artefacts. Its own test coverage is a side-benefit; the primary purpose of this test plan is artefact validation, not `pitch.py` coverage.

Required unit test count: **≥ 20**. Delivered: **28**.

### 1.1 Script-timing parse invariants

**Test U1 — `test_pitch_3min_beat_cues_are_parseable`**
Loads `pitch_assets/script/pitch_3min.md`, runs `parse_beat_cues(md_text)`, expects a list of 5 `BeatCue(name, start_s, end_s)` tuples exactly matching: `Hook(0, 20)`, `Architecture(20, 60)`, `Curves(60, 120)`, `BeforeAfter(120, 160)`, `Close(160, 180)`. Any deviation fails. Cites `pitch_demo.md §3.1`.

**Test U2 — `test_pitch_3min_total_duration_is_exactly_180s`**
Sums `cue.end_s - cue.start_s` across the 5 beats; asserts `== 180`. Cites `pitch_demo.md §3.1` final line "20 + 40 + 60 + 40 + 20 = 180".

**Test U3 — `test_pitch_3min_beats_are_non_overlapping_and_contiguous`**
For each adjacent pair, asserts `beat[i].end_s == beat[i+1].start_s`. No gaps, no overlaps. Cites `pitch_demo.md §3.1`.

**Test U4 — `test_pitch_3min_audio_cues_fit_inside_their_beat`**
For every audio cue inside a beat (e.g., `hindi_brief.wav @ 0:02 + 4s`, `trained_reply_hinglish.wav @ 2:20 + 6s`), asserts `beat.start_s ≤ audio_start_s < audio_start_s + audio_duration_s ≤ beat.end_s`. Cites `pitch_demo.md §3.1` Beats 1 and 4.

### 1.2 Q&A answer-length invariants (8 entries)

**Test U5 — `test_qa_prep_has_exactly_8_entries`**
Parses `pitch_assets/script/qa_prep.md` markdown table; asserts `len(qa_entries) == 8`. Cites `pitch_demo.md §3.2` header "8 questions".

**Test U6 — `test_qa_prep_questions_1_5_cite_design_md`**
For entries 1–5, asserts each Q row contains the literal token `DESIGN.md §15`. Cites `pitch_demo.md §3.2`.

**Test U7 — `test_qa_prep_questions_6_8_are_new_and_name_judge`**
For entries 6–8, asserts each Q row contains the literal token `(new —` and a judge-surname match from the DESIGN.md §2.4 panel list. Cites `pitch_demo.md §3.2` entries Q6–Q8.

**Test U8 — `test_qa_prep_every_answer_fits_20s_budget`** (parametrised ×8)
For each of the 8 answers, runs `spoken_word_count(answer_text)` and asserts `word_count ≤ 50` (150 wpm × 20 s / 60 = 50 words). Cites `pitch_demo.md §3.2` "≤ 20 s spoken".

**Test U9 — `test_qa_prep_every_answer_has_proof_link`**
For each entry, asserts the Proof-link column is non-empty and parses as either a `docs/modules/*.md §N.M` reference or a resolvable URL. Cites `pitch_demo.md §3.2` format spec.

### 1.3 Blog-outline invariants

**Test U10 — `test_blog_post_has_exactly_5_sections`**
Parses `pitch_assets/blog/post.md`, counts `^## ` H2 headers; asserts `== 5` with labels matching §3.4's ordered list. Cites `pitch_demo.md §3.4`.

**Test U11 — `test_blog_post_has_6_code_links_that_parse`**
Extracts URLs from the bullet list in Section 5 + inline links; asserts exactly 6 URLs: env Space, demo Space, model, dataset, Colab, GitHub. Each URL parses via `urllib.parse.urlparse` with non-empty `scheme` and `netloc`. Cites `pitch_demo.md §3.4` "Code links embedded in prose".

**Test U12 — `test_blog_post_urls_resolve_via_head_request`** (network-gated, skipped in offline CI via `@pytest.mark.network`)
HEAD-requests each of the 6 URLs; asserts HTTP status in `{200, 301, 302, 401}` (401 allowed for unpublished private HF repos pre-freeze). Cites `pitch_demo.md §3.4` "verified to resolve at publish time".

**Test U13 — `test_blog_post_word_count_within_tolerance`**
Strips markdown, counts whitespace-separated tokens; asserts `500 ≤ words ≤ 600` (550 ± 50 per §3.4 "~550 words"). Cites `pitch_demo.md §3.4`.

**Test U14 — `test_blog_post_lints_clean_via_mdformat`**
Runs `mdformat.text(blog_md, options={"number": True, "wrap": 100})` and asserts the diff is empty (file is already `mdformat`-canonical). Cites `pitch_demo.md §3.4` "Audio-embed rule" and frontmatter spec.

### 1.4 Video-script invariants

**Test U15 — `test_youtube_script_scene_count_times_duration_under_2min`**
Parses `pitch_assets/video/driftcall_demo_script.md`, extracts scene table, sums `(scene.end_s - scene.start_s)`; asserts `total ≤ 120` s **and** `total == 115` (the §3.3 target of 1:55). Cites `pitch_demo.md §3.3`.

**Test U16 — `test_youtube_script_has_exactly_6_scenes`**
Asserts 6 rows in the scene table with labels matching §3.3's list (Cold open, Problem statement, Environment tour, Training curves, Before/After, Close). Cites `pitch_demo.md §3.3`.

**Test U17 — `test_youtube_script_scenes_are_contiguous_with_no_gap`**
For each adjacent scene pair, asserts `scene[i].end_s == scene[i+1].start_s`. Scene 1 starts at 0; scene 6 ends at ≤ 120. Cites `pitch_demo.md §3.3`.

### 1.5 Deck invariants

**Test U18 — `test_pitch_deck_pdf_has_exactly_5_pages`**
Opens `pitch_assets/deck/driftcall_deck.pdf` via `PdfReader`; asserts `len(reader.pages) == 5`. Matches slide labels `01_hook` … `05_close`. Cites `pitch_demo.md §3.5` and §2.1 `deck/slides/` listing.

**Test U19 — `test_pitch_deck_pdf_page_size_is_1920x1080`**
For every page, asserts `page.mediabox.width == 1920 and page.mediabox.height == 1080` (tolerance 1 unit for rounding). Cites `pitch_demo.md §3.5`.

**Test U20 — `test_pitch_deck_slide_png_set_is_complete`**
Asserts the 5 PNG files listed in `pitch_demo.md §2.1` (`01_hook.png` … `05_close.png`) all exist under `pitch_assets/deck/slides/` and each opens via `PIL.Image.open` without error. Cites `pitch_demo.md §2.1`.

### 1.6 Placeholder hygiene

**Test U21 — `test_no_team_placeholder_in_published_artifacts`**
Loads the artefacts marked "published" in `pitch_demo.md §2.2` (blog/post.md, video/*.srt, deck/*.pdf contents via `extract_text`); asserts `<team>` is **not** present (the literal string). Unpublished source markdown (`pitch_3min.md`, `qa_prep.md`, `frontmatter.yaml` before rename) is allowed to retain the placeholder per §9 Open Question 1. Cites `pitch_demo.md §9`.

**Test U22 — `test_no_tbd_todo_pattern_x_in_published_artifacts`** (parametrised over tokens `TBD`, `TODO`, `pattern X`, `XXX`, `FIXME`)
Scans every published artefact's textual content; asserts none of the banned tokens appears (case-insensitive for TBD/TODO/FIXME; case-sensitive for `pattern X` and `XXX`). Cites `pitch_demo.md §1` "No placeholders" + `DRIFTCALL/CLAUDE.md §3.4` "No placeholder language".

### 1.7 Canonical-gloss byte-identity

**Test U23 — `test_canonical_english_gloss_is_byte_identical_at_5_locations`**
Calls `extract_canonical_gloss_sites()` which returns a list of `(path, line, extracted_string)` tuples for the 5 required locations: (1) `script/pitch_3min.md` Beat 1 caption bar, (2) `video/driftcall_demo_script.md` Scene 1 caption cell, (3) `blog/post.md` Section 1 inline caption beside audio embed, (4) `pitch_assets/manifest.yaml` `audio.hindi_brief.transcript_en`, (5) `video/driftcall_demo_script.md` Shot 5.1 burned-in caption. Asserts all 5 extracted strings `== "Friday I need to go to Bangalore, under ₹8000, after 6pm"` with **byte-identity** (`bytes(s, "utf-8") == bytes(expected, "utf-8")`). Also asserts the `₹` codepoint is U+20B9 (not `Rs.`, not `INR`, not `₹` escaped). Cites `pitch_demo.md §3.1`, §3.3, §3.4, §4, §8.3.

### 1.8 Manifest hash invariants

**Test U24 — `test_manifest_yaml_lists_all_5_audio_video_artifacts_with_hashes`**
Loads `pitch_assets/manifest.yaml`, asserts `artifacts` contains at minimum 5 entries with sha256 fields (video.mp4, deck.pdf, audio.hindi_brief, audio.trained_reply, blog.post — the "5 audio/video artefacts" per the task brief, stretched to cover the 5 hash-critical artefacts including deck and blog). For each, asserts `sha256` is a 64-char lowercase hex string (regex `^[0-9a-f]{64}$`). Cites `pitch_demo.md §4`.

**Test U25 — `test_manifest_sha256_matches_on_disk_sha256`** (parametrised ×5)
For each manifest entry, computes `hashlib.sha256(open(path, 'rb').read()).hexdigest()` and asserts equality with the manifest-pinned sha256. Uses pinned fixtures (`hindi_brief_wav_pinned_sha256`, etc.) for cross-check as a defence-in-depth against accidental fixture drift. Cites `pitch_demo.md §4` "hash mismatch at presentation time blocks the release".

**Test U26 — `test_manifest_locked_at_timestamp_is_valid_iso8601_ist`**
Parses `manifest.locked_at` as ISO-8601; asserts offset is `+05:30` (IST) and date component is `2026-04-26` (freeze day). Cites `pitch_demo.md §4` "frozen on 2026-04-26 14:00 IST".

### 1.9 Audio file format invariants

**Test U27 — `test_audio_samples_are_16khz_mono_wav`** (parametrised over the 5 `audio_samples/*.wav` files)
Opens each `.wav` via `mutagen.wave.WAVE`; asserts `info.sample_rate == 16000`, `info.channels == 1`, `info.bits_per_sample in {16, 24}`. Cites `pitch_demo.md §2.1` comments "16 kHz mono".

**Test U28 — `test_audio_durations_match_manifest_within_100ms`**
For `hindi_brief.wav` (manifest says 4.0 s) and `trained_reply_hinglish.wav` (6.0 s), asserts measured duration is within 0.1 s of the manifest value. Cites `pitch_demo.md §4` `duration_s` fields.

---

## 2. Property tests

Required property test count: **≥ 5**. Delivered: **6**. All use `hypothesis` where input-space generation is meaningful; hash-identity properties use deterministic parametric input instead of `hypothesis` (there is no value space to explore — equality is the property).

### 2.1 Property P1 — Canonical gloss is byte-identical at all 5 required sites

```python
@pytest.mark.parametrize("site_idx", range(5))
def test_prop_canonical_gloss_byte_identical(site_idx: int) -> None:
    """pitch_demo.md §3.1 Beat 1, §3.3 Scene 1, §3.4 Section 1, §4 manifest, §8.3 Shot 5.1."""
    sites = extract_canonical_gloss_sites()
    expected = b"Friday I need to go to Bangalore, under \xe2\x82\xb9""8000, after 6pm"
    assert sites[site_idx].extracted.encode("utf-8") == expected
```

Invariant: **all 5 sites emit the same bytes**. Any whitespace, punctuation, or unicode-normalisation drift fails. This is the doc's single load-bearing string — if it drifts, the pitch, video, blog, and manifest desync in a way a judge may notice. Cites `pitch_demo.md §3.1, §3.3, §3.4, §4, §8.3`.

### 2.2 Property P2 — Every Indic clip has English caption within ≤ 3 s of audio start

```python
@given(clip=indic_clip_strategy())  # hypothesis strategy enumerates every Indic audio usage
def test_prop_indic_clip_has_caption_within_3s(clip: IndicClipUsage) -> None:
    """risk-12 mitigation. pitch_demo.md §1 'English captions on every Indic clip'."""
    caption = find_caption_for(clip)
    assert caption is not None, f"no caption for {clip.path} at {clip.context}"
    delta = caption.start_s - clip.audio_start_s
    assert -1.0 <= delta <= 3.0, f"caption {delta:+.2f}s off for {clip.path}"
```

The hypothesis strategy `indic_clip_strategy()` generates every tuple `(artifact, scene/beat, audio_file, audio_start_s)` from the script, video script, and blog post. The property asserts a caption (burned-in or SRT) starts within [-1, +3] s of the audio onset. Cites `pitch_demo.md §3.3` "every single Indic utterance … carries a burned-in English caption on the frame *and* a matching SRT cue".

### 2.3 Property P3 — Beat durations sum to 180 under arbitrary re-parses

```python
@given(text_perturbation=st.sampled_from(["lf", "crlf", "trailing_ws", "bom"]))
def test_prop_beat_sum_invariant_under_text_normalisation(text_perturbation: str) -> None:
    """Beat cues parse identically across line-ending / whitespace / BOM variations."""
    original = Path("pitch_assets/script/pitch_3min.md").read_bytes()
    perturbed = apply_perturbation(original, text_perturbation)
    beats = parse_beat_cues(perturbed.decode("utf-8-sig"))
    assert sum(b.end_s - b.start_s for b in beats) == 180
```

Invariant: parser is robust to line-ending and BOM drift across editors — a git-checkout on Windows must not change the parsed timing. Cites `pitch_demo.md §3.1` strict 180 s.

### 2.4 Property P4 — Every Q&A answer stays under 50 words regardless of whitespace

```python
@given(entry_idx=st.integers(0, 7), ws_perturbation=st.sampled_from(["double_space", "tab", "nbsp"]))
def test_prop_qa_answer_under_50_words(entry_idx: int, ws_perturbation: str) -> None:
    """pitch_demo.md §3.2 ≤ 20 s at 150 wpm = 50 words."""
    entry = load_qa_entries()[entry_idx]
    perturbed = apply_ws_perturbation(entry.answer, ws_perturbation)
    assert spoken_word_count(perturbed) <= 50
```

Invariant: word-count tokeniser normalises exotic whitespace (double-space, tab, NBSP from copy-paste) so the count remains ≤ 50. Cites `pitch_demo.md §3.2` 20 s budget.

### 2.5 Property P5 — Manifest sha256 is a function of file bytes (purity)

```python
@given(artifact_id=st.sampled_from(MANIFEST_ARTIFACT_IDS))
def test_prop_manifest_sha256_matches_on_disk_bytes(artifact_id: str) -> None:
    """pitch_demo.md §4 'any change after that point is a critic-reviewed deviation'."""
    entry = load_manifest().by_id(artifact_id)
    measured = hashlib.sha256(Path(entry.path).read_bytes()).hexdigest()
    assert measured == entry.sha256
```

Invariant: the manifest is a pure function of the on-disk bytes. Any drift is a freeze violation. Cites `pitch_demo.md §4`.

### 2.6 Property P6 — Every published artefact is placeholder-free

```python
@given(artefact=st.sampled_from(PUBLISHED_ARTEFACTS), token=st.sampled_from(["<team>", "TBD", "TODO", "pattern X", "XXX", "FIXME"]))
def test_prop_published_artefacts_have_no_placeholders(artefact: Path, token: str) -> None:
    """pitch_demo.md §1 'No placeholders' + DRIFTCALL/CLAUDE.md §3.4."""
    text = extract_text(artefact)
    assert token not in text, f"{token!r} found in {artefact}"
```

Invariant: no banned token appears in any published artefact, for any (artefact, token) pair. Cites `pitch_demo.md §1`.

---

## 3. Integration tests

Integration tests live in `DRIFTCALL/tests/test_pitch_demo_integration.py`.

### 3.1 `pitch.run_rehearsal()` — full timing simulator with ±5 s per-beat tolerance

**Test I1 — `test_run_rehearsal_returns_timing_report_within_tolerance`**
Invokes `run_rehearsal(script_path="pitch_assets/script/pitch_3min.md", wpm=150, audio_cues_from_manifest=True)` which simulates delivery by:
1. tokenising each beat's spoken lines;
2. computing spoken time = `words / (wpm / 60)`;
3. adding audio-cue durations from the manifest for beats that play audio (Beat 1: +4 s, Beat 4: +4 s + 6 s);
4. returning a `TimingReport(beats: list[BeatActual(name, target_s, simulated_s, delta_s)], total_s: float)`.

Asserts `all(abs(beat.delta_s) ≤ 5 for beat in report.beats)` and `abs(report.total_s - 180) ≤ 5`. Cites `pitch_demo.md §3.1` and §8.1 rehearsal transcript.

**Test I2 — `test_run_rehearsal_flags_overrun_beyond_5s`**
Feeds a deliberately padded script (adds 30 extra words to Beat 3); asserts `report.beats[2].delta_s > 5` and `report.total_s > 185`. Confirms the simulator flags over-runs — this is the CI signal for drafts that grow beyond the 3-min ceiling. Cites `pitch_demo.md §5.9`.

**Test I3 — `test_run_rehearsal_cuts_to_close_at_240s`**
With `run_rehearsal(..., cut_to_close_at_s=160)` enabled, confirms that when Beat 4 rehearsal at 2:40 still has unread text, the simulator truncates Beat 4 and emits the compressed close from §5.9 instead — then asserts `total_s ≤ 180`. Rehearses the fallback. Cites `pitch_demo.md §5.9`.

### 3.2 Video SRT file validates end-to-end

**Test I4 — `test_driftcall_demo_srt_parses_via_pysrt`**
Runs `pysrt.open("pitch_assets/video/driftcall_demo.srt")`; asserts at least 6 cues (one per scene), all cues have well-formed `start` and `end` times, no cue overlaps the next, last cue's `end` ≤ `video.duration_s` from manifest. Cites `pitch_demo.md §3.3`.

**Test I5 — `test_srt_cues_cover_every_indic_utterance`**
For each Indic audio usage listed in §3.3 (Scenes 1 and 5), asserts an SRT cue exists whose time range contains the audio's time range and whose text contains the canonical English gloss (exact for Scene 1, scene-specific for Scene 5's trained reply). Cites `pitch_demo.md §3.3` caption rule.

**Test I6 — `test_srt_and_burned_in_captions_agree_on_text`**
Parses burned-in captions from the video-script markdown's `Caption` cells; asserts each burned-in caption text equals (or is a substring of) the corresponding SRT cue text. Prevents the SRT and the on-frame text from disagreeing. Cites `pitch_demo.md §3.3`.

### 3.3 Blog markdown lints clean end-to-end

**Test I7 — `test_blog_post_lints_via_mdformat_and_frontmatter_loads`**
Runs `mdformat` on `pitch_assets/blog/post.md` with the project's mdformat config; asserts output == input (file is canonical). Then loads `pitch_assets/blog/frontmatter.yaml` via `yaml.safe_load`; asserts keys `{author, tags, thumbnail}` present, `tags` is a list of 6 strings matching §3.4, `thumbnail == "slides/04_before_after.png"`. Cites `pitch_demo.md §3.4`.

**Test I8 — `test_blog_audio_embeds_use_relative_paths`**
Scans `post.md` for `<audio>` tags; asserts every `src` attribute starts with `./post_assets/` (relative), not `https://` (absolute). Cites `pitch_demo.md §3.4` "Audio-embed rule".

### 3.4 Deck exports PDF within slide-count bound (dry run)

**Test I9 — `test_deck_pdf_exports_within_slide_count_and_size_bound`**
Runs a dry export check: opens `driftcall_deck.pdf` via `PdfReader`, asserts page count == 5, asserts file size ≤ 25 MB (generous ceiling under the ≤ 50 MB video requirement; the deck is typically < 5 MB but we do not re-render). Asserts every page passes `reader.pages[i].extract_text()` with at least 10 non-whitespace characters (rules out blank/corrupt pages). Cites `pitch_demo.md §3.5`.

**Test I10 — `test_deck_pdf_has_no_embedded_javascript`**
Scans the PDF bytes for `/JS` or `/JavaScript` action objects via `pypdf`; asserts none present. Security hygiene for the projector laptop. Cites `pitch_demo.md §3.5` static-PDF design choice.

### 3.5 End-to-end deliverable bundle validation

**Test I11 — `test_full_pitch_asset_bundle_passes_all_validators`**
Top-level integration: calls `validate_pitch_bundle("pitch_assets/")` which runs, in order: manifest load → artefact presence check → sha256 match → script parse → Q&A parse → blog parse → video parse → deck parse → canonical-gloss cross-check → placeholder scan → caption-coverage scan. Returns a `BundleReport` with `passed: bool` and `failures: list[str]`. Asserts `report.passed is True`. This is the release-gate CI check referenced in `pitch_demo.md §4`. Cites `pitch_demo.md §2.1` and §4.

**Test I12 — `test_probe_report_badge_embeds_in_blog`**
Uses the shared `probe_report_no_exploits` fixture (from `evaluation_tests.md §5.4`); asserts that when `validate_pitch_bundle(..., probe_report=probe_report_no_exploits)` is called, the blog's Section 2 (or an addendum) contains a "reward-hacking probe: 0 exploits on 200 held-out episodes" claim consistent with the fixture's `n_episodes=200, total_hits=0`. If the number in the blog disagrees with the fixture, the test fails. Cites `pitch_demo.md §3.2` Q2 proof-link + `rewards.md §7` probe report.

---

## 4. Coverage target

**N/A — artefact-only module, no code coverage.**

Per `pitch_demo.md §2`: "There is no Python module imported by other modules; everything here is markdown, media files, and slide files." The `driftcall.pitch` validator helper (~200 LOC) is exercised to ≥ 85% line coverage as a side-effect of §1–§3 tests, but this is **not** the target metric.

The target metric is **deliverables-checklist coverage**: every item in DESIGN.md §13's deliverable list (and `pitch_demo.md §1`'s four sole-ownership items) must be touched by at least one assertion.

### 4.1 DESIGN.md §13 deliverables — assertion coverage matrix

| # | Deliverable (DESIGN.md §13 + pitch_demo.md §1) | Covered by |
|---|---|---|
| D-1 | **3-minute live pitch** with verbatim script | U1, U2, U3, U4, P3, I1, I2, I3 |
| D-2 | **Q&A prep** (8 questions, ≤ 20 s answers) | U5, U6, U7, U8, U9, P4 |
| D-3 | **Pitch deck** (≤ 5 slides, 1920×1080, PDF) | U18, U19, U20, I9, I10 |
| D-4 | **YouTube video** (< 2 min, H.264, ≤ 50 MB) | U15, U16, U17, I4, I5, I6, I11 |
| D-5 | **SRT captions** (English, every Indic clip) | I4, I5, I6, P2 |
| D-6 | **HF blog post** (< 2-min read, 5 sections, 6 links) | U10, U11, U12, U13, U14, I7, I8 |
| D-7 | **Audio samples** (5 `.wav` files, 16 kHz mono) | U27, U28 |
| D-8 | **Manifest** (`pitch_assets/manifest.yaml` with sha256 pins) | U24, U25, U26, P5 |
| D-9 | **Canonical English gloss** byte-identical across sites | U23, P1 |
| D-10 | **Reward-hacking probe report** reference (cross-artefact) | I12 |

**Target:** 10 / 10 = **100% deliverables coverage** (no deliverable without an assertion).

### 4.2 Artefact-file coverage matrix

Every file path listed in `pitch_demo.md §2.1` under `DRIFTCALL/pitch_assets/` must be touched by at least one presence or integrity assertion:

| Path | Assertion |
|---|---|
| `script/pitch_3min.md` | U1, U2, U3, U4, P3 |
| `script/qa_prep.md` | U5, U6, U7, U8, U9 |
| `script/presenter_cheatsheet.md` | I11 (bundle presence) |
| `deck/driftcall_deck.pdf` | U18, U19, I9, I10 |
| `deck/driftcall_deck.pptx` | I11 (presence only) |
| `deck/slides/*.png` (×5) | U20 |
| `video/driftcall_demo.mp4` | U24, U25, I11 |
| `video/driftcall_demo.srt` | I4, I5, I6 |
| `video/driftcall_demo_script.md` | U15, U16, U17 |
| `video/broll/*.mov` (×4) | I11 (presence only — no render check) |
| `blog/post.md` | U10–U14, I7, I8 |
| `blog/post_assets/*.wav` (×2) | U27 |
| `blog/post_assets/fig_reward_curves.png` | I11 (presence only) |
| `blog/frontmatter.yaml` | I7 |
| `audio_samples/*.wav` (×5) | U27, U28 |
| `manifest.yaml` | U24, U25, U26, P5 |

**Target:** 100% artefact-path coverage. If a new file is added to `pitch_demo.md §2.1`, this plan MUST add a corresponding assertion before the PR is accepted.

---

## 5. Fixtures

All fixtures live in `DRIFTCALL/tests/conftest.py` unless noted. Pinned-hash fixtures return `bytes` or `str` constants — the actual artefact files are expected on disk; the fixture only holds the expected sha256 for cross-checking against the manifest.

### 5.1 `hindi_brief_wav_pinned_sha256` (local)

**Shape:** `str` — the 64-char lowercase hex sha256 of the canonical `audio_samples/hindi_brief.wav` (16 kHz mono, 4.0 s, Kokoro-generated Hindi TTS of "Bhai Friday ko Bangalore jaana hai, 8000 rupees max, 6pm ke baad"). Pinned at freeze time 2026-04-26 14:00 IST.

**Scope:** `tests/test_pitch_demo.py`. Not shared with other plans.

**Generated:** The `.wav` file is produced by `audio.md §2.1`'s Kokoro Indic voicepack, committed under `pitch_assets/audio_samples/`. The sha256 is computed once and pasted into `tests/fixtures/pitch_demo/hashes.py`. Manifest (`manifest.yaml`) must mirror this hash — any drift is a Sev-1 CI failure.

**Used by:** U25, U27, U28, P5, I11.

### 5.2 `before_trained_mp4_pinned_sha256` (local)

**Shape:** `str` — sha256 of `video/broll/base_model_keyerror.mov` (the untrained-model crash B-roll played in video Scene 2 and pitch Beat 4 left column). ~15 s @ 1920×1080 H.264, < 10 MB.

**Scope:** `tests/test_pitch_demo.py`.

**Generated:** Screen-recorded once against the base (untrained) Gemma 4 E2B running against a drift-triggered episode; saved to `pitch_assets/video/broll/`. Sha256 computed at freeze.

**Used by:** U25, I11.

### 5.3 `after_trained_mp4_pinned_sha256` (local)

**Shape:** `str` — sha256 of `video/broll/trained_model_adapts.mov` (the trained-LoRA adaptation B-roll played in video Scene 5 and pitch Beat 4 right column). ~20 s @ 1920×1080 H.264, < 15 MB.

**Scope:** `tests/test_pitch_demo.py`.

**Generated:** Screen-recorded against the trained-LoRA Demo Space with manual drift-toggle firing at turn 4; saved to `pitch_assets/video/broll/`. Sha256 computed at freeze.

**Used by:** U25, I11.

### 5.4 `final_deck_pdf_pinned_sha256` (local)

**Shape:** `str` — sha256 of `deck/driftcall_deck.pdf` (5 slides, 1920×1080, exported from Keynote/PPT). Pinned at freeze.

**Scope:** `tests/test_pitch_demo.py`.

**Generated:** Deck authored in Keynote, exported to PDF, committed. Sha256 computed at freeze.

**Used by:** U25, I9, I10, I11.

### 5.5 `probe_report_no_exploits` (shared)

**Source:** `evaluation_tests.md §5.4` — **consumed as-is, not re-defined here.**

**Shape:** `ProbeReport` with `n_episodes=200`, `per_class` containing all 5 known exploit classes at `count=0`, `rate=0.0`, `example_episode_id=None`, `total_hits=0`, `novel_classes=()`.

**Scope:** shared across `evaluation_tests.md` (definer) and `pitch_demo_tests.md` only, per `evaluation_tests.md §5.6` cross-plan contract. **Not** consumed by `training_tests.md` or `risk_book_tests.md`. Fixture body lives in `tests/conftest.py`; `tests/conftest_lock.py` sha256-locks it to prevent silent drift.

**Used by:** I12 (probe-artefact badge in blog). `pitch_demo_tests.md` consumes but does not mutate; if this plan needs a mutated variant, it MUST define a derived fixture `probe_report_no_exploits_for_pitch(probe_report_no_exploits)` per the §5.6 contract.

### 5.6 Fixture integrity cross-check

A `conftest_lock.py` check runs at collection time and sha256-verifies each pinned fixture's `.wav` / `.mp4` / `.pdf` against its hash constant. If a file drifts without a corresponding hash update, pytest fails collection with `FixtureIntegrityError: <path> hash drift; update tests/fixtures/pitch_demo/hashes.py and manifest.yaml together`. This catches the common failure mode of re-exporting the deck without re-running the hash-freeze script.

---

## 6. Summary

**Counts delivered:**

- Unit tests: **28** (target ≥ 20) ✓
- Property tests: **6** (target ≥ 5) ✓
- Integration tests: **12** (target: rehearsal simulator ±5 s per beat + SRT validate + blog lint + deck PDF within slide-count bound — all delivered via I1–I12) ✓
- Coverage: **N/A** (artefact-only); deliverables-checklist coverage = **10 / 10 = 100%** ✓
- Fixtures: **5** (4 local hash-pinned + 1 shared `probe_report_no_exploits` per `evaluation_tests.md §5.6`) ✓

**Tooling lock:** pytest ≥ 8.0, hypothesis ≥ 6.100, mdformat ≥ 0.7, pypdf ≥ 4.0, pysrt ≥ 1.1, mutagen ≥ 1.47. No actual video rendering in any test; all media checks are header/metadata dry-probes.

**Cross-plan links:**
- `evaluation_tests.md §5.4, §5.6` — shared `probe_report_no_exploits` fixture.
- `training_tests.md` — no shared fixtures (training does not consume `probe_report_no_exploits`; its probe entry-point is tested via direct mocks per `training_tests.md` U45). Cross-check is documentary only: if a pitch asset quotes a specific training-curve number, manual consistency with `training_tests.md §3.4` applies.
- `risk_book_tests.md` — documentary consistency only — pitch's blog badge (I12) should agree with risk-book's zero-exploit baseline; NOT a shared pytest fixture.
- `deploy_demo_space_tests.md` — Beat 4's `broll/trained_model_adapts.mov` is produced against the Demo Space; any Demo Space behaviour change that alters the B-roll invalidates `after_trained_mp4_pinned_sha256` and forces a re-freeze.

**Open questions:**
1. `pitch_demo.md §9 Open Question 1` — `<team>` placeholder. Once locked, U21 stops being a tripwire and becomes a strict check. Until then, U21 treats unpublished sources leniently (see §1.6).
2. `pitch_demo.md §9 Open Question 2` — evaluation number drift (≥ 2 pp). If `evaluation.md` final numbers replace the DESIGN.md §15 quotes, the blog's Section 3 (and U13's word-count tolerance) must re-freeze.
3. Whether to gate U12 (URL HEAD-request liveness) on a `@pytest.mark.network` marker that is skipped in offline CI. **Recommendation:** skip in offline CI, enforce in the release-gate run triggered at 13:30 IST on pitch day (30 min before publish). Orchestrator to confirm.
