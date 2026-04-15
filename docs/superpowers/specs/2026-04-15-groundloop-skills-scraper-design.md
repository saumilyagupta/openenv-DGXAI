# GroundLoop Skills-Scraper — Design Spec

**Date:** 2026-04-15
**Sub-project:** #2 of 8 in the GroundLoop MCP decomposition
**Status:** Design — awaiting user approval
**Depends on:** Nothing (dependency-free, first vertical slice)

---

## 1. Purpose

Scrape every installed Claude Code `SKILL.md` file on disk into a structured, section-level corpus that downstream GroundLoop components (kb-indexer, ralph-orchestrator) can query as the **Layer B "reasoning policy" knowledge base**.

The corpus answers questions like "how should I write tests?" or "what's the security checklist?" by surfacing the matching skill section — not the whole skill file.

## 2. Scope

**In scope:**

- Walk 4 source directories on the local filesystem.
- Parse YAML frontmatter + markdown body of each `SKILL.md`.
- Split the body into section-level nodes (H2/H3 boundaries).
- Emit a deterministic JSONL corpus + manifest.
- Provide a CLI entrypoint.

**Out of scope (for this sub-project):**

- Indexing (BM25/vector) — handled by kb-indexer (#3).
- Live file-watching / hot-reload — Phase 2.
- Remote skill fetching (marketplaces over network) — only local disk files.
- Skill deduplication beyond exact-body match — no fuzzy semantic dedup.

## 3. Architecture

```
groundloop/
  skills_scraper/
    __init__.py
    cli.py          # `python -m groundloop.skills_scraper`
    discovery.py    # Walk source dirs, yield SKILL.md paths
    parser.py       # Parse frontmatter + body
    chunker.py      # Split body into section nodes
    tagger.py       # Deterministic tag inference
    models.py       # Pydantic v2 schemas
    writer.py       # JSONL + manifest emission
    config.py       # Source-root paths, defaults
tests/groundloop/skills_scraper/
  fixtures/fake_skills/
  test_discovery.py
  test_parser.py
  test_chunker.py
  test_tagger.py
  test_writer.py
  test_e2e.py
```

Small focused files, each with one responsibility. Nothing exceeds ~200 lines.

## 4. Component Contracts

### 4.1 `discovery.py`

```python
def walk_sources(sources: list[SourceRoot]) -> Iterator[tuple[Path, SourceRoot]]:
    """Yield (skill_md_path, source_root) for every SKILL.md found."""
```

- Each `SourceRoot` has a label and a glob pattern (e.g. `"~/.claude/skills/*/SKILL.md"`).
- Expands `~`. Skips unreadable / non-regular files with WARN log.

### 4.2 `parser.py`

```python
def parse_skill(path: Path) -> ParsedSkill:
    """Return frontmatter dict + markdown body. Raises ParseError on malformed YAML."""
```

- Uses `python-frontmatter`.
- Missing frontmatter → empty dict, full file is body.
- Returns `ParsedSkill(frontmatter: dict, body: str, mtime: float)`.

### 4.3 `chunker.py`

```python
def chunk_body(body: str) -> list[SectionChunk]:
    """Split markdown into section chunks on H2/H3. H4+ folds into parent H3."""
```

- Uses `markdown-it-py` to parse the AST.
- Each `SectionChunk` has `section_path: list[str]` (root-to-leaf heading titles), `section_body: str`.
- Body shorter than 80 chars → merged with the next chunk to avoid pure-heading noise.
- If the file has no H2/H3 at all, emit a single chunk with `section_path=[skill_name]` and the whole body.

### 4.4 `tagger.py`

```python
def infer_tags(skill_name: str, section_title: str, body: str) -> list[str]:
    """Deterministic rule-based tags. Returns ["domain:python", "phase:test", ...]."""
```

Tag rules (hard-coded, auditable):

- **Domain** (first match wins): `python`, `javascript`, `go`, `kotlin`, `security`, `frontend`, `backend`, `data`, `api`, `mcp`, `devops`, `general`.
- **Phase** (multi-label): `plan`, `build`, `test`, `review`, `deploy`, `debug`, `docs`.
- Rules match `skill_name` first, then `section_title` keywords, then body n-grams.

### 4.5 `models.py` (Pydantic v2, frozen)

```python
class SkillNode(BaseModel, frozen=True):
    id: str                      # sha256(source_path + "#" + "/".join(section_path))[:16]
    skill_name: str
    skill_description: str
    skill_type: Literal["flexible", "rigid"] | None
    section_path: tuple[str, ...]
    section_title: str
    section_body: str
    source_path: str
    source_root: str
    tags: tuple[str, ...]
    trigger_hints: str
    mtime: float
    body_hash: str              # sha256(section_body)
    alias_sources: tuple[str, ...]  # other paths with same body_hash

class ScrapeManifest(BaseModel, frozen=True):
    generated_at: str           # ISO 8601 UTC
    sources: list[SourceRoot]
    scraped_files: int
    skipped_files: int
    total_nodes: int
    errors: list[ScrapeError]
    corpus_sha256: str          # hash of the JSONL file
```

### 4.6 `writer.py`

```python
def write_corpus(nodes: Iterable[SkillNode], out_path: Path) -> Path:
    """Emit JSONL (one node per line, sorted by id for determinism)."""

def write_manifest(manifest: ScrapeManifest, out_path: Path) -> Path:
    """Emit manifest.json (pretty-printed)."""
```

### 4.7 `cli.py`

```
python -m groundloop.skills_scraper \
  [--sources default|<yaml-path>] \
  [--output groundloop/kb/skills_corpus.jsonl] \
  [--verbose]
```

- Default sources defined in `config.py`.
- Exit 0 on success (nodes > 0).
- Exit 1 on hard failure (0 nodes scraped, or I/O error on output).
- Prints summary to stdout: `scraped=N files, skipped=M, nodes=K, errors=E`.

## 5. Data Flow

```
CLI
 └─> discovery.walk_sources()       → [(Path, SourceRoot), ...]
      └─> parser.parse_skill()      → ParsedSkill
           └─> chunker.chunk_body() → [SectionChunk, ...]
                └─> tagger.infer_tags() → [tag, ...]
                     └─> SkillNode(...)
 └─> dedup by body_hash + skill_name + section_path
 └─> writer.write_corpus() + write_manifest()
```

Dedup collapses exact-body duplicates that appear in multiple source roots (e.g., `coding-standards` in both `~/.claude/skills/` and `~/.claude/.agents/skills/`). Kept node records all aliases.

## 6. Error Handling

| Condition | Action |
|---|---|
| Unreadable file | WARN, skip, record in `manifest.errors` |
| Missing frontmatter | Proceed with empty frontmatter |
| Malformed YAML | WARN, skip file, record error |
| Empty body | Skip file |
| Empty chunk (all whitespace) | Drop chunk |
| No H2/H3 in body | Emit single chunk with full body |
| Duplicate `(skill_name, section_path, body_hash)` | Dedupe, append to `alias_sources` |
| Zero total nodes | CLI exit 1 |
| Output path unwritable | CLI exit 1 |

Every error is structured (`ScrapeError(path, stage, reason)`), never silent.

## 7. Testing

### 7.1 Coverage target

**90%** line coverage (higher than project default of 80% — pure data pipeline, cheap to cover, downstream depends on correctness).

### 7.2 Fixtures

`tests/groundloop/skills_scraper/fixtures/fake_skills/` contains:

1. `normal/SKILL.md` — standard frontmatter + 3 H2 sections with nested H3s.
2. `no_frontmatter/SKILL.md` — bare markdown, no YAML header.
3. `malformed_yaml/SKILL.md` — broken frontmatter, should be skipped.
4. `single_section/SKILL.md` — no H2/H3 at all.
5. `dup_a/coding-standards/SKILL.md` + `dup_b/coding-standards/SKILL.md` — identical body, tests dedup.

### 7.3 Test plan

- `test_discovery.py` — finds all 6 fixture files, skips unreadable.
- `test_parser.py` — correct frontmatter extraction, raises on malformed, handles missing.
- `test_chunker.py` — correct section boundaries, H4 folds into H3, single-section fallback, small-body merge.
- `test_tagger.py` — each tag rule has a positive + negative case.
- `test_writer.py` — JSONL deterministic ordering by `id`, manifest shape.
- `test_e2e.py` — run `cli.main` on fixture dir, assert corpus JSONL row count, manifest `total_nodes`, dedup behavior, exit code.

All tests run in <5 seconds total.

## 8. Acceptance Criteria

1. `python -m groundloop.skills_scraper` completes on real system in < 10 seconds.
2. Scrapes all ~91 `SKILL.md` files present on disk.
3. Emits a JSONL corpus where each row parses back into a valid `SkillNode`.
4. Emits a `manifest.json` with non-empty `sources`, `corpus_sha256`, counts.
5. Re-running with unchanged source files produces byte-identical corpus (determinism).
6. Test suite passes with ≥90% line coverage.
7. `ruff check` and `mypy --strict` are clean on `groundloop/skills_scraper/`.

## 9. Non-Goals Restated

- No BM25 or vector indexing (that's kb-indexer #3).
- No hot-reload / file watching.
- No LLM-based tagging.
- No network fetching.

## 10. Risks

| Risk | Mitigation |
|---|---|
| `markdown-it-py` changes AST shape between versions | Pin version in `requirements.txt`; unit tests lock expected chunk output. |
| Some skills use H1 as section boundary instead of H2 | Chunker handles H1, H2, H3 uniformly — H1-as-title is a common variation. |
| Frontmatter schema varies wildly across 91 skills | Only two fields are required downstream (`name`, `description`). All else is opportunistic. |
| Skill files have no YAML frontmatter at all | Tolerated — empty frontmatter, still emit nodes. |
| Dedup collapses legitimately-different skills with identical body (unlikely but possible) | Dedup key includes `skill_name`, not just body. |

## 11. Dependencies (new)

Add to `requirements.txt`:

```
python-frontmatter>=1.0
markdown-it-py>=3.0
```

Already present: `pydantic>=2`, `pytest`, `pytest-cov`, `ruff`, `mypy`.

## 12. Deliverables

1. `groundloop/skills_scraper/` package with all modules above.
2. `tests/groundloop/skills_scraper/` with fixtures + test suite.
3. `groundloop/kb/skills_corpus.jsonl` (generated output, git-ignored).
4. `groundloop/kb/skills_corpus.manifest.json` (generated, git-ignored).
5. Updated `requirements.txt`.
6. Entry in `README.md` explaining how to run the scraper.

---

## Spec Self-Review (done inline)

- ✅ No "TBD" / placeholders.
- ✅ No internal contradictions (checked: granularity = section-level throughout; dedup key consistent between §4 and §6).
- ✅ Scope is single-implementation-plan-sized (~1-2 days of focused TDD work).
- ✅ Ambiguity check: "body shorter than 80 chars" is concrete; "H4+ folds into parent H3" is explicit; tag rules are enumerated.

Ready for user review.
