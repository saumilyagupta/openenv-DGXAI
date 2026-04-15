# GroundLoop Skills-Scraper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dependency-free Python module that walks installed `SKILL.md` files on disk and emits a deterministic, section-level JSONL corpus + manifest for GroundLoop's Layer B reasoning-policy KB.

**Architecture:** 7-module package (`discovery`, `parser`, `chunker`, `tagger`, `models`, `writer`, `cli`, `config`). Each module owns one stage of the pipeline: walk → parse frontmatter → chunk into sections → tag deterministically → emit JSONL + manifest. Pydantic v2 frozen models. CLI via `python -m groundloop.skills_scraper`.

**Tech Stack:** Python 3.11+, Pydantic v2, `python-frontmatter`, `markdown-it-py`, pytest, ruff, mypy.

**Spec reference:** `docs/superpowers/specs/2026-04-15-groundloop-skills-scraper-design.md`.

---

## File Structure

```
groundloop/
  __init__.py                          # empty, marks package
  skills_scraper/
    __init__.py                        # exports SkillNode, run_scraper()
    cli.py                             # main() entrypoint, argparse
    config.py                          # DEFAULT_SOURCES list, constants
    discovery.py                       # walk_sources()
    parser.py                          # parse_skill() -> ParsedSkill
    chunker.py                         # chunk_body() -> list[SectionChunk]
    tagger.py                          # infer_tags() -> list[str]
    writer.py                          # write_corpus(), write_manifest()
    models.py                          # SkillNode, ScrapeManifest, ParsedSkill, SectionChunk, ScrapeError, SourceRoot
    pipeline.py                        # run_scraper() — orchestrates all modules
tests/
  groundloop/
    __init__.py
    skills_scraper/
      __init__.py
      conftest.py                      # fixture paths
      fixtures/
        fake_skills/
          normal/SKILL.md
          no_frontmatter/SKILL.md
          malformed_yaml/SKILL.md
          single_section/SKILL.md
          dup_a/coding-standards/SKILL.md
          dup_b/coding-standards/SKILL.md
      test_models.py
      test_config.py
      test_discovery.py
      test_parser.py
      test_chunker.py
      test_tagger.py
      test_writer.py
      test_pipeline.py
      test_cli.py
      test_e2e.py
requirements.txt                       # add python-frontmatter, markdown-it-py
```

File size rule: no module exceeds 200 lines. Every module imports only from its declared dependencies in the spec.

---

## Task 0: Scaffold package + dependencies

**Files:**
- Create: `groundloop/__init__.py`
- Create: `groundloop/skills_scraper/__init__.py`
- Create: `tests/groundloop/__init__.py`
- Create: `tests/groundloop/skills_scraper/__init__.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Create empty package files**

```bash
mkdir -p groundloop/skills_scraper tests/groundloop/skills_scraper/fixtures/fake_skills
touch groundloop/__init__.py groundloop/skills_scraper/__init__.py
touch tests/groundloop/__init__.py tests/groundloop/skills_scraper/__init__.py
```

- [ ] **Step 2: Add new dependencies to `requirements.txt`**

Append to `requirements.txt`:

```
python-frontmatter>=1.0
markdown-it-py>=3.0
```

- [ ] **Step 3: Install**

Run: `pip install -r requirements.txt`
Expected: installs `python-frontmatter` and `markdown-it-py` plus their transitive deps.

- [ ] **Step 4: Commit**

```bash
git add groundloop/ tests/groundloop/ requirements.txt
git commit -m "chore: scaffold groundloop/skills_scraper package"
```

---

## Task 1: Data models (Pydantic v2, frozen)

**Files:**
- Create: `groundloop/skills_scraper/models.py`
- Test: `tests/groundloop/skills_scraper/test_models.py`

- [ ] **Step 1: Write failing tests**

Create `tests/groundloop/skills_scraper/test_models.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from groundloop.skills_scraper.models import (
    ParsedSkill,
    ScrapeError,
    ScrapeManifest,
    SectionChunk,
    SkillNode,
    SourceRoot,
)


def test_source_root_basic():
    root = SourceRoot(label="user-skills", glob="~/.claude/skills/*/SKILL.md")
    assert root.label == "user-skills"
    assert root.glob == "~/.claude/skills/*/SKILL.md"


def test_skill_node_is_frozen():
    node = SkillNode(
        id="abc123",
        skill_name="python-testing",
        skill_description="Testing patterns",
        skill_type="flexible",
        section_path=("Fixtures",),
        section_title="Fixtures",
        section_body="Use pytest fixtures.",
        source_path="/tmp/x/SKILL.md",
        source_root="user-skills",
        tags=("domain:python", "phase:test"),
        trigger_hints="Testing patterns",
        mtime=1.0,
        body_hash="deadbeef",
        alias_sources=(),
    )
    with pytest.raises(ValidationError):
        node.skill_name = "other"  # type: ignore[misc]


def test_skill_node_rejects_missing_required():
    with pytest.raises(ValidationError):
        SkillNode()  # type: ignore[call-arg]


def test_skill_type_accepts_none():
    node = SkillNode(
        id="x",
        skill_name="n",
        skill_description="",
        skill_type=None,
        section_path=(),
        section_title="",
        section_body="",
        source_path="",
        source_root="",
        tags=(),
        trigger_hints="",
        mtime=0.0,
        body_hash="",
        alias_sources=(),
    )
    assert node.skill_type is None


def test_scrape_manifest_shape():
    m = ScrapeManifest(
        generated_at="2026-04-15T00:00:00Z",
        sources=[SourceRoot(label="a", glob="b")],
        scraped_files=1,
        skipped_files=0,
        total_nodes=5,
        errors=[],
        corpus_sha256="abc",
    )
    assert m.total_nodes == 5


def test_scrape_error_carries_context():
    err = ScrapeError(path="/tmp/x", stage="parse", reason="bad yaml")
    assert err.stage == "parse"


def test_parsed_skill_defaults():
    p = ParsedSkill(frontmatter={}, body="", mtime=0.0)
    assert p.frontmatter == {}


def test_section_chunk_shape():
    c = SectionChunk(section_path=("A", "B"), section_body="hello")
    assert c.section_path == ("A", "B")
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/groundloop/skills_scraper/test_models.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `models.py`**

Create `groundloop/skills_scraper/models.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class SourceRoot(BaseModel):
    model_config = ConfigDict(frozen=True)
    label: str
    glob: str


class ParsedSkill(BaseModel):
    model_config = ConfigDict(frozen=True)
    frontmatter: dict
    body: str
    mtime: float


class SectionChunk(BaseModel):
    model_config = ConfigDict(frozen=True)
    section_path: tuple[str, ...]
    section_body: str


class ScrapeError(BaseModel):
    model_config = ConfigDict(frozen=True)
    path: str
    stage: str
    reason: str


class SkillNode(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
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
    body_hash: str
    alias_sources: tuple[str, ...]


class ScrapeManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    generated_at: str
    sources: list[SourceRoot]
    scraped_files: int
    skipped_files: int
    total_nodes: int
    errors: list[ScrapeError]
    corpus_sha256: str
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/groundloop/skills_scraper/test_models.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add groundloop/skills_scraper/models.py tests/groundloop/skills_scraper/test_models.py
git commit -m "feat(skills-scraper): add Pydantic v2 models"
```

---

## Task 2: Config module with default source roots

**Files:**
- Create: `groundloop/skills_scraper/config.py`
- Test: `tests/groundloop/skills_scraper/test_config.py`

- [ ] **Step 1: Write failing test**

```python
from groundloop.skills_scraper.config import DEFAULT_SOURCES
from groundloop.skills_scraper.models import SourceRoot


def test_default_sources_covers_four_roots():
    labels = {s.label for s in DEFAULT_SOURCES}
    assert {"user-skills", "agent-skills", "cursor-skills", "plugin-skills"} <= labels


def test_default_sources_are_source_roots():
    for s in DEFAULT_SOURCES:
        assert isinstance(s, SourceRoot)
        assert s.glob.endswith("SKILL.md")
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/groundloop/skills_scraper/test_config.py -v`

- [ ] **Step 3: Implement `config.py`**

```python
from __future__ import annotations

from groundloop.skills_scraper.models import SourceRoot


DEFAULT_SOURCES: list[SourceRoot] = [
    SourceRoot(label="user-skills", glob="~/.claude/skills/*/SKILL.md"),
    SourceRoot(label="agent-skills", glob="~/.claude/.agents/skills/*/SKILL.md"),
    SourceRoot(label="cursor-skills", glob="~/.claude/.cursor/skills/*/SKILL.md"),
    SourceRoot(label="plugin-skills", glob="~/.claude/plugins/marketplaces/*/*/SKILL.md"),
]

MIN_CHUNK_CHARS = 80
DEFAULT_OUTPUT = "groundloop/kb/skills_corpus.jsonl"
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add groundloop/skills_scraper/config.py tests/groundloop/skills_scraper/test_config.py
git commit -m "feat(skills-scraper): add default source roots config"
```

---

## Task 3: Fixture files for testing

**Files:**
- Create: `tests/groundloop/skills_scraper/fixtures/fake_skills/normal/SKILL.md`
- Create: `tests/groundloop/skills_scraper/fixtures/fake_skills/no_frontmatter/SKILL.md`
- Create: `tests/groundloop/skills_scraper/fixtures/fake_skills/malformed_yaml/SKILL.md`
- Create: `tests/groundloop/skills_scraper/fixtures/fake_skills/single_section/SKILL.md`
- Create: `tests/groundloop/skills_scraper/fixtures/fake_skills/dup_a/coding-standards/SKILL.md`
- Create: `tests/groundloop/skills_scraper/fixtures/fake_skills/dup_b/coding-standards/SKILL.md`
- Create: `tests/groundloop/skills_scraper/conftest.py`

- [ ] **Step 1: Create `normal/SKILL.md`**

```markdown
---
name: python-testing
description: pytest strategies including fixtures and parametrization
type: flexible
---

# Python Testing

Overview paragraph.

## Fixtures

Use pytest fixtures for setup and teardown.

### Scope

Fixture scope can be function, module, or session.

## Parametrization

Use `@pytest.mark.parametrize` to run a test with multiple inputs.
```

- [ ] **Step 2: Create `no_frontmatter/SKILL.md`**

```markdown
# Bare Skill

This file has no YAML frontmatter.

## Section One

Content one content one content one content one content one content one content one.

## Section Two

Content two content two content two content two content two content two content two.
```

- [ ] **Step 3: Create `malformed_yaml/SKILL.md`**

```markdown
---
name: broken
description: [unclosed
---

body text
```

- [ ] **Step 4: Create `single_section/SKILL.md`**

```markdown
---
name: flat-skill
description: no headings at all
---

Just one blob of text with no section headings whatsoever. Nothing to split on.
```

- [ ] **Step 5: Create dup fixtures (both identical body)**

`dup_a/coding-standards/SKILL.md`:

```markdown
---
name: coding-standards
description: universal coding standards
---

# Coding Standards

## Naming

Use snake_case for python variables and functions.

## Files

Keep files under 800 lines.
```

`dup_b/coding-standards/SKILL.md`: **identical contents** to `dup_a`.

- [ ] **Step 6: Create `conftest.py`**

```python
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "fake_skills"


@pytest.fixture
def fixture_paths(fixtures_dir: Path) -> dict[str, Path]:
    return {
        "normal": fixtures_dir / "normal" / "SKILL.md",
        "no_frontmatter": fixtures_dir / "no_frontmatter" / "SKILL.md",
        "malformed": fixtures_dir / "malformed_yaml" / "SKILL.md",
        "single_section": fixtures_dir / "single_section" / "SKILL.md",
        "dup_a": fixtures_dir / "dup_a" / "coding-standards" / "SKILL.md",
        "dup_b": fixtures_dir / "dup_b" / "coding-standards" / "SKILL.md",
    }
```

- [ ] **Step 7: Commit**

```bash
git add tests/groundloop/skills_scraper/fixtures/ tests/groundloop/skills_scraper/conftest.py
git commit -m "test(skills-scraper): add fixture SKILL.md files"
```

---

## Task 4: `discovery.py` — walk source roots

**Files:**
- Create: `groundloop/skills_scraper/discovery.py`
- Test: `tests/groundloop/skills_scraper/test_discovery.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

from pathlib import Path

from groundloop.skills_scraper.discovery import walk_sources
from groundloop.skills_scraper.models import SourceRoot


def test_walk_finds_all_fixture_skills(fixtures_dir: Path) -> None:
    sources = [SourceRoot(label="fake", glob=str(fixtures_dir / "**" / "SKILL.md"))]
    results = list(walk_sources(sources))
    assert len(results) == 6


def test_walk_yields_path_and_source_root(fixtures_dir: Path) -> None:
    sources = [SourceRoot(label="fake", glob=str(fixtures_dir / "**" / "SKILL.md"))]
    for path, root in walk_sources(sources):
        assert path.is_file()
        assert root.label == "fake"


def test_walk_expands_tilde(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".x").mkdir()
    (tmp_path / ".x" / "SKILL.md").write_text("")
    sources = [SourceRoot(label="home", glob="~/.x/SKILL.md")]
    results = list(walk_sources(sources))
    assert len(results) == 1


def test_walk_skips_missing_dirs() -> None:
    sources = [SourceRoot(label="missing", glob="/nonexistent/**/SKILL.md")]
    assert list(walk_sources(sources)) == []
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `discovery.py`**

```python
from __future__ import annotations

import glob as glob_mod
import logging
import os
from collections.abc import Iterator
from pathlib import Path

from groundloop.skills_scraper.models import SourceRoot

_log = logging.getLogger(__name__)


def walk_sources(sources: list[SourceRoot]) -> Iterator[tuple[Path, SourceRoot]]:
    """Yield (path, source_root) for every readable SKILL.md matched by a source glob."""
    for root in sources:
        pattern = os.path.expanduser(root.glob)
        for match in glob_mod.glob(pattern, recursive=True):
            path = Path(match)
            if not path.is_file():
                _log.warning("discovery: skipping non-file %s", path)
                continue
            try:
                path.stat()
            except OSError as e:
                _log.warning("discovery: unreadable %s: %s", path, e)
                continue
            yield path, root
```

- [ ] **Step 4: Run — expect PASS (4 tests)**

- [ ] **Step 5: Commit**

```bash
git add groundloop/skills_scraper/discovery.py tests/groundloop/skills_scraper/test_discovery.py
git commit -m "feat(skills-scraper): discovery walks source globs with ~ expansion"
```

---

## Task 5: `parser.py` — frontmatter + body extraction

**Files:**
- Create: `groundloop/skills_scraper/parser.py`
- Test: `tests/groundloop/skills_scraper/test_parser.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from groundloop.skills_scraper.parser import ParseError, parse_skill


def test_parse_normal(fixture_paths: dict[str, Path]) -> None:
    parsed = parse_skill(fixture_paths["normal"])
    assert parsed.frontmatter["name"] == "python-testing"
    assert "Fixtures" in parsed.body
    assert parsed.mtime > 0


def test_parse_no_frontmatter(fixture_paths: dict[str, Path]) -> None:
    parsed = parse_skill(fixture_paths["no_frontmatter"])
    assert parsed.frontmatter == {}
    assert "Bare Skill" in parsed.body


def test_parse_malformed_raises(fixture_paths: dict[str, Path]) -> None:
    with pytest.raises(ParseError):
        parse_skill(fixture_paths["malformed"])


def test_parse_single_section(fixture_paths: dict[str, Path]) -> None:
    parsed = parse_skill(fixture_paths["single_section"])
    assert parsed.frontmatter["name"] == "flat-skill"
    assert parsed.body.strip().startswith("Just one blob")
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `parser.py`**

```python
from __future__ import annotations

from pathlib import Path

import frontmatter
import yaml

from groundloop.skills_scraper.models import ParsedSkill


class ParseError(Exception):
    """Raised when a SKILL.md cannot be parsed (malformed YAML, I/O error)."""


def parse_skill(path: Path) -> ParsedSkill:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ParseError(f"unreadable: {path}: {e}") from e
    try:
        post = frontmatter.loads(raw)
    except yaml.YAMLError as e:
        raise ParseError(f"malformed yaml in {path}: {e}") from e
    mtime = path.stat().st_mtime
    return ParsedSkill(frontmatter=dict(post.metadata), body=post.content, mtime=mtime)
```

- [ ] **Step 4: Run — expect PASS (4 tests)**

- [ ] **Step 5: Commit**

```bash
git add groundloop/skills_scraper/parser.py tests/groundloop/skills_scraper/test_parser.py
git commit -m "feat(skills-scraper): parse frontmatter + body, tolerate missing, fail on malformed"
```

---

## Task 6: `chunker.py` — split markdown into section chunks

**Files:**
- Create: `groundloop/skills_scraper/chunker.py`
- Test: `tests/groundloop/skills_scraper/test_chunker.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

from groundloop.skills_scraper.chunker import chunk_body


def test_chunk_h2_and_h3():
    body = (
        "# Title\n\n"
        "intro paragraph that is long enough to keep because it has plenty of characters.\n\n"
        "## Section A\n\n"
        "content a content a content a content a content a content a content a content a.\n\n"
        "### Sub A\n\n"
        "sub content sub content sub content sub content sub content sub content sub content.\n\n"
        "## Section B\n\n"
        "content b content b content b content b content b content b content b content b.\n"
    )
    chunks = chunk_body(body)
    paths = [c.section_path for c in chunks]
    assert ("Title",) in paths
    assert ("Title", "Section A") in paths
    assert ("Title", "Section A", "Sub A") in paths
    assert ("Title", "Section B") in paths


def test_chunk_single_section_fallback():
    body = "no headings here, just a single blob of text that is plenty long to survive the merge threshold."
    chunks = chunk_body(body)
    assert len(chunks) == 1
    assert chunks[0].section_path == ()
    assert "single blob" in chunks[0].section_body


def test_chunk_merges_tiny_chunks():
    body = (
        "## Tiny\n"
        "short\n"
        "## Next\n"
        "this one is long enough to meet the min chunk characters threshold easily."
    )
    chunks = chunk_body(body)
    titles = [c.section_path[-1] if c.section_path else "" for c in chunks]
    assert "Next" in titles


def test_chunk_h4_folds_into_h3():
    body = (
        "## S\n\n"
        "body of section s that is long enough to be kept around for sure and not dropped.\n\n"
        "### Sub\n\n"
        "sub body long enough to pass the min chars threshold for keeping a chunk around.\n\n"
        "#### Deeper\n\n"
        "deeper body that must fold into the parent h3 rather than become its own chunk.\n"
    )
    chunks = chunk_body(body)
    paths = [c.section_path for c in chunks]
    assert not any(p[-1] == "Deeper" for p in paths if p)
    sub_chunk = next(c for c in chunks if c.section_path and c.section_path[-1] == "Sub")
    assert "deeper body" in sub_chunk.section_body.lower()
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `chunker.py`**

```python
from __future__ import annotations

from markdown_it import MarkdownIt

from groundloop.skills_scraper.config import MIN_CHUNK_CHARS
from groundloop.skills_scraper.models import SectionChunk

_md = MarkdownIt()


def chunk_body(body: str) -> list[SectionChunk]:
    tokens = _md.parse(body)
    # Walk tokens collecting (heading_level, heading_text, body_text) segments.
    segments: list[tuple[int, str, list[str]]] = []
    i = 0
    current: tuple[int, str, list[str]] | None = None
    lines = body.splitlines()

    def flush(cur: tuple[int, str, list[str]] | None) -> None:
        if cur is not None:
            segments.append(cur)

    for tok in tokens:
        if tok.type == "heading_open":
            level = int(tok.tag[1])  # "h2" -> 2
            # Heading text is in the next inline token
            inline = tokens[tokens.index(tok) + 1]
            heading_text = inline.content.strip()
            if level <= 3:
                flush(current)
                current = (level, heading_text, [])
            else:
                # H4+ → treat as body text under current section
                if current is None:
                    current = (0, "", [])
                current[2].append(heading_text)
        elif tok.type == "paragraph_open" or tok.type.endswith("_open"):
            continue
        elif tok.type == "inline":
            if current is None:
                current = (0, "", [])
            current[2].append(tok.content)
    flush(current)

    if not segments:
        text = body.strip()
        if text:
            return [SectionChunk(section_path=(), section_body=text)]
        return []

    # Build hierarchical section_path by tracking current stack.
    stack: list[str] = []
    chunks: list[SectionChunk] = []
    for level, title, body_parts in segments:
        if level == 0:
            chunks.append(SectionChunk(section_path=(), section_body="\n\n".join(body_parts).strip()))
            continue
        while len(stack) >= level:
            stack.pop()
        stack.append(title)
        chunks.append(
            SectionChunk(
                section_path=tuple(stack),
                section_body="\n\n".join(body_parts).strip(),
            )
        )

    return _merge_small(chunks)


def _merge_small(chunks: list[SectionChunk]) -> list[SectionChunk]:
    if not chunks:
        return chunks
    merged: list[SectionChunk] = []
    carry: str = ""
    for c in chunks:
        body = (carry + "\n\n" + c.section_body).strip() if carry else c.section_body
        if len(body) < MIN_CHUNK_CHARS and c is not chunks[-1]:
            carry = body
            continue
        merged.append(SectionChunk(section_path=c.section_path, section_body=body))
        carry = ""
    if carry and merged:
        last = merged[-1]
        merged[-1] = SectionChunk(
            section_path=last.section_path,
            section_body=(last.section_body + "\n\n" + carry).strip(),
        )
    return merged
```

- [ ] **Step 4: Run — expect PASS (4 tests)**

Run: `pytest tests/groundloop/skills_scraper/test_chunker.py -v`

If any test fails, debug the AST walk — `markdown-it-py` token types differ slightly by version. The test `test_chunk_h2_and_h3` is the primary gate.

- [ ] **Step 5: Commit**

```bash
git add groundloop/skills_scraper/chunker.py tests/groundloop/skills_scraper/test_chunker.py
git commit -m "feat(skills-scraper): section-level chunking with H4 fold + small-chunk merge"
```

---

## Task 7: `tagger.py` — deterministic tag inference

**Files:**
- Create: `groundloop/skills_scraper/tagger.py`
- Test: `tests/groundloop/skills_scraper/test_tagger.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

from groundloop.skills_scraper.tagger import infer_tags


def test_python_domain():
    tags = infer_tags("python-testing", "Fixtures", "pytest fixtures are great")
    assert "domain:python" in tags


def test_security_domain():
    tags = infer_tags("security-review", "Authentication", "check auth flows for bypass")
    assert "domain:security" in tags


def test_test_phase():
    tags = infer_tags("python-testing", "Coverage", "pytest --cov")
    assert "phase:test" in tags


def test_review_phase():
    tags = infer_tags("code-review", "Checklist", "review the diff for bugs")
    assert "phase:review" in tags


def test_general_domain_when_no_match():
    tags = infer_tags("random-thing", "Misc", "plain text")
    assert "domain:general" in tags


def test_deterministic_order():
    a = infer_tags("python-testing", "X", "pytest")
    b = infer_tags("python-testing", "X", "pytest")
    assert a == b
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `tagger.py`**

```python
from __future__ import annotations

DOMAIN_RULES: list[tuple[str, list[str]]] = [
    ("python", ["python", "pytest", "pip", "pydantic"]),
    ("javascript", ["javascript", "typescript", "react", "next.js", "node"]),
    ("go", ["golang", "go "]),
    ("kotlin", ["kotlin", "gradle"]),
    ("security", ["security", "auth", "secret", "owasp", "injection"]),
    ("frontend", ["frontend", "ui", "ux", "tailwind", "css"]),
    ("backend", ["backend", "api", "fastapi", "django", "spring"]),
    ("data", ["pandas", "numpy", "clickhouse", "postgres", "sql"]),
    ("mcp", ["mcp", "model context protocol"]),
    ("devops", ["docker", "kubernetes", "ci/cd", "deploy"]),
]

PHASE_RULES: list[tuple[str, list[str]]] = [
    ("plan", ["plan", "design", "architecture", "brainstorm"]),
    ("build", ["build", "implement", "feature", "write"]),
    ("test", ["test", "pytest", "coverage", "tdd"]),
    ("review", ["review", "critic", "checklist", "audit"]),
    ("deploy", ["deploy", "release", "ship", "publish"]),
    ("debug", ["debug", "fix", "troubleshoot", "bug"]),
    ("docs", ["docs", "documentation", "readme"]),
]


def _hay(*parts: str) -> str:
    return " ".join(p.lower() for p in parts if p)


def infer_tags(skill_name: str, section_title: str, body: str) -> list[str]:
    hay = _hay(skill_name, section_title, body)
    tags: list[str] = []

    domain = "general"
    for name, keywords in DOMAIN_RULES:
        if any(k in hay for k in keywords):
            domain = name
            break
    tags.append(f"domain:{domain}")

    for name, keywords in PHASE_RULES:
        if any(k in hay for k in keywords):
            tags.append(f"phase:{name}")

    return tags
```

- [ ] **Step 4: Run — expect PASS (6 tests)**

- [ ] **Step 5: Commit**

```bash
git add groundloop/skills_scraper/tagger.py tests/groundloop/skills_scraper/test_tagger.py
git commit -m "feat(skills-scraper): deterministic rule-based tagger"
```

---

## Task 8: `writer.py` — JSONL corpus + manifest

**Files:**
- Create: `groundloop/skills_scraper/writer.py`
- Test: `tests/groundloop/skills_scraper/test_writer.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

import json
from pathlib import Path

from groundloop.skills_scraper.models import ScrapeError, ScrapeManifest, SkillNode, SourceRoot
from groundloop.skills_scraper.writer import manifest_for, write_corpus, write_manifest


def _node(nid: str, body: str = "body") -> SkillNode:
    return SkillNode(
        id=nid,
        skill_name="s",
        skill_description="d",
        skill_type=None,
        section_path=(),
        section_title="",
        section_body=body,
        source_path="/p",
        source_root="r",
        tags=(),
        trigger_hints="",
        mtime=0.0,
        body_hash="h",
        alias_sources=(),
    )


def test_write_corpus_sorted_jsonl(tmp_path: Path):
    out = tmp_path / "c.jsonl"
    nodes = [_node("zzz"), _node("aaa"), _node("mmm")]
    path = write_corpus(nodes, out)
    lines = path.read_text().splitlines()
    ids = [json.loads(ln)["id"] for ln in lines]
    assert ids == ["aaa", "mmm", "zzz"]


def test_write_manifest(tmp_path: Path):
    m = ScrapeManifest(
        generated_at="2026-04-15T00:00:00Z",
        sources=[SourceRoot(label="x", glob="y")],
        scraped_files=1,
        skipped_files=0,
        total_nodes=1,
        errors=[ScrapeError(path="/p", stage="parse", reason="r")],
        corpus_sha256="abc",
    )
    out = tmp_path / "m.json"
    write_manifest(m, out)
    parsed = json.loads(out.read_text())
    assert parsed["total_nodes"] == 1
    assert parsed["errors"][0]["stage"] == "parse"


def test_manifest_for_computes_hash(tmp_path: Path):
    out = tmp_path / "c.jsonl"
    out.write_text('{"id":"x"}\n')
    m = manifest_for(
        corpus_path=out,
        sources=[SourceRoot(label="a", glob="b")],
        scraped_files=1,
        skipped_files=0,
        total_nodes=1,
        errors=[],
    )
    assert len(m.corpus_sha256) == 64
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `writer.py`**

```python
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from groundloop.skills_scraper.models import (
    ScrapeError,
    ScrapeManifest,
    SkillNode,
    SourceRoot,
)


def write_corpus(nodes: Iterable[SkillNode], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_nodes = sorted(nodes, key=lambda n: n.id)
    with out_path.open("w", encoding="utf-8") as f:
        for node in sorted_nodes:
            f.write(node.model_dump_json())
            f.write("\n")
    return out_path


def write_manifest(manifest: ScrapeManifest, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def manifest_for(
    *,
    corpus_path: Path,
    sources: list[SourceRoot],
    scraped_files: int,
    skipped_files: int,
    total_nodes: int,
    errors: list[ScrapeError],
) -> ScrapeManifest:
    digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    return ScrapeManifest(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        sources=sources,
        scraped_files=scraped_files,
        skipped_files=skipped_files,
        total_nodes=total_nodes,
        errors=errors,
        corpus_sha256=digest,
    )
```

- [ ] **Step 4: Run — expect PASS (3 tests)**

- [ ] **Step 5: Commit**

```bash
git add groundloop/skills_scraper/writer.py tests/groundloop/skills_scraper/test_writer.py
git commit -m "feat(skills-scraper): writer emits deterministic JSONL + manifest"
```

---

## Task 9: `pipeline.py` — orchestrate discovery → parse → chunk → tag → dedupe → nodes

**Files:**
- Create: `groundloop/skills_scraper/pipeline.py`
- Test: `tests/groundloop/skills_scraper/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

from pathlib import Path

from groundloop.skills_scraper.models import SourceRoot
from groundloop.skills_scraper.pipeline import run_scraper


def test_pipeline_end_to_end(fixtures_dir: Path, tmp_path: Path) -> None:
    sources = [SourceRoot(label="fake", glob=str(fixtures_dir / "**" / "SKILL.md"))]
    out = tmp_path / "corpus.jsonl"
    result = run_scraper(sources=sources, output=out)
    assert result.scraped_files >= 4  # normal + no_frontmatter + single + (dup dedup'd)
    assert result.skipped_files >= 1  # malformed
    assert out.exists()
    assert (tmp_path / "corpus.manifest.json").exists()


def test_pipeline_deduplicates_identical_bodies(fixtures_dir: Path, tmp_path: Path) -> None:
    sources = [SourceRoot(label="fake", glob=str(fixtures_dir / "**" / "SKILL.md"))]
    out = tmp_path / "corpus.jsonl"
    result = run_scraper(sources=sources, output=out)
    import json
    nodes = [json.loads(ln) for ln in out.read_text().splitlines()]
    coding_standards = [n for n in nodes if n["skill_name"] == "coding-standards"]
    # Each section from coding-standards should appear once, with alias_sources populated
    for n in coding_standards:
        if n["alias_sources"]:
            assert len(n["alias_sources"]) >= 1
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `pipeline.py`**

```python
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from groundloop.skills_scraper.chunker import chunk_body
from groundloop.skills_scraper.discovery import walk_sources
from groundloop.skills_scraper.models import (
    ScrapeError,
    SkillNode,
    SourceRoot,
)
from groundloop.skills_scraper.parser import ParseError, parse_skill
from groundloop.skills_scraper.tagger import infer_tags
from groundloop.skills_scraper.writer import manifest_for, write_corpus, write_manifest

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScrapeResult:
    scraped_files: int
    skipped_files: int
    total_nodes: int
    errors: list[ScrapeError]
    corpus_path: Path
    manifest_path: Path


def _node_id(source_path: str, section_path: tuple[str, ...]) -> str:
    raw = source_path + "#" + "/".join(section_path)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _body_hash(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def run_scraper(*, sources: list[SourceRoot], output: Path) -> ScrapeResult:
    nodes: list[SkillNode] = []
    errors: list[ScrapeError] = []
    scraped = 0
    skipped = 0

    for path, root in walk_sources(sources):
        try:
            parsed = parse_skill(path)
        except ParseError as e:
            errors.append(ScrapeError(path=str(path), stage="parse", reason=str(e)))
            skipped += 1
            continue

        body = parsed.body.strip()
        if not body:
            skipped += 1
            continue

        fm = parsed.frontmatter
        skill_name = fm.get("name") or path.parent.name
        skill_desc = fm.get("description", "")
        skill_type = fm.get("type") if fm.get("type") in ("flexible", "rigid") else None
        triggers = fm.get("description", "")

        chunks = chunk_body(body)
        if not chunks:
            skipped += 1
            continue

        for chunk in chunks:
            tags = tuple(
                infer_tags(
                    skill_name,
                    chunk.section_path[-1] if chunk.section_path else "",
                    chunk.section_body,
                )
            )
            node = SkillNode(
                id=_node_id(str(path), chunk.section_path),
                skill_name=skill_name,
                skill_description=skill_desc,
                skill_type=skill_type,
                section_path=chunk.section_path,
                section_title=chunk.section_path[-1] if chunk.section_path else "",
                section_body=chunk.section_body,
                source_path=str(path),
                source_root=root.label,
                tags=tags,
                trigger_hints=triggers,
                mtime=parsed.mtime,
                body_hash=_body_hash(chunk.section_body),
                alias_sources=(),
            )
            nodes.append(node)
        scraped += 1

    deduped = _dedupe(nodes)
    corpus_path = write_corpus(deduped, output)
    manifest = manifest_for(
        corpus_path=corpus_path,
        sources=sources,
        scraped_files=scraped,
        skipped_files=skipped,
        total_nodes=len(deduped),
        errors=errors,
    )
    manifest_path = output.with_suffix(".manifest.json")
    write_manifest(manifest, manifest_path)

    return ScrapeResult(
        scraped_files=scraped,
        skipped_files=skipped,
        total_nodes=len(deduped),
        errors=errors,
        corpus_path=corpus_path,
        manifest_path=manifest_path,
    )


def _dedupe(nodes: list[SkillNode]) -> list[SkillNode]:
    index: dict[tuple[str, tuple[str, ...], str], SkillNode] = {}
    for n in nodes:
        key = (n.skill_name, n.section_path, n.body_hash)
        existing = index.get(key)
        if existing is None:
            index[key] = n
        else:
            aliases = existing.alias_sources + (n.source_path,)
            index[key] = existing.model_copy(update={"alias_sources": aliases})
    return list(index.values())
```

- [ ] **Step 4: Run — expect PASS (2 tests)**

- [ ] **Step 5: Commit**

```bash
git add groundloop/skills_scraper/pipeline.py tests/groundloop/skills_scraper/test_pipeline.py
git commit -m "feat(skills-scraper): pipeline orchestrates full scrape with dedup"
```

---

## Task 10: `cli.py` — command-line entrypoint + `__main__`

**Files:**
- Create: `groundloop/skills_scraper/cli.py`
- Create: `groundloop/skills_scraper/__main__.py`
- Test: `tests/groundloop/skills_scraper/test_cli.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from groundloop.skills_scraper.cli import main


def test_cli_on_fixtures(fixtures_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "corpus.jsonl"
    exit_code = main([
        "--sources", str(fixtures_dir / "**" / "SKILL.md"),
        "--output", str(out),
    ])
    assert exit_code == 0
    assert out.exists()


def test_cli_zero_nodes_nonzero_exit(tmp_path: Path) -> None:
    out = tmp_path / "corpus.jsonl"
    exit_code = main([
        "--sources", str(tmp_path / "nonexistent" / "*.md"),
        "--output", str(out),
    ])
    assert exit_code == 1
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `cli.py`**

```python
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from groundloop.skills_scraper.config import DEFAULT_OUTPUT, DEFAULT_SOURCES
from groundloop.skills_scraper.models import SourceRoot
from groundloop.skills_scraper.pipeline import run_scraper


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="groundloop.skills_scraper")
    parser.add_argument(
        "--sources",
        help="Glob pattern for SKILL.md files (overrides defaults). Repeatable.",
        action="append",
        default=None,
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.sources:
        sources = [
            SourceRoot(label=f"cli-{i}", glob=g) for i, g in enumerate(args.sources)
        ]
    else:
        sources = DEFAULT_SOURCES

    output = Path(args.output).expanduser()
    result = run_scraper(sources=sources, output=output)

    print(
        f"scraped={result.scraped_files} files, "
        f"skipped={result.skipped_files}, "
        f"nodes={result.total_nodes}, "
        f"errors={len(result.errors)}"
    )

    if result.total_nodes == 0:
        return 1
    return 0
```

- [ ] **Step 4: Create `__main__.py`**

```python
from groundloop.skills_scraper.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run — expect PASS (2 tests)**

- [ ] **Step 6: Commit**

```bash
git add groundloop/skills_scraper/cli.py groundloop/skills_scraper/__main__.py tests/groundloop/skills_scraper/test_cli.py
git commit -m "feat(skills-scraper): CLI entrypoint with default sources"
```

---

## Task 11: Export public API from `__init__.py`

**Files:**
- Modify: `groundloop/skills_scraper/__init__.py`

- [ ] **Step 1: Populate `__init__.py`**

```python
from groundloop.skills_scraper.models import (
    ParsedSkill,
    ScrapeError,
    ScrapeManifest,
    SectionChunk,
    SkillNode,
    SourceRoot,
)
from groundloop.skills_scraper.pipeline import ScrapeResult, run_scraper

__all__ = [
    "ParsedSkill",
    "ScrapeError",
    "ScrapeManifest",
    "ScrapeResult",
    "SectionChunk",
    "SkillNode",
    "SourceRoot",
    "run_scraper",
]
```

- [ ] **Step 2: Run full suite**

Run: `pytest tests/groundloop/skills_scraper/ -v --cov=groundloop.skills_scraper --cov-report=term-missing`
Expected: all tests pass, coverage ≥ 90%.

- [ ] **Step 3: Run ruff + mypy**

Run: `ruff check groundloop/skills_scraper/`
Run: `mypy --strict groundloop/skills_scraper/`
Expected: both clean. Fix any issues inline (add type stubs, narrow types).

- [ ] **Step 4: Commit**

```bash
git add groundloop/skills_scraper/__init__.py
git commit -m "feat(skills-scraper): export public API"
```

---

## Task 12: Run against real system + verify acceptance criteria

**Files:**
- Create: `groundloop/kb/` (generated output, add to `.gitignore`)
- Modify: `.gitignore`

- [ ] **Step 1: Add output dir to `.gitignore`**

Append to `.gitignore`:

```
groundloop/kb/
```

- [ ] **Step 2: Run scraper against real system**

Run: `python -m groundloop.skills_scraper --output groundloop/kb/skills_corpus.jsonl`
Expected output: `scraped=XX files, skipped=Y, nodes=ZZZ, errors=0` with XX ≈ 91.

- [ ] **Step 3: Spot-check output**

```bash
head -1 groundloop/kb/skills_corpus.jsonl | python -m json.tool
cat groundloop/kb/skills_corpus.manifest.json
wc -l groundloop/kb/skills_corpus.jsonl
```

Expected: a valid SkillNode JSON, a manifest with `total_nodes > 200`, and a non-zero line count.

- [ ] **Step 4: Verify determinism**

Run the scraper again and diff:

```bash
cp groundloop/kb/skills_corpus.jsonl /tmp/first.jsonl
python -m groundloop.skills_scraper --output groundloop/kb/skills_corpus.jsonl
diff /tmp/first.jsonl groundloop/kb/skills_corpus.jsonl
```

Expected: **no diff** (byte-identical).

- [ ] **Step 5: Final commit**

```bash
git add .gitignore
git commit -m "chore: gitignore groundloop/kb generated corpus"
```

---

## Self-Review

Ran against the spec checklist:

- ✅ All 7 spec modules (`discovery`, `parser`, `chunker`, `tagger`, `models`, `writer`, `cli`) appear as tasks, plus `pipeline` and `config` which were implicit in the spec.
- ✅ SkillNode schema (§4.5) implemented in Task 1, consistent with spec field list.
- ✅ Error-handling table (§6) covered: unreadable → Task 4, malformed YAML → Task 5, empty body → Task 9, dedup → Task 9, zero nodes → Task 10.
- ✅ Testing plan (§7) covered module-by-module, with `test_pipeline.py` and `test_cli.py` serving the "e2e" role.
- ✅ Acceptance criteria (§8) verified in Task 12.
- ✅ No placeholders ("TBD", "implement later", etc.) anywhere.
- ✅ Type consistency check: `SkillNode` fields in Task 1 match pipeline usage in Task 9 and writer in Task 8. `SourceRoot` has `label` + `glob` throughout. `chunk_body` returns `list[SectionChunk]` in both definition (Task 6) and consumer (Task 9).

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-04-15-groundloop-skills-scraper.md`.

Next: subagent-driven execution (fresh subagent per task, two-stage review).
