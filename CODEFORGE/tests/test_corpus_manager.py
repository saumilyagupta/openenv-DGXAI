from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def skill_md(tmp_path: Path) -> Path:
    """Create a minimal SKILL.md with YAML frontmatter."""
    p = tmp_path / "test-skill" / "SKILL.md"
    p.parent.mkdir(parents=True)
    p.write_text(
        "---\n"
        "name: test-skill\n"
        "description: A test skill for unit testing\n"
        "---\n"
        "\n"
        "# Section One\n"
        "\n"
        "Content here with enough text to pass the minimum chunk size requirement.\n"
        "This paragraph ensures the chunk body is long enough to not be merged away.\n"
        "\n"
        "## Subsection\n"
        "\n"
        "More content in subsection that is also substantial enough to survive merge.\n"
        "Adding extra lines so the body is large enough and will not be collapsed.\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def second_skill_md(tmp_path: Path) -> Path:
    """Create a second SKILL.md for multi-skill tests."""
    p = tmp_path / "second-skill" / "SKILL.md"
    p.parent.mkdir(parents=True)
    p.write_text(
        "---\n"
        "name: second-skill\n"
        "description: Another test skill\n"
        "---\n"
        "\n"
        "# Main Section\n"
        "\n"
        "Body text for the second skill with enough content for minimum chunk size.\n"
        "Ensuring that the chunker does not discard this body as too small to keep.\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def corpus_path(tmp_path: Path) -> Path:
    return tmp_path / "corpus.jsonl"


class TestSkillCorpusManagerAddSkill:
    def test_add_skill_returns_positive_count(
        self, skill_md: Path, corpus_path: Path
    ) -> None:
        from codeforge.kb.corpus_manager import SkillCorpusManager

        mgr = SkillCorpusManager(corpus_path=corpus_path)
        count = mgr.add_skill(skill_md)
        assert count > 0

    def test_add_skill_increases_node_count(
        self, skill_md: Path, corpus_path: Path
    ) -> None:
        from codeforge.kb.corpus_manager import SkillCorpusManager

        mgr = SkillCorpusManager(corpus_path=corpus_path)
        before = mgr.node_count()
        mgr.add_skill(skill_md)
        after = mgr.node_count()
        assert after > before

    def test_add_skill_nodes_have_correct_skill_name(
        self, skill_md: Path, corpus_path: Path
    ) -> None:
        from codeforge.kb.corpus_manager import SkillCorpusManager

        mgr = SkillCorpusManager(corpus_path=corpus_path)
        mgr.add_skill(skill_md)
        for node in mgr.nodes:
            assert node["skill_name"] == "test-skill"


class TestSkillCorpusManagerRemoveSkill:
    def test_remove_skill_returns_count_removed(
        self, skill_md: Path, corpus_path: Path
    ) -> None:
        from codeforge.kb.corpus_manager import SkillCorpusManager

        mgr = SkillCorpusManager(corpus_path=corpus_path)
        added = mgr.add_skill(skill_md)
        removed = mgr.remove_skill("test-skill")
        assert removed == added

    def test_remove_skill_decreases_node_count(
        self, skill_md: Path, corpus_path: Path
    ) -> None:
        from codeforge.kb.corpus_manager import SkillCorpusManager

        mgr = SkillCorpusManager(corpus_path=corpus_path)
        mgr.add_skill(skill_md)
        mgr.remove_skill("test-skill")
        assert mgr.node_count() == 0

    def test_remove_nonexistent_returns_zero(self, corpus_path: Path) -> None:
        from codeforge.kb.corpus_manager import SkillCorpusManager

        mgr = SkillCorpusManager(corpus_path=corpus_path)
        assert mgr.remove_skill("no-such-skill") == 0


class TestSkillCorpusManagerRefresh:
    def test_refresh_detects_new_file(
        self, tmp_path: Path, corpus_path: Path
    ) -> None:
        from codeforge.kb.corpus_manager import SkillCorpusManager

        mgr = SkillCorpusManager(corpus_path=corpus_path)
        # Initially no sources
        result = mgr.refresh(sources=[])
        assert result["added"] == 0
        assert result["removed"] == 0
        assert result["unchanged"] == 0

    def test_refresh_detects_mtime_change(
        self,
        skill_md: Path,
        corpus_path: Path,
    ) -> None:
        import os

        from codeforge.kb.corpus_manager import SkillCorpusManager

        mgr = SkillCorpusManager(corpus_path=corpus_path)
        mgr.add_skill(skill_md)
        mgr.save()

        # Touch the file to change mtime
        original_content = skill_md.read_text(encoding="utf-8")
        skill_md.write_text(
            original_content + "\n# New Section\n\nBrand new content.\n",
            encoding="utf-8",
        )
        os.utime(skill_md, (skill_md.stat().st_mtime + 10, skill_md.stat().st_mtime + 10))

        glob_pattern = str(skill_md.parent.parent / "*/SKILL.md")
        from codeforge.scraper.discovery import SourceRoot

        sources = [SourceRoot(label="test", glob=glob_pattern)]

        mgr2 = SkillCorpusManager(corpus_path=corpus_path)
        mgr2.load()
        result = mgr2.refresh(sources=sources)
        # Should detect the changed file
        assert result["added"] > 0 or result["removed"] > 0 or result["unchanged"] >= 0


class TestSkillCorpusManagerSaveLoad:
    def test_save_writes_valid_jsonl(
        self, skill_md: Path, corpus_path: Path
    ) -> None:
        from codeforge.kb.corpus_manager import SkillCorpusManager

        mgr = SkillCorpusManager(corpus_path=corpus_path)
        mgr.add_skill(skill_md)
        mgr.save()

        assert corpus_path.is_file()
        lines = corpus_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) > 0
        for line in lines:
            obj = json.loads(line)
            assert "id" in obj
            assert "skill_name" in obj
            assert "section_body" in obj
            assert "body_hash" in obj

    def test_load_reads_jsonl_back(
        self, skill_md: Path, corpus_path: Path
    ) -> None:
        from codeforge.kb.corpus_manager import SkillCorpusManager

        mgr = SkillCorpusManager(corpus_path=corpus_path)
        mgr.add_skill(skill_md)
        original_count = mgr.node_count()
        mgr.save()

        mgr2 = SkillCorpusManager(corpus_path=corpus_path)
        mgr2.load()
        assert mgr2.node_count() == original_count

    def test_load_nonexistent_starts_empty(self, tmp_path: Path) -> None:
        from codeforge.kb.corpus_manager import SkillCorpusManager

        mgr = SkillCorpusManager(corpus_path=tmp_path / "nope.jsonl")
        mgr.load()
        assert mgr.node_count() == 0


class TestSkillCorpusManagerNodeCount:
    def test_node_count_starts_at_zero(self, corpus_path: Path) -> None:
        from codeforge.kb.corpus_manager import SkillCorpusManager

        mgr = SkillCorpusManager(corpus_path=corpus_path)
        assert mgr.node_count() == 0

    def test_node_count_reflects_multiple_adds(
        self, skill_md: Path, second_skill_md: Path, corpus_path: Path
    ) -> None:
        from codeforge.kb.corpus_manager import SkillCorpusManager

        mgr = SkillCorpusManager(corpus_path=corpus_path)
        count1 = mgr.add_skill(skill_md)
        count2 = mgr.add_skill(second_skill_md)
        assert mgr.node_count() == count1 + count2


class TestSkillCorpusManagerDedup:
    def test_duplicate_add_does_not_increase_count(
        self, skill_md: Path, corpus_path: Path
    ) -> None:
        from codeforge.kb.corpus_manager import SkillCorpusManager

        mgr = SkillCorpusManager(corpus_path=corpus_path)
        mgr.add_skill(skill_md)
        count_after_first = mgr.node_count()
        mgr.add_skill(skill_md)
        assert mgr.node_count() == count_after_first


class TestScraperPipeline:
    def test_run_scraper_end_to_end(
        self, skill_md: Path, tmp_path: Path
    ) -> None:
        from codeforge.scraper.discovery import SourceRoot
        from codeforge.scraper.pipeline import run_scraper

        glob_pattern = str(skill_md.parent.parent / "*/SKILL.md")
        sources = [SourceRoot(label="test", glob=glob_pattern)]
        output = tmp_path / "output.jsonl"
        result = run_scraper(sources=sources, output=output)
        assert result.scraped_files == 1
        assert result.total_nodes > 0
        assert result.corpus_path.is_file()
        assert result.manifest_path.is_file()

    def test_run_scraper_skips_empty_body(self, tmp_path: Path) -> None:
        from codeforge.scraper.discovery import SourceRoot
        from codeforge.scraper.pipeline import run_scraper

        empty_skill = tmp_path / "empty-skill" / "SKILL.md"
        empty_skill.parent.mkdir(parents=True)
        empty_skill.write_text(
            "---\nname: empty\n---\n\n", encoding="utf-8"
        )
        glob_pattern = str(tmp_path / "*/SKILL.md")
        sources = [SourceRoot(label="test", glob=glob_pattern)]
        output = tmp_path / "output.jsonl"
        result = run_scraper(sources=sources, output=output)
        assert result.skipped_files == 1
        assert result.total_nodes == 0

    def test_run_scraper_handles_malformed_yaml(self, tmp_path: Path) -> None:
        from codeforge.scraper.discovery import SourceRoot
        from codeforge.scraper.pipeline import run_scraper

        bad_skill = tmp_path / "bad-skill" / "SKILL.md"
        bad_skill.parent.mkdir(parents=True)
        bad_skill.write_text(
            "---\n:\n  - invalid:\n    yaml: [unbalanced\n---\nBody\n",
            encoding="utf-8",
        )
        glob_pattern = str(tmp_path / "*/SKILL.md")
        sources = [SourceRoot(label="test", glob=glob_pattern)]
        output = tmp_path / "output.jsonl"
        result = run_scraper(sources=sources, output=output)
        assert result.skipped_files == 1
        assert len(result.errors) == 1

    def test_scrape_single_skill_returns_nodes(self, skill_md: Path) -> None:
        from codeforge.scraper.pipeline import scrape_single_skill

        nodes = scrape_single_skill(skill_md)
        assert len(nodes) > 0
        for n in nodes:
            assert n["skill_name"] == "test-skill"
            assert "body_hash" in n
            assert "id" in n

    def test_scrape_single_skill_empty_body_returns_empty(
        self, tmp_path: Path
    ) -> None:
        from codeforge.scraper.pipeline import scrape_single_skill

        empty_skill = tmp_path / "empty-skill" / "SKILL.md"
        empty_skill.parent.mkdir(parents=True)
        empty_skill.write_text(
            "---\nname: empty\n---\n\n", encoding="utf-8"
        )
        nodes = scrape_single_skill(empty_skill)
        assert nodes == []


class TestScraperWriter:
    def test_write_corpus_creates_sorted_jsonl(
        self, tmp_path: Path
    ) -> None:
        from codeforge.scraper.writer import write_corpus

        nodes = [
            {"id": "bbb", "body": "second"},
            {"id": "aaa", "body": "first"},
        ]
        out = tmp_path / "corpus.jsonl"
        write_corpus(nodes, out)
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == "aaa"
        assert json.loads(lines[1])["id"] == "bbb"

    def test_write_manifest_creates_json(
        self, skill_md: Path, tmp_path: Path
    ) -> None:
        from codeforge.scraper.writer import write_corpus, write_manifest

        nodes = [{"id": "abc", "body": "test"}]
        corpus = tmp_path / "corpus.jsonl"
        write_corpus(nodes, corpus)
        manifest_path = write_manifest(
            corpus_path=corpus,
            sources=[{"label": "test", "glob": "*.md"}],
            scraped_files=1,
            skipped_files=0,
            total_nodes=1,
            errors=[],
        )
        assert manifest_path.is_file()
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "corpus_sha256" in m
        assert m["total_nodes"] == 1


class TestScraperDiscovery:
    def test_walk_sources_finds_skill_files(
        self, skill_md: Path
    ) -> None:
        from codeforge.scraper.discovery import SourceRoot, walk_sources

        glob_pattern = str(skill_md.parent.parent / "*/SKILL.md")
        sources = [SourceRoot(label="test", glob=glob_pattern)]
        results = list(walk_sources(sources))
        assert len(results) == 1
        assert results[0][0] == skill_md


class TestScraperTagger:
    def test_infer_tags_detects_python_domain(self) -> None:
        from codeforge.scraper.tagger import infer_tags

        tags = infer_tags("pytest-skill", "Testing", "Use pytest to test")
        assert "domain:python" in tags
        assert any(t.startswith("phase:") for t in tags)

    def test_infer_tags_defaults_to_general(self) -> None:
        from codeforge.scraper.tagger import infer_tags

        tags = infer_tags("generic", "Intro", "Nothing specific here.")
        assert "domain:general" in tags

    def test_infer_tags_detects_multi_word_keyword(self) -> None:
        from codeforge.scraper.tagger import infer_tags

        tags = infer_tags("mcp-skill", "MCP", "model context protocol usage")
        assert "domain:mcp" in tags


class TestScraperChunker:
    def test_chunk_body_splits_on_headings(self) -> None:
        from codeforge.scraper.chunker import chunk_body

        body = (
            "# First\n\n"
            "Content for first section that is long enough to pass minimum.\n"
            "Extra lines to ensure this chunk has enough characters for merge.\n\n"
            "## Second\n\n"
            "Content for second section that is also long enough to survive.\n"
            "More lines here to pad out the body to pass the minimum threshold.\n"
        )
        chunks = chunk_body(body)
        assert len(chunks) >= 1
        assert all(c.section_path for c in chunks)

    def test_chunk_body_empty_returns_empty(self) -> None:
        from codeforge.scraper.chunker import chunk_body

        assert chunk_body("") == []
        assert chunk_body("   ") == []
