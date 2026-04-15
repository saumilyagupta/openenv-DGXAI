from groundloop.skills_scraper.config import DEFAULT_SOURCES
from groundloop.skills_scraper.models import SourceRoot


def test_default_sources_covers_four_roots():
    labels = {s.label for s in DEFAULT_SOURCES}
    assert {"user-skills", "agent-skills", "cursor-skills", "plugin-skills"} <= labels


def test_default_sources_are_source_roots():
    for s in DEFAULT_SOURCES:
        assert isinstance(s, SourceRoot)
        assert s.glob.endswith("SKILL.md")
