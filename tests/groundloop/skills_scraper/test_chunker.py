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
