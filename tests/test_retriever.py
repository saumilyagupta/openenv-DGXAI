"""Tests for the BM25 retriever."""

import pytest
from server.retriever import BM25Retriever


@pytest.fixture(scope="module")
def retriever():
    return BM25Retriever("data/evidence.json")


class TestBM25Retriever:

    def test_loads_evidence(self, retriever):
        assert len(retriever.snippets) == 2000

    def test_bm25_initialized(self, retriever):
        assert retriever.bm25 is not None

    def test_search_returns_results(self, retriever):
        results = retriever.search("water boiling point")
        assert len(results) > 0
        assert len(results) <= 3  # default top_k=3

    def test_search_result_structure(self, retriever):
        results = retriever.search("DNA genetics")
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "relevance_score" in r
            assert isinstance(r["relevance_score"], float)

    def test_search_relevance_ordering(self, retriever):
        results = retriever.search("photosynthesis chlorophyll")
        if len(results) >= 2:
            scores = [r["relevance_score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_search_custom_top_k(self, retriever):
        results = retriever.search("history war", top_k=5)
        assert len(results) <= 5

    def test_empty_query(self, retriever):
        results = retriever.search("")
        assert isinstance(results, list)

    def test_nonexistent_path(self):
        r = BM25Retriever("data/nonexistent.json")
        assert len(r.snippets) == 0
        assert r.search("anything") == []
