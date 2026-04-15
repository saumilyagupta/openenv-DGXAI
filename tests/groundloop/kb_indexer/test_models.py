import pytest
from pydantic import ValidationError

from groundloop.kb_indexer.models import SearchResult


def test_search_result_frozen():
    r = SearchResult(
        node_id="abc",
        skill_name="python-testing",
        section_path=("Fixtures",),
        section_body="body",
        tags=("domain:python",),
        source_path="/p",
        score=1.23,
        rank=1,
    )
    with pytest.raises(ValidationError):
        r.rank = 2  # type: ignore[misc]


def test_search_result_requires_fields():
    with pytest.raises(ValidationError):
        SearchResult()  # type: ignore[call-arg]
