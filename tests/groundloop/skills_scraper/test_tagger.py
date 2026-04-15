from __future__ import annotations

from groundloop.skills_scraper.tagger import infer_tags


def test_python_domain():
    tags = infer_tags("python-testing", "Fixtures", "pytest fixtures are great")
    assert "domain:python" in tags


def test_api_domain():
    tags = infer_tags("api-design", "Endpoints", "rest endpoints and openapi")
    assert "domain:api" in tags


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
