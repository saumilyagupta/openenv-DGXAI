from __future__ import annotations

import re

DOMAIN_RULES: list[tuple[str, list[str]]] = [
    ("python", ["python", "pytest", "pip", "pydantic"]),
    ("javascript", ["javascript", "typescript", "react", "next.js", "node"]),
    ("go", ["golang", "go"]),
    ("kotlin", ["kotlin", "gradle"]),
    ("security", ["security", "auth", "secret", "owasp", "injection"]),
    ("frontend", ["frontend", "ui", "ux", "tailwind", "css"]),
    ("api", ["api", "endpoint", "rest", "graphql", "openapi"]),
    ("backend", ["backend", "fastapi", "django", "spring"]),
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

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")


def _tokenize(text: str) -> frozenset[str]:
    """Split lowercased text into alphanumeric tokens."""
    return frozenset(_TOKEN_RE.findall(text.lower()))


def _matches(keyword: str, tokens: frozenset[str], joined_lower: str) -> bool:
    if " " in keyword:
        return keyword in joined_lower
    return keyword in tokens


def infer_tags(skill_name: str, section_title: str, body: str) -> list[str]:
    """Infer domain and phase tags from skill name, section title, and body."""
    joined_lower = " ".join(
        p.lower() for p in (skill_name, section_title, body) if p
    )
    tokens = _tokenize(joined_lower)
    tags: list[str] = []

    domain = "general"
    for name, keywords in DOMAIN_RULES:
        if any(_matches(k, tokens, joined_lower) for k in keywords):
            domain = name
            break
    tags.append(f"domain:{domain}")

    for name, keywords in PHASE_RULES:
        if any(_matches(k, tokens, joined_lower) for k in keywords):
            tags.append(f"phase:{name}")

    return tags
