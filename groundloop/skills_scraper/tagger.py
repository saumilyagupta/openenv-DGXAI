from __future__ import annotations

DOMAIN_RULES: list[tuple[str, list[str]]] = [
    ("python", ["python", "pytest", "pip", "pydantic"]),
    ("javascript", ["javascript", "typescript", "react", "next.js", "node"]),
    ("go", ["golang", "go "]),
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
