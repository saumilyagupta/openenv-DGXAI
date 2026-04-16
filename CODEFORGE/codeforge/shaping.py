from __future__ import annotations


def citation_shaping_bonus(
    *,
    submit_files: dict[str, str],
    prior_citations: list[dict[str, object]],
    prior_cluster_hits: list[str],
) -> float:
    """Retroactive shaping bonus for prior queries whose cited skills appear in submitted code.

    +0.01 per cited skill name found in the code text, max 0.05.
    Only fires on submit. See SYSTEM_DESIGN §4.8.4.
    """
    if not prior_citations:
        return 0.0
    cited_skills: set[str] = set()
    for c in prior_citations:
        sn = c.get("skill_name")
        if isinstance(sn, str):
            cited_skills.add(sn)
    code_text = " ".join(submit_files.values()).lower()
    overlap = sum(1 for skill in cited_skills if skill.replace("-", "_") in code_text)
    return min(overlap * 0.01, 0.05)
