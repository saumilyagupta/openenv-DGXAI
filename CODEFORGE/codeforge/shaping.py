from __future__ import annotations

import re

# Match explicit citation comments: # cited: skill-name or # ref: skill-name
_CITATION_RE = re.compile(r"#\s*(?:cited|ref|source|from):\s*(\S+)", re.IGNORECASE)


def citation_shaping_bonus(
    *,
    submit_files: dict[str, str],
    prior_citations: list[dict[str, object]],
    prior_cluster_hits: list[str],
) -> float:
    """Retroactive shaping bonus for prior queries whose cited skills appear in submitted code.

    +0.01 per cited skill name found as an explicit citation comment in the code,
    max 0.05.  Only fires on submit with reward > 0.  See SYSTEM_DESIGN §4.8.4.

    Uses explicit comment matching (``# cited: skill-name``) instead of substring
    matching to prevent common-word skill names from trivially matching any code.
    """
    if not prior_citations:
        return 0.0

    cited_skills: set[str] = set()
    for c in prior_citations:
        sn = c.get("skill_name")
        if isinstance(sn, str):
            cited_skills.add(sn.lower())

    code_text = "\n".join(submit_files.values())
    code_citations = {m.group(1).lower() for m in _CITATION_RE.finditer(code_text)}

    overlap = len(cited_skills & code_citations)
    return min(overlap * 0.01, 0.05)
