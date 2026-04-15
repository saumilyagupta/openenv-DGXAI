from __future__ import annotations

from markdown_it import MarkdownIt

from groundloop.skills_scraper.config import MIN_CHUNK_CHARS
from groundloop.skills_scraper.models import SectionChunk

_md = MarkdownIt()


def chunk_body(body: str) -> list[SectionChunk]:
    tokens = _md.parse(body)
    # Walk tokens collecting (heading_level, heading_text, body_text) segments.
    segments: list[tuple[int, str, list[str]]] = []
    current: tuple[int, str, list[str]] | None = None

    def flush(cur: tuple[int, str, list[str]] | None) -> None:
        if cur is not None:
            segments.append(cur)

    for tok in tokens:
        if tok.type == "heading_open":
            level = int(tok.tag[1])  # "h2" -> 2
            # Heading text is in the next inline token
            inline = tokens[tokens.index(tok) + 1]
            heading_text = inline.content.strip()
            if level <= 3:
                flush(current)
                current = (level, heading_text, [])
            else:
                # H4+ → treat as body text under current section.
                # Level-0 sentinel captures any body/H4 content that
                # precedes the first H1-H3. Pipeline normalizes its
                # section_path=() to (skill_name,) per spec §4.3.
                if current is None:
                    current = (0, "", [])
                current[2].append(heading_text)
        elif tok.type == "paragraph_open" or tok.type.endswith("_open"):
            continue
        elif tok.type == "inline":
            # See level-0 sentinel note above.
            if current is None:
                current = (0, "", [])
            current[2].append(tok.content)
    flush(current)

    if not segments:
        text = body.strip()
        if text:
            return [SectionChunk(section_path=(), section_body=text)]
        return []

    # Build hierarchical section_path by tracking current stack.
    stack: list[str] = []
    chunks: list[SectionChunk] = []
    for level, title, body_parts in segments:
        if level == 0:
            chunks.append(
                SectionChunk(
                    section_path=(),
                    section_body="\n\n".join(body_parts).strip(),
                )
            )
            continue
        while len(stack) >= level:
            stack.pop()
        stack.append(title)
        chunks.append(
            SectionChunk(
                section_path=tuple(stack),
                section_body="\n\n".join(body_parts).strip(),
            )
        )

    return _merge_small(chunks)


def _merge_small(chunks: list[SectionChunk]) -> list[SectionChunk]:
    if not chunks:
        return chunks
    merged: list[SectionChunk] = []
    carry: str = ""
    for c in chunks:
        body = (carry + "\n\n" + c.section_body).strip() if carry else c.section_body
        if len(body) < MIN_CHUNK_CHARS and c is not chunks[-1]:
            carry = body
            continue
        merged.append(SectionChunk(section_path=c.section_path, section_body=body))
        carry = ""
    if carry and merged:
        last = merged[-1]
        merged[-1] = SectionChunk(
            section_path=last.section_path,
            section_body=(last.section_body + "\n\n" + carry).strip(),
        )
    return merged
