from __future__ import annotations

from markdown_it import MarkdownIt
from markdown_it.token import Token
from pydantic import BaseModel, ConfigDict

MIN_CHUNK_CHARS = 80
MAX_HEADING_LEVEL = 3

_md = MarkdownIt()


class SectionChunk(BaseModel):
    model_config = ConfigDict(frozen=True)
    section_path: tuple[str, ...]
    section_body: str


def _segment_tokens(
    tokens: list[Token],
) -> list[tuple[int, str, list[str]]]:
    """Walk token stream, returning heading-delimited segments.

    Each segment is (heading_level, heading_text, body_texts).
    Level 0 is a sentinel for content before the first H1-H3.
    """
    segments: list[tuple[int, str, list[str]]] = []
    current: tuple[int, str, list[str]] | None = None

    for i, tok in enumerate(tokens):
        if tok.type == "heading_open":
            level = int(tok.tag[1])  # "h2" -> 2
            heading_text = tokens[i + 1].content.strip()
            if level <= MAX_HEADING_LEVEL:
                if current is not None:
                    segments.append(current)
                current = (level, heading_text, [])
            else:
                if current is None:
                    current = (0, "", [])
                current[2].append(heading_text)
        elif tok.type.endswith("_open"):
            continue
        elif tok.type == "inline":
            if current is None:
                current = (0, "", [])
            current[2].append(tok.content)

    if current is not None:
        segments.append(current)
    return segments


def chunk_body(body: str) -> list[SectionChunk]:
    """Split markdown body into section chunks by H1-H3 headings."""
    tokens = _md.parse(body)
    segments = _segment_tokens(tokens)

    if not segments:
        text = body.strip()
        if text:
            return [SectionChunk(section_path=(), section_body=text)]
        return []

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
    """Merge chunks smaller than MIN_CHUNK_CHARS into their successor."""
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
