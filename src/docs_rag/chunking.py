"""Split Markdown docs into heading-scoped sections, then into size-capped chunks.

Every chunk remembers the section it came from (doc path + heading path). The eval's gold
labels are sections, so retrieval quality can be compared across chunk sizes fairly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*(?:\{[^}]*\})?\s*$")
INCLUDE = re.compile(r"\{\*.*?\*\}", re.S)  # mkdocs code-include directives: no prose in them
FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)


@dataclass(frozen=True)
class Section:
    doc: str  # path relative to the corpus root, e.g. "tutorial/query-params.md"
    heading: str  # "Query Parameters > Optional parameters"
    text: str

    @property
    def key(self) -> str:
        return f"{self.doc}#{self.heading}"


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    heading: str
    text: str

    @property
    def section_key(self) -> str:
        return f"{self.doc}#{self.heading}"


def sections(path: Path, root: Path) -> list[Section]:
    raw = FRONT_MATTER.sub("", path.read_text(encoding="utf-8"))
    raw = INCLUDE.sub("", raw)
    doc = path.relative_to(root).as_posix()
    out: list[Section] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text and stack:
            out.append(Section(doc, " > ".join(t for _, t in stack), text))
        buf.clear()

    for line in raw.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            stack = [(lv, t) for lv, t in stack if lv < level] + [(level, m.group(2).strip())]
        else:
            buf.append(line)
    flush()
    return out


def chunk_section(s: Section, max_chars: int) -> list[str]:
    """Pack paragraphs into chunks of at most max_chars (a single long paragraph is split hard)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", s.text) if p.strip()]
    pieces: list[str] = []
    for p in paras:
        while len(p) > max_chars:
            pieces.append(p[:max_chars])
            p = p[max_chars:]
        pieces.append(p)
    chunks: list[str] = []
    cur = ""
    for p in pieces:
        if cur and len(cur) + 2 + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def load_corpus(root: Path) -> list[Section]:
    return [s for p in sorted(root.rglob("*.md")) for s in sections(p, root)]


def build_chunks(secs: list[Section], max_chars: int) -> list[Chunk]:
    out = []
    for s in secs:
        for i, text in enumerate(chunk_section(s, max_chars)):
            # The heading path goes into the embedded text: it carries a lot of the meaning.
            out.append(Chunk(f"{s.key}::{i}", s.doc, s.heading, f"{s.heading}\n\n{text}"))
    return out
