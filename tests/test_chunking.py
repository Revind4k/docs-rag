from pathlib import Path

from docs_rag.chunking import Section, build_chunks, chunk_section, sections
from docs_rag.retrieve import Hit, rrf


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "doc.md"
    p.write_text(text)
    return p


def test_sections_track_heading_path_and_skip_fenced_code(tmp_path):
    p = write(tmp_path, "# Top\n\nintro\n\n## Child\n\nbody\n\n```python\n# not a heading\n```\n\n# Other\n\nmore\n")
    secs = sections(p, tmp_path)
    assert [s.heading for s in secs] == ["Top", "Top > Child", "Other"]
    assert "# not a heading" in secs[1].text


def test_include_directives_and_front_matter_are_removed(tmp_path):
    p = write(tmp_path, "---\ntitle: x\n---\n# H\n\ntext {* ../../docs_src/x.py hl[3] *} more\n")
    (s,) = sections(p, tmp_path)
    assert "docs_src" not in s.text and "title:" not in s.text


def test_chunks_respect_the_size_cap_and_keep_section_identity():
    s = Section("a.md", "H", "\n\n".join(["word " * 40] * 10))
    pieces = chunk_section(s, 500)
    assert len(pieces) > 1 and all(len(p) <= 500 for p in pieces)
    chunks = build_chunks([s], 500)
    assert {c.section_key for c in chunks} == {"a.md#H"}
    assert all(c.text.startswith("H\n\n") for c in chunks)


def hit(i: str) -> Hit:
    return Hit(i, "d.md", "h", "", 0.0)


def test_rrf_rewards_agreement_between_rankings():
    a = [hit("x"), hit("y"), hit("z")]
    b = [hit("y"), hit("w")]
    fused = [h.id for h in rrf([a, b], k=4)]
    assert fused[0] == "y"  # second in one list and first in the other beats first in only one
    assert set(fused) == {"x", "y", "z", "w"}
