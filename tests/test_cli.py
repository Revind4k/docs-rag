"""CLI formatting, no database needed."""
import pytest

from docs_rag.cli import format_hits, main, snippet
from docs_rag.retrieve import Hit


def hit(rank: int, score: float, content: str = "Query Parameters > Optional parameters\n\nSet the default to None.") -> Hit:
    return Hit(f"id{rank}", "tutorial/query-params.md", "Query Parameters > Optional parameters", content, score)


def test_rows_are_ranked_with_scores_and_sections():
    out = format_hits([hit(1, 0.91234), hit(2, 0.5)], "hybrid_rerank")
    lines = out.splitlines()
    assert lines[0] == "hybrid_rerank (2 hits)"
    assert lines[1].split()[:2] == ["1", "0.9123"]
    assert "tutorial/query-params.md  §  Query Parameters > Optional parameters" in lines[1]
    assert lines[2].split()[:2] == ["2", "0.5000"]


def test_text_flag_prints_the_chunk_without_its_heading_prefix():
    out = format_hits([hit(1, 0.9)], "vector", show_text=True)
    assert out.splitlines()[2].strip() == "Set the default to None."


def test_long_chunks_are_cut_to_width_on_one_line():
    long = "Query Parameters > Optional parameters\n\n" + "word " * 100
    s = snippet(hit(1, 0.9, long), 40)
    assert len(s) <= 40 and s.endswith("…") and "\n" not in s


def test_empty_result_says_so():
    assert format_hits([], "keyword").splitlines()[1].strip() == "no results"


def test_rejects_out_of_range_k_before_touching_the_database():
    with pytest.raises(SystemExit):
        main(["cors", "-k", "0"])
