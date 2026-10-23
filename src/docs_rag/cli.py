"""Print the chunks a query retrieves, with scores.

    uv run --env-file .env docs-rag-search "how do I make a query parameter optional"
    uv run --env-file .env docs-rag-search "cors" --method keyword -k 3 --text
    uv run --env-file .env docs-rag-search "background tasks" --compare

Useful for seeing why a question gets the answer it does: which sections came back,
in what order, and how far apart their scores are.
"""
from __future__ import annotations

import argparse
import sys
import textwrap

from .retrieve import METHODS, Hit


def snippet(hit: Hit, width: int) -> str:
    """Start of a chunk on one line. Chunks are stored as "<heading path>\n\n<text>"; drop the heading."""
    text = hit.content.removeprefix(hit.heading).lstrip()
    body = " ".join(text.split())
    return body if len(body) <= width else body[: width - 1].rstrip() + "…"


def format_hits(hits: list[Hit], method: str, show_text: bool = False, width: int = 72) -> str:
    lines = [f"{method} ({len(hits)} hits)"]
    if not hits:
        lines.append("  no results")
        return "\n".join(lines)
    for rank, h in enumerate(hits, start=1):
        lines.append(f"  {rank:>2}  {h.score:>8.4f}  {h.doc}  §  {h.heading}")
        if show_text:
            lines.append(textwrap.indent(snippet(h, width), " " * 16))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="docs-rag-search", description="Show retrieved chunks and their scores.")
    p.add_argument("query")
    p.add_argument("--method", choices=METHODS, default="hybrid_rerank")
    p.add_argument("-k", type=int, default=5, help="number of chunks to show (default 5)")
    p.add_argument("--config", choices=("small", "large"), default="small", help="chunk size: small = 500 chars, large = 2,000")
    p.add_argument("--text", action="store_true", help="also print the start of each chunk")
    p.add_argument("--compare", action="store_true", help="run all four methods and print them one after another")
    args = p.parse_args(argv)
    if not 1 <= args.k <= 50:
        p.error("-k must be between 1 and 50")

    from . import db
    from .retrieve import search

    conn = db.connect()
    try:
        methods = METHODS if args.compare else (args.method,)
        blocks = [format_hits(search(conn, args.query, method=m, config=args.config, k=args.k), m, args.text) for m in methods]
    finally:
        conn.close()
    print("\n\n".join(blocks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
