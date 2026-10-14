"""Chunk the corpus at each configured size, embed locally, and load into Postgres.

    uv run python -m docs_rag.ingest
"""
from __future__ import annotations

import time
from pathlib import Path

from . import db
from .chunking import build_chunks, load_corpus
from .embed import embed_passages

CONFIGS = {"small": 500, "large": 2000}  # max characters per chunk
CORPUS = Path(__file__).resolve().parents[2] / "corpus"


def main() -> None:
    secs = load_corpus(CORPUS)
    print(f"{len(secs)} sections from {len({s.doc for s in secs})} docs")
    conn = db.connect()
    db.init(conn)
    for config, max_chars in CONFIGS.items():
        t = time.time()
        chunks = build_chunks(secs, max_chars)
        vecs = embed_passages([c.text for c in chunks])
        conn.execute("DELETE FROM chunks WHERE config = %s", (config,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (config, id, doc, heading, content, embedding) VALUES (%s, %s, %s, %s, %s, %s)",
                [(config, c.id, c.doc, c.heading, c.text, v) for c, v in zip(chunks, vecs)],
            )
        print(f"{config}: {len(chunks)} chunks of <= {max_chars} chars in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
