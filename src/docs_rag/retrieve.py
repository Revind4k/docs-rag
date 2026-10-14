"""Four retrieval methods over the same chunks: vector, keyword, hybrid (RRF), hybrid + rerank."""
from __future__ import annotations

from dataclasses import dataclass

import psycopg

from .embed import embed_query, rerank

METHODS = ("vector", "keyword", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    id: str
    doc: str
    heading: str
    content: str
    score: float

    @property
    def section_key(self) -> str:
        return f"{self.doc}#{self.heading}"


def vector(conn: psycopg.Connection, config: str, q: str, k: int) -> list[Hit]:
    rows = conn.execute(
        """SELECT id, doc, heading, content, 1 - (embedding <=> %s) AS score
           FROM chunks WHERE config = %s ORDER BY embedding <=> %s LIMIT %s""",
        (embed_query(q), config, embed_query(q), k),
    ).fetchall()
    return [Hit(*r) for r in rows]


def keyword(conn: psycopg.Connection, config: str, q: str, k: int) -> list[Hit]:
    rows = conn.execute(
        """SELECT id, doc, heading, content, ts_rank_cd(tsv, query) AS score
           FROM chunks, websearch_to_tsquery('english', %s) query
           WHERE config = %s AND tsv @@ query ORDER BY score DESC LIMIT %s""",
        (q, config, k),
    ).fetchall()
    if rows:
        return [Hit(*r) for r in rows]
    # websearch_to_tsquery ANDs every term; fall back to OR so long questions still match.
    terms = " | ".join(w for w in "".join(c if c.isalnum() else " " for c in q).split() if len(w) > 2)
    if not terms:
        return []
    rows = conn.execute(
        """SELECT id, doc, heading, content, ts_rank_cd(tsv, query) AS score
           FROM chunks, to_tsquery('english', %s) query
           WHERE config = %s AND tsv @@ query ORDER BY score DESC LIMIT %s""",
        (terms, config, k),
    ).fetchall()
    return [Hit(*r) for r in rows]


def rrf(rankings: list[list[Hit]], k: int, c: int = 60) -> list[Hit]:
    """Reciprocal rank fusion: score = sum of 1 / (c + rank) over the lists a chunk appears in."""
    scores: dict[str, float] = {}
    hits: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking, start=1):
            scores[h.id] = scores.get(h.id, 0.0) + 1.0 / (c + rank)
            hits.setdefault(h.id, h)
    ordered = sorted(scores, key=scores.get, reverse=True)[:k]
    return [Hit(hits[i].id, hits[i].doc, hits[i].heading, hits[i].content, scores[i]) for i in ordered]


def search(conn: psycopg.Connection, q: str, method: str = "hybrid_rerank", config: str = "small", k: int = 10) -> list[Hit]:
    if method == "vector":
        return vector(conn, config, q, k)
    if method == "keyword":
        return keyword(conn, config, q, k)
    pool = rrf([vector(conn, config, q, 50), keyword(conn, config, q, 50)], k=50 if method == "hybrid_rerank" else k)
    if method == "hybrid":
        return pool
    if method == "hybrid_rerank":
        scores = rerank(q, [h.content for h in pool])
        ranked = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)[:k]
        return [Hit(h.id, h.doc, h.heading, h.content, float(s)) for h, s in ranked]
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
