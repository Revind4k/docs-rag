"""Grounded answers: retrieve, then ask Claude to answer only from the retrieved chunks, with citations."""
from __future__ import annotations

import anthropic
import psycopg

from .retrieve import Hit, search

MODEL = "claude-opus-5-5"

SYSTEM = """You answer questions about FastAPI using only the numbered documentation excerpts provided.
Cite the excerpts you use inline as [1], [2], ... If the excerpts do not contain the answer, say that the docs
excerpts do not cover it instead of answering from memory. Keep answers short and include a code snippet only
when an excerpt shows one."""


def build_prompt(question: str, hits: list[Hit]) -> str:
    excerpts = "\n\n".join(f"[{i}] ({h.doc} > {h.heading})\n{h.content}" for i, h in enumerate(hits, start=1))
    return f"Documentation excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def answer(conn: psycopg.Connection, question: str, client: anthropic.Anthropic | None = None, k: int = 5) -> dict:
    hits = search(conn, question, method="hybrid_rerank", config="small", k=k)
    client = client or anthropic.Anthropic()
    r = client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
        # On a safety-classifier refusal, re-run on a fallback model inside the same call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if r.stop_reason == "refusal":
        text = "The model declined to answer this question."
    else:
        text = "".join(b.text for b in r.content if b.type == "text").strip()
    return {
        "answer": text,
        "sources": [{"n": i, "doc": h.doc, "section": h.heading} for i, h in enumerate(hits, start=1)],
    }
