"""HTTP API.  uv run --env-file .env uvicorn docs_rag.api:app"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from . import db
from .answer import answer
from .retrieve import search

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["conn"] = db.connect()
    yield
    state["conn"].close()


app = FastAPI(title="docs-rag", lifespan=lifespan)


@app.get("/search")
def search_docs(
    q: str = Query(min_length=2, max_length=500),
    method: Literal["vector", "keyword", "hybrid", "hybrid_rerank"] = "hybrid_rerank",
    k: int = Query(5, ge=1, le=20),
):
    hits = search(state["conn"], q, method=method, config="small", k=k)
    return [{"doc": h.doc, "section": h.heading, "score": round(h.score, 4), "text": h.content} for h in hits]


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


@app.post("/ask")
def ask(req: AskRequest):
    return answer(state["conn"], req.question)
