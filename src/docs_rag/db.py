"""Postgres + pgvector storage. One table holds every chunking configuration side by side."""
from __future__ import annotations

import os

import psycopg
from pgvector.psycopg import register_vector

from .embed import DIM

DSN = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/docrag")

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    config    TEXT NOT NULL,
    id        TEXT NOT NULL,
    doc       TEXT NOT NULL,
    heading   TEXT NOT NULL,
    content   TEXT NOT NULL,
    embedding vector({DIM}) NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    PRIMARY KEY (config, id)
);
CREATE INDEX IF NOT EXISTS chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv);
"""


def connect() -> psycopg.Connection:
    conn = psycopg.connect(DSN, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
