"""Integration test against the ingested database (skipped if it is not there)."""
import psycopg
import pytest
from fastapi.testclient import TestClient

from docs_rag import db


def has_corpus() -> bool:
    try:
        with psycopg.connect(db.DSN) as c:
            return c.execute("SELECT count(*) FROM chunks").fetchone()[0] > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not has_corpus(), reason="corpus not ingested")


def test_search_finds_the_right_tutorial_section():
    from docs_rag.api import app

    with TestClient(app) as client:
        r = client.get("/search", params={"q": "how do I make a query parameter optional", "k": 3})
        assert r.status_code == 200
        assert any(h["doc"] == "tutorial/query-params.md" for h in r.json())


def test_search_validates_input():
    from docs_rag.api import app

    with TestClient(app) as client:
        assert client.get("/search", params={"q": "x"}).status_code == 422
        assert client.get("/search", params={"q": "ok query", "method": "magic"}).status_code == 422
