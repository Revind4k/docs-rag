"""Score every retrieval method x chunk size against the frozen questions.

A retrieval is a hit at k if any of the top-k chunks comes from the gold section.
    uv run python eval/run_eval.py
"""
from __future__ import annotations

import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from docs_rag import db, retrieve
from docs_rag.ingest import CONFIGS

ROOT = Path(__file__).resolve().parents[1]
KS = (1, 5, 10)


def score(ranked_keys: list[str], gold: str) -> tuple[dict[int, int], float]:
    hits = {k: int(gold in ranked_keys[:k]) for k in KS}
    rank = next((i for i, key in enumerate(ranked_keys[:10], start=1) if key == gold), None)
    return hits, (1.0 / rank if rank else 0.0)


def main() -> None:
    questions = [json.loads(line) for line in (ROOT / "eval/questions.jsonl").read_text().splitlines()]
    conn = db.connect()
    rows = []
    for config in CONFIGS:
        for method in retrieve.METHODS:
            recall = {k: 0 for k in KS}
            mrr, latencies = 0.0, []
            for q in questions:
                t = time.perf_counter()
                hits = retrieve.search(conn, q["question"], method=method, config=config, k=10)
                latencies.append((time.perf_counter() - t) * 1000)
                h, rr = score([x.section_key for x in hits], q["gold"])
                for k in KS:
                    recall[k] += h[k]
                mrr += rr
            n = len(questions)
            rows.append({
                "config": config, "method": method,
                **{f"recall@{k}": recall[k] / n for k in KS},
                "mrr@10": mrr / n,
                "p50_ms": statistics.median(latencies),
            })
            r = rows[-1]
            print(f"{config:5} {method:14} R@1 {r['recall@1']:.2f}  R@5 {r['recall@5']:.2f}  R@10 {r['recall@10']:.2f}  MRR {r['mrr@10']:.3f}  p50 {r['p50_ms']:.0f} ms")
    out = ROOT / f"eval/results/{datetime.now():%Y%m%dT%H%M%S}.json"
    out.write_text(json.dumps({"n": len(questions), "results": rows}, indent=2))


if __name__ == "__main__":
    main()
