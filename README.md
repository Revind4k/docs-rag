# docs-rag

Retrieval-augmented question answering over the FastAPI documentation, built so retrieval quality can be **measured**. It indexes the same corpus at two chunk sizes in Postgres + pgvector, compares four retrieval methods on a frozen 100-question eval, and answers with Claude using cited excerpts.

## Results

100 questions, each written by Claude from one documentation section; that section is the gold label (see [Eval](#eval)). Recall@k is the share of questions whose gold section appears in the top k chunks.

| Chunks | Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | p50 latency |
|---|---|---|---|---|---|---|
| 500 chars | vector | 0.39 | 0.63 | 0.76 | 0.494 | 12 ms |
| 500 chars | keyword (Postgres full-text) | 0.19 | 0.44 | 0.67 | 0.310 | 9 ms |
| 500 chars | hybrid (RRF) | 0.39 | 0.74 | 0.84 | 0.535 | 22 ms |
| 500 chars | **hybrid + cross-encoder rerank** | **0.52** | **0.80** | **0.90** | **0.637** | 554 ms |
| 2,000 chars | vector | 0.42 | 0.72 | 0.84 | 0.543 | 14 ms |
| 2,000 chars | keyword | 0.12 | 0.25 | 0.41 | 0.188 | 7 ms |
| 2,000 chars | hybrid (RRF) | 0.30 | 0.68 | 0.85 | 0.466 | 21 ms |
| 2,000 chars | hybrid + rerank | 0.52 | 0.83 | 0.88 | 0.647 | 1,720 ms |

What this shows:
- **Fusion and reranking compound.** On 500-character chunks, hybrid search lifts recall@5 from 0.63 to 0.74, and reranking the fused top 50 lifts it to 0.80. Recall@1 goes from 0.39 to 0.52.
- **Chunk size cuts both ways.** Larger chunks help pure vector search (more context per embedding) but hurt keyword search (diluted term statistics), so the hybrid gets worse with them.
- **Reranking is the expensive step.** It runs on CPU, so latency grows with chunk length: 554 ms on small chunks vs 1.7 s on large. The API therefore uses small chunks with hybrid + rerank.

Latency is measured on an Apple M4 laptop, CPU only.

## How it works

```
corpus/*.md ── chunking.py ──> sections (doc + heading path) ──> chunks (500 / 2,000 chars)
                                                                      │  bge-small-en-v1.5 (local ONNX)
                                                                      v
                                       Postgres: pgvector HNSW index + tsvector GIN index
                                                                      │
query ──> vector top 50 + keyword top 50 ──> reciprocal rank fusion ──> ms-marco cross-encoder ──> top k
                                                                                                    │
                                                                    Claude answers from top 5, cites [n]
```

- **Section-aware chunking.** Markdown is split on headings, ignoring `#` inside code fences. mkdocs include directives are stripped, and each chunk is prefixed with its heading path (`Query Parameters > Optional parameters`), which carries much of the meaning.
- **Fully local retrieval.** Embeddings and reranking run as ONNX models through `fastembed`, so there is no embedding API and no GPU. Only `/ask` calls Claude.
- **Grounded answers.** Claude may answer only from the numbered excerpts, must cite them, and must say when the excerpts do not cover the question. A server-side fallback model handles a refusal.

## Eval

`eval/make_questions.py` samples documentation sections and has Claude (Opus 5.5) write the question a developer would type into a search box. The prompt forbids copying the heading or any distinctive phrase verbatim, so keyword search cannot win by string matching. The questions are generated once and committed in `eval/questions.jsonl`, so every run is scored against the same set.

- **Release notes are excluded from question sampling.** In a first pass they made up 35 of 100 questions: hundreds of tiny version entries, many restating tutorial content, which makes the gold label ambiguous. They stay in the index as distractors.
- **Limitation.** Some questions are also answered by a second section (the docs repeat themselves), so these recall numbers are a lower bound.

## Run it

```bash
createdb docrag                                   # needs Postgres with the pgvector extension
uv sync
uv run python -m docs_rag.ingest                  # chunk + embed + load (about 5 min on CPU)
uv run python eval/run_eval.py                    # reproduce the table
echo 'ANTHROPIC_API_KEY=...' > .env               # gitignored; only needed for /ask
uv run --env-file .env uvicorn docs_rag.api:app
curl 'localhost:8000/search?q=how+do+I+make+a+query+parameter+optional'
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' -d '{"question":"How do I return a 201 status code?"}'
uv run pytest
```

### See what a query retrieves

`docs-rag-search` prints the chunks a query pulls back and their scores, so you can see why an answer came out the way it did. `--compare` runs all four methods on the same query:

```text
$ uv run docs-rag-search "how do I make a query parameter optional" -k 3 --compare
vector (3 hits)
   1    0.8333  tutorial/query-params.md  §  Query Parameters > Optional parameters
   2    0.8180  tutorial/query-params.md  §  Query Parameters > Required query parameters
   3    0.8100  tutorial/query-params-str-validations.md  §  Query Parameters and String Validations > Alternative (old): `Query` as the default value

keyword (3 hits)
   1    0.0222  tutorial/query-params-str-validations.md  §  Query Parameters and String Validations > Alternative (old): `Query` as the default value
   2    0.0169  tutorial/query-params.md  §  Query Parameters > Required query parameters
   3    0.0045  tutorial/body.md  §  Request Body > Create your data model

hybrid (3 hits)
   1    0.0323  tutorial/query-params-str-validations.md  §  Query Parameters and String Validations > Alternative (old): `Query` as the default value
   2    0.0323  tutorial/query-params.md  §  Query Parameters > Required query parameters
   3    0.0303  tutorial/query-params-str-validations.md  §  Query Parameters and String Validations > Default values

hybrid_rerank (3 hits)
   1    8.7663  tutorial/query-params.md  §  Query Parameters > Optional parameters
   2    8.5362  tutorial/query-params.md  §  Query Parameters > Required query parameters
   3    8.2423  tutorial/query-params-str-validations.md  §  Query Parameters and String Validations > Alternative (old): `Query` as the default value
```

The section that actually answers the question ("Optional parameters") is missing from keyword search, and fusion drops it out of the top 3 because it only ranked well on one side. The cross-encoder reads the question and each chunk together and puts it back at #1. Scores are not comparable across methods: cosine similarity, `ts_rank_cd`, RRF and reranker logits are on different scales. Add `--text` to print the start of each chunk, `--config large` for the 2,000-character chunks.

## Corpus

`corpus/` is the English FastAPI documentation at tag `0.143.1`, copyright Sebastián Ramírez, used under the MIT License (`corpus/LICENSE-fastapi`). This project is not affiliated with FastAPI.
