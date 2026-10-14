"""Build the retrieval eval set: one Claude-written question per sampled docs section.

The section a question was written from is its gold label. Run once and commit the output,
so every later retrieval run is scored against the same frozen questions:

    uv run --env-file .env python eval/make_questions.py
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from pydantic import BaseModel

from docs_rag.chunking import load_corpus

N = 100
SEED = 7
MODEL = "claude-opus-5-5"
ROOT = Path(__file__).resolve().parents[1]

PROMPT = """Below is one section of the FastAPI documentation.

Write ONE question a developer might type into a docs search box that this section answers well.
Rules:
- Phrase it the way a user would ask it, in your own words. Do not copy the section heading or
  any distinctive phrase from the section verbatim.
- It must be answerable from this section alone.
- If the section is only navigation, a link list, or too thin to answer anything, set answerable to false.

Section heading: {heading}
Document: {doc}

{text}"""


class Question(BaseModel):
    question: str
    answerable: bool


def main() -> None:
    # Release notes are hundreds of tiny version entries that would dominate a random sample and often
    # restate tutorial content (ambiguous gold). They stay in the index as distractors, not as questions.
    secs = [s for s in load_corpus(ROOT / "corpus") if len(s.text) >= 300 and s.doc != "release-notes.md"]
    random.Random(SEED).shuffle(secs)
    client = anthropic.Anthropic()

    def ask(s):
        r = client.messages.parse(
            model=MODEL,
            max_tokens=2000,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": PROMPT.format(heading=s.heading, doc=s.doc, text=s.text[:6000])}],
            output_format=Question,
        )
        if r.stop_reason != "end_turn" or r.parsed_output is None:
            return None
        return s, r.parsed_output

    out = []
    with ThreadPoolExecutor(8) as pool:
        for res in pool.map(ask, secs[: int(N * 1.3)]):
            if res and res[1].answerable and len(out) < N:
                s, q = res
                out.append({"id": f"q{len(out) + 1:03d}", "question": q.question, "gold": s.key})
    path = ROOT / "eval" / "questions.jsonl"
    path.write_text("".join(json.dumps(o) + "\n" for o in out))
    print(f"wrote {len(out)} questions to {path}")


if __name__ == "__main__":
    main()
