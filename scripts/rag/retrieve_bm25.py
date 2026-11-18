"""BM25 retrieval for Algorithm 2: top passages for every AfriMMLU question (test + validation).

Writes one line per question, the format every retriever (BM25, AfriE5, KinyaColBERT) produces
and the retrieval-augmented lm-eval task (tasks/afrimmlu_rag) reads:

    {"question": "...", "split": "test", "id": "afrimmlu-kin-0", "subject": "...",
     "passages": [{"pid": "...", "source": "wikipedia", "title": "...", "text": "...", "score": 31.2}, ...]}

Questions are matched to lm-eval's documents by their question text. --keep passages are stored
(more than the prompt uses), so the number of passages k can be varied later without retrieving again.

    python scripts/rag/retrieve_bm25.py /local/scratch/$USER/data/corpus/wiki_kinnews/passages.jsonl \
        results/retrieval/bm25/afrimmlu_kin.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from coverage_check import tokenize  # same tokenizer as the coverage check


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("passages", help="passages.jsonl from build_corpus.py")
    ap.add_argument("output", help="retrieval file to write (.jsonl)")
    ap.add_argument("--splits", nargs="+", default=["test", "validation"])
    ap.add_argument("--keep", type=int, default=10, help="passages stored per question")
    args = ap.parse_args()

    from kin_eval.data import load

    passages = [json.loads(l) for l in open(args.passages, encoding="utf-8") if l.strip()]
    bm25 = BM25Okapi([tokenize(f"{p['title']} {p['text']}") for p in passages])
    print(f"indexed {len(passages)} passages")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for split in args.splits:
            for q in load("afrimmlu", split):
                query = q["question"] + " " + " ".join(q.get("choices") or [])
                scores = bm25.get_scores(tokenize(query))
                top, used = [], set()
                for i in np.argsort(-scores, kind="stable"):
                    doc = (passages[i].get("source"), passages[i]["doc_id"])
                    if doc in used:          # one passage per article, so the passages are k different sources
                        continue
                    used.add(doc)
                    p = passages[i]
                    top.append({"pid": p["pid"], "source": p.get("source"), "title": p["title"],
                                "text": p["text"], "score": round(float(scores[i]), 3)})
                    if len(top) == args.keep:
                        break
                f.write(json.dumps({"question": q["question"], "split": split, "id": q["id"],
                                    "subject": q["meta"].get("subject"), "passages": top},
                                   ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} questions -> {out}")


if __name__ == "__main__":
    main()