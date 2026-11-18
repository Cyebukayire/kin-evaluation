"""Coverage check for Algorithm 2: does the corpus contain what AfriMMLU questions need?

Samples a fixed set of AfriMMLU test questions (default: 10 per subject), retrieves the top-k BM25
passages for each (at most one per article) from passages.jsonl, and writes a spreadsheet to judge by hand. For each passage,
fill in `helpful` with yes / partly / no: would this passage help someone answer the question?

    python scripts/coverage_check.py /local/scratch/$USER/data/corpus/wiki_kinnews/passages.jsonl \
        results/coverage/wiki_kinnews_afrimmlu.xlsx

The query is the question plus its four choices, which is what the LLM prompt contains in
Algorithm 2. Requires the AfriMMLU split downloaded with: python scripts/download_data.py afrimmlu
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def tokenize(text: str) -> list[str]:
    """Lowercase, split on anything that is not a letter or digit (so n'abaturage -> n, abaturage)."""
    return re.findall(r"[^\W_]+", text.lower())


def sample_questions(questions: list[dict], per_subject: int, seed: int) -> list[dict]:
    by_subject = defaultdict(list)
    for q in questions:
        by_subject[q["meta"].get("subject") or "unknown"].append(q)
    rng = random.Random(seed)
    picked = []
    for subject in sorted(by_subject):
        qs = by_subject[subject]
        picked += rng.sample(qs, min(per_subject, len(qs)))
    return picked


def write_sheet(rows: list[dict], path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "judgments"
    cols = [("qid", 18), ("subject", 18), ("question", 50), ("choices", 40), ("answer", 20),
            ("rank", 6), ("score", 8), ("source", 11), ("title", 22), ("passage", 80), ("helpful", 10), ("notes", 30)]
    ws.append([c for c, _ in cols])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for i, (_, width) in enumerate(cols, start=1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = width
    for r in rows:
        ws.append([r.get(c, "") for c, _ in cols])
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    dv = DataValidation(type="list", formula1='"yes,partly,no"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"K2:K{len(rows) + 1}")
    ws.freeze_panes = "C2"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("passages", help="passages.jsonl from build_corpus.py")
    ap.add_argument("output", help="spreadsheet to write (.xlsx)")
    ap.add_argument("--per-subject", type=int, default=10)
    ap.add_argument("--k", type=int, default=5, help="passages per question")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--questions", help="optional .jsonl of questions (default: AfriMMLU test split)")
    args = ap.parse_args()

    if args.questions:
        questions = [json.loads(l) for l in open(args.questions, encoding="utf-8") if l.strip()]
    else:
        from kin_eval.data import load
        questions = load("afrimmlu", "test")
    picked = sample_questions(questions, args.per_subject, args.seed)

    passages = [json.loads(l) for l in open(args.passages, encoding="utf-8") if l.strip()]
    bm25 = BM25Okapi([tokenize(f"{p['title']} {p['text']}") for p in passages])

    rows = []
    for q in picked:
        query = q["question"] + " " + " ".join(q.get("choices") or [])
        scores = bm25.get_scores(tokenize(query))
        top, used = [], set()
        for i in sorted(range(len(scores)), key=lambda i: -scores[i]):
            doc = (passages[i].get("source"), passages[i]["doc_id"])
            if doc not in used:          # one passage per article, so the k passages are k different sources
                top.append(i)
                used.add(doc)
            if len(top) == args.k:
                break
        for rank, i in enumerate(top, start=1):
            p = passages[i]
            rows.append({"qid": q["id"], "subject": q["meta"].get("subject"), "question": q["question"],
                         "choices": "\n".join(f"{'ABCD'[j]}) {c}" for j, c in enumerate(q.get("choices") or [])),
                         "answer": q["answer"], "rank": rank, "score": round(float(scores[i]), 2),
                         "source": p.get("source", ""), "title": p["title"], "passage": p["text"], "pid": p["pid"]})
    write_sheet(rows, Path(args.output))
    print(f"{len(picked)} questions x top-{args.k} = {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()