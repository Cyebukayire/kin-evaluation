"""Step 1 of the KinyaColBERT pipeline: write the plain text files MorphoKIN will parse.

KinyaColBERT (c4ir-rw/ac-ai-models) reads aligned files: line n of *_id.txt and line n of the
parsed *_text.txt describe the same item. This script writes the plain (not yet parsed) files,
in the same layout as the kinya-ag-retrieval dataset, into KCB_WORK_DIR (configs/paths.env):

  corpus_passage_id.txt              one passage ID per line      (wiki-775-0, news-train-12-1, ...)
  corpus_passage_text.txt            "title : text", one passage per line
  afrimmlu_query_id.txt              one question ID per line     (afrimmlu-kin-test-0, ...-validation-0, ...)
  afrimmlu_query_text.txt            question + its 4 choices     (the same query BM25 used)
  export_stats.json                  counts, settings and checksums of inputs and outputs

Queries cover AfriMMLU test then validation (583), in the order BM25 used. Tabs and line breaks
inside a text become spaces, so every item stays on exactly one line (MorphoKIN reads one item per
line and writes tabs between words). Nothing else in the text changes.

    python scripts/kinyacolbert/export_texts.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kin_eval.paths import get_path

SPLITS = ["test", "validation"]


def one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_lines(path: Path, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        if not line:
            raise ValueError(f"{path.name}: line {i + 1} is empty; MorphoKIN post-processing drops empty lines")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    corpus_file = get_path("CORPUS_DIR") / "passages.jsonl"
    afrimmlu_dir = get_path("AFRIMMLU_DIR")
    out = get_path("KCB_WORK_DIR")
    out.mkdir(parents=True, exist_ok=True)

    passages = [json.loads(l) for l in open(corpus_file, encoding="utf-8") if l.strip()]
    pid = [p["pid"] for p in passages]
    ptext = [one_line(f"{p['title']} : {p['text']}") for p in passages]
    if len(set(pid)) != len(pid):
        raise ValueError("duplicate passage IDs in the corpus")

    qid, qfull = [], []
    inputs = {corpus_file.name: sha256(corpus_file)}
    for split in SPLITS:
        f = afrimmlu_dir / f"{split}.jsonl"
        inputs[f"afrimmlu/{f.name}"] = sha256(f)
        for q in (json.loads(l) for l in open(f, encoding="utf-8") if l.strip()):
            qid.append(q["id"].replace("afrimmlu-kin-", f"afrimmlu-kin-{split}-"))  # test and validation reuse indices
            qfull.append(one_line(q["question"] + " " + " ".join(q.get("choices") or [])))
    if len(set(qid)) != len(qid):
        raise ValueError("duplicate query IDs")

    files = {"corpus_passage_id.txt": pid, "corpus_passage_text.txt": ptext,
             "afrimmlu_query_id.txt": qid, "afrimmlu_query_text.txt": qfull}
    for name, lines in files.items():
        write_lines(out / name, lines)

    stats = {"passages": len(pid), "queries": len(qid), "query_splits": SPLITS,
             "passage_text": "title : text", "query_text": "question + choices (same as BM25)",
             "inputs_sha256": inputs,
             "outputs_sha256": {name: sha256(out / name) for name in files}}
    (out / "export_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()