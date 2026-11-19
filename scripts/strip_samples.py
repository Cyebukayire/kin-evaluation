"""Make text-free copies of lm-eval sample logs from retrieval runs, safe to commit.

Retrieval-run logs contain the passages shown to the model (inside "arguments" and doc["context"]),
and KINNEWS text cannot go in the public repo. This writes, next to each samples_*.jsonl.gz,
a stripped_samples_*.jsonl.gz with those two fields removed. Everything needed for per-question
analysis stays: doc_id, the question, choices, subject, gold answer, the model's scores, acc.

    python scripts/strip_samples.py results/raw/rag/bm25-k3/gemma2-27b/nochat
"""
import glob
import gzip
import json
import sys
from pathlib import Path


def main(run_dir: str) -> None:
    files = sorted(glob.glob(str(Path(run_dir) / "*" / "samples_*.jsonl.gz")))
    if not files:
        sys.exit(f"no samples_*.jsonl.gz under {run_dir}")
    for f in files:
        out = Path(f).with_name("stripped_" + Path(f).name)
        with gzip.open(f, "rt", encoding="utf-8") as src, gzip.open(out, "wt", encoding="utf-8") as dst:
            for line in src:
                s = json.loads(line)
                s.pop("arguments", None)
                if isinstance(s.get("doc"), dict):
                    s["doc"].pop("context", None)
                dst.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])