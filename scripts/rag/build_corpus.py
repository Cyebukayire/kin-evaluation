"""Build the retrieval corpus (Algorithm 2) from Kinyarwanda Wikipedia and/or KINNEWS.

Inputs:
  --wiki DIR       WikiExtractor --json output (wiki_00, wiki_01, ...; one JSON article per line)
  --kinnews CSV    KINNEWS raw split(s) (columns: label, kin_label, en_label, url, title, content);
                   give train.csv and test.csv together. Use the RAW version: the cleaned one has
                   punctuation and stopwords removed, which an LLM cannot read well.

Output: one passage per line, ready for BM25 / AfriE5 / KinyaColBERT:

    {"pid": "wiki-775-0", "source": "wikipedia", "doc_id": "775", "title": "Rwanda", "url": "...",
     "text": "...", "n_words": 112}

plus stats.json recording what was kept and dropped per source, for the write-up.

    python scripts/build_corpus.py --wiki Wikipedia_rw \
        --kinnews /local/scratch/$USER/data/kinnews/train.csv /local/scratch/$USER/data/kinnews/test.csv \
        --out /local/scratch/$USER/data/corpus/wiki_kinnews
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ENGLISH_STOPWORDS = set("the of and is in to was for that with as by on are from it an be this which".split())


def english_fraction(text: str) -> float:
    words = re.findall(r"[a-z]+", text.lower())
    return sum(w in ENGLISH_STOPWORDS for w in words) / max(1, len(words))


def clean_lines(title: str, text: str) -> list[str]:
    """Article text as a list of non-empty lines, without the '= Title =' header WikiExtractor adds."""
    text = html.unescape(html.unescape(text))             # &amp;lt; -> &lt; -> <
    text = re.sub(r"&\s*nbsp;", " ", text)                 # "&amp; nbsp;" leftovers
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and not re.fullmatch(r"=+\s*.*?\s*=+", ln)]
    return lines


def chunk(lines: list[str], target: int, max_words: int) -> list[str]:
    """Split an article into passages of about `target` words (never more than max_words).

    Cuts at the sentence end closest to `target` words, looking between 0.7 * target and max_words;
    without one, cuts at max_words. A final piece shorter than target / 3 is merged with the one before,
    and the two are split evenly so neither exceeds max_words.
    """
    words = " ".join(lines).split()
    passages, start = [], 0
    while start < len(words):
        if len(words) - start <= max_words:
            passages.append(words[start:])
            break
        lo, hi = start + int(0.7 * target), start + max_words
        ends = [i + 1 for i in range(lo, hi) if words[i].endswith((".", "!", "?"))]
        cut = min(ends, key=lambda e: abs(e - start - target)) if ends else hi
        passages.append(words[start:cut])
        start = cut
    if len(passages) > 1 and len(passages[-1]) < target // 3:
        merged = passages[-2] + passages[-1]
        half = len(merged) // 2
        passages[-2:] = [merged[:half], merged[half:]] if len(merged) > max_words else [merged]
    return [" ".join(p) for p in passages]


def wiki_docs(folder: str):
    files = sorted(glob.glob(str(Path(folder) / "wiki_*")))
    if not files:
        raise SystemExit(f"no wiki_* files in {folder}")
    for f in files:
        for line in open(f, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                yield {"source": "wikipedia", "doc_id": str(d["id"]), "title": d["title"], "url": d.get("url"),
                       "lines": clean_lines(d["title"], d["text"])}


def kinnews_docs(paths: list[str]):
    csv.field_size_limit(10**8)
    for path in paths:
        split = Path(path).stem
        with open(path, encoding="utf-8", newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                text = (row.get("content") or "").strip()
                yield {"source": "kinnews", "doc_id": f"{split}-{i}", "title": (row.get("title") or "").strip(),
                       "url": row.get("url"), "category": row.get("en_label"),
                       "lines": [ln.strip() for ln in text.split("\n") if ln.strip()]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wiki", help="folder with WikiExtractor --json files (wiki_*)")
    ap.add_argument("--kinnews", nargs="*", default=[], help="KINNEWS raw CSV file(s)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--min-words", type=int, default=20, help="drop documents shorter than this")
    ap.add_argument("--target-words", type=int, default=100, help="approximate passage length")
    ap.add_argument("--max-words", type=int, default=150, help="hard upper bound on passage length")
    ap.add_argument("--max-english", type=float, default=0.15,
                    help="drop documents whose share of English function words is above this")
    args = ap.parse_args()
    if not args.wiki and not args.kinnews:
        raise SystemExit("give --wiki and/or --kinnews")

    sources = []
    if args.wiki:
        sources.append(wiki_docs(args.wiki))
    if args.kinnews:
        sources.append(kinnews_docs(args.kinnews))

    stats = defaultdict(lambda: {"documents_in": 0, "dropped": Counter(), "documents_kept": 0,
                                 "passages": 0, "words": 0})
    seen_texts = set()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "passages.jsonl").open("w", encoding="utf-8") as out:
        for docs in sources:
            for doc in docs:
                st = stats[doc["source"]]
                st["documents_in"] += 1
                body = " ".join(doc["lines"])
                if len(body.split()) < args.min_words:
                    st["dropped"]["too_short"] += 1
                    continue
                key = " ".join(body.lower().split())
                if key in seen_texts:
                    st["dropped"]["duplicate"] += 1
                    continue
                seen_texts.add(key)
                if english_fraction(body) > args.max_english:
                    st["dropped"]["mostly_english"] += 1
                    continue
                st["documents_kept"] += 1
                prefix = "wiki" if doc["source"] == "wikipedia" else "news"
                for i, text in enumerate(chunk(doc["lines"], args.target_words, args.max_words)):
                    n = len(text.split())
                    rec = {"pid": f"{prefix}-{doc['doc_id']}-{i}", "source": doc["source"], "doc_id": doc["doc_id"],
                           "title": doc["title"], "url": doc["url"], "text": text, "n_words": n}
                    if doc.get("category"):
                        rec["category"] = doc["category"]
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    st["passages"] += 1
                    st["words"] += n

    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    wiki_files = sorted(glob.glob(str(Path(args.wiki) / "wiki_*"))) if args.wiki else []
    report = {"inputs": {"wiki": args.wiki, "kinnews": args.kinnews,
                         "sha256": {**{Path(f).name: sha256(f) for f in wiki_files},
                                    **{p: sha256(p) for p in args.kinnews}}},
              "settings": {k: getattr(args, k) for k in ("min_words", "target_words", "max_words", "max_english")},
              "sources": {}}
    for name, st in stats.items():
        st["dropped"] = dict(st["dropped"])
        st["mean_passage_words"] = round(st["words"] / max(1, st["passages"]), 1)
        report["sources"][name] = st
    report["total_passages"] = sum(st["passages"] for st in stats.values())
    (out_dir / "stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()