"""KinyaColBERT retrieval for Algorithm 2: top passages for every AfriMMLU question (test + validation).

Thin wrapper around the authors' code (DeepKIN-AgAI, scripts/eval/colbert/agai_eval_kinya_col_bert_mrr.py).
Model loading, trim(508), get_colbert_embeddings, the mean/std standardization, L2 normalization,
pairwise_score and the division by Q.size(1) are theirs. Two changes are for speed only, and give the same numbers:
  1. the mean/std pool is concatenated once at the end, not once per passage (their line 52);
  2. passages are normalized once and scored in chunks with their model.pairwise_score, not one pair at a time
     (their lines 94-99). Shorter passages in a chunk are padded by repeating their last token, which cannot change
     a max. A built-in check scores the first question against 200 passages with their exact one-by-one loop and
     stops if the two disagree.

Output: the same format as retrieve_bm25.py (one line per question, one passage per article), which the
retrieval-augmented lm-eval task (tasks/afrimmlu_rag) reads.

    python scripts/kinyacolbert/retrieve_kinyacolbert.py results/retrieval/kinyacolbert/afrimmlu_kin.jsonl
    python scripts/kinyacolbert/retrieve_kinyacolbert.py results/retrieval/kinyacolbert/smoke_5q.jsonl --limit 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from kin_eval.paths import get_path  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", help="retrieval file to write (.jsonl)")
    ap.add_argument("--work-dir", default=None, help="folder with the id and parsed text files (default: KCB_WORK_DIR)")
    ap.add_argument("--passages", default=None, help="passages.jsonl from build_corpus.py (default: CORPUS_DIR/passages.jsonl)")
    ap.add_argument("--ckpt", default=None, help="model file (default: KINYACOLBERT_CKPT)")
    ap.add_argument("--splits", nargs="+", default=["test", "validation"])
    ap.add_argument("--keep", type=int, default=10, help="passages stored per question")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N questions (smoke test)")
    ap.add_argument("--chunk", type=int, default=256, help="passages scored together")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = ap.parse_args()

    work = Path(args.work_dir or get_path("KCB_WORK_DIR"))
    passages_file = Path(args.passages or Path(get_path("CORPUS_DIR")) / "passages.jsonl")
    ckpt = Path(args.ckpt or get_path("KINYACOLBERT_CKPT"))
    out = Path(args.output)
    if not out.is_absolute():
        out = REPO / out
    if out.exists() and not args.force:
        sys.exit(f"refusing to overwrite {out} (pass --force)")

    sys.path.insert(0, str(get_path("DEEPKIN_AGAI")))
    import torch
    import torch.nn.functional as F
    from deepkin.clib.libkinlp.kinlpy import ParsedFlexSentence
    from deepkin.data.morpho_qa_triple_data import DOCUMENT_TYPE_ID, QUESTION_TYPE_ID
    from deepkin.models.kinyabert import KinyaColBERT
    from deepkin.utils.misc_functions import read_lines
    from kin_eval.data import load

    files = {
        "passage_id": work / "corpus_passage_id.txt",
        "passage_text": work / "parsed_corpus_passage_text.txt",
        "query_id": work / "afrimmlu_query_id.txt",
        "query_text": work / "parsed_afrimmlu_query_text.txt",
    }
    t0 = time.time()

    # --- inputs and the links between them -------------------------------------------------------------
    pids, ptexts = read_lines(str(files["passage_id"])), read_lines(str(files["passage_text"]))
    qids, qtexts = read_lines(str(files["query_id"])), read_lines(str(files["query_text"]))
    assert len(pids) == len(ptexts), f"passage ids {len(pids)} vs parsed texts {len(ptexts)}"
    assert len(qids) == len(qtexts), f"query ids {len(qids)} vs parsed texts {len(qtexts)}"
    meta = {}
    for line in open(passages_file, encoding="utf-8"):
        if line.strip():
            p = json.loads(line)
            meta[p["pid"]] = p
    missing = [x for x in pids if x not in meta]
    assert not missing, f"{len(missing)} passage ids not in {passages_file}, e.g. {missing[:3]}"

    questions = []  # (query id, split, lm-eval question), in export order
    for split in args.splits:
        for i, q in enumerate(load("afrimmlu", split)):
            questions.append((f"afrimmlu-kin-{split}-{i}", split, q))
    assert {x[0] for x in questions} == set(qids), "AfriMMLU questions do not match afrimmlu_query_id.txt"
    print(f"Got: {len(qids)} queries, {len(pids)} passages", flush=True)

    # --- model: their lines 34-38 ------------------------------------------------------------------------
    device = torch.device("cuda:0")
    model, margs = KinyaColBERT.from_pretrained(device, str(ckpt), ret_args=True)
    model.float()
    model.eval()
    metric = model.similarity_metric
    print(f"similarity_metric={metric}, colbert_embedding_dim={margs.colbert_embedding_dim}", flush=True)

    with torch.no_grad():
        # --- passages: their lines 46-58 (pool concatenated once) -----------------------------------------
        D_list = []
        for itr, txt in enumerate(ptexts):
            passage = ParsedFlexSentence(txt)
            passage.trim(508)
            D_list.append(model.get_colbert_embeddings([passage], DOCUMENT_TYPE_ID))
            if itr % 5000 == 0:
                print(f"  passages embedded: {itr}/{len(ptexts)} ({time.time() - t0:.0f}s)", flush=True)
        pool = torch.cat([D.view(-1, D.size(-1)) for D in D_list])
        Doc_Mean, Doc_Stdev = pool.mean(dim=0), pool.std(dim=0)
        del pool

        # --- queries: their lines 61-72 (all queries, so the stats do not depend on --limit) -------------
        Q_emb = {}
        for qid, txt in zip(qids, qtexts):
            query = ParsedFlexSentence(txt)
            query.trim(508)
            Q_emb[qid] = model.get_colbert_embeddings([query], QUESTION_TYPE_ID)
        pool = torch.cat([Q.view(-1, Q.size(-1)) for Q in Q_emb.values()])
        Query_Mean, Query_Stdev = pool.mean(dim=0), pool.std(dim=0)
        del pool
        print(f"embedded everything ({time.time() - t0:.0f}s)", flush=True)

        def norm_query(qid):  # their lines 91-92
            Q = (Q_emb[qid] - Query_Mean) / Query_Stdev
            return F.normalize(Q, p=2, dim=2)

        todo = questions[: args.limit] if args.limit else questions

        # --- reference: their exact one-by-one loop (lines 94-100), first question, first 200 passages -----
        Q0 = norm_query(todo[0][0])
        ref = []
        for D in D_list[:200]:
            D = (D - Doc_Mean) / Doc_Stdev
            D = F.normalize(D, p=2, dim=2)
            ref.append(model.pairwise_score(Q0, D).squeeze().item() / Q0.size(1))
        ref = torch.tensor(ref, device=device)

        # --- normalize every passage once, then group by length into padded chunks --------------------------
        lengths = [D.size(1) for D in D_list]
        for i in range(len(D_list)):
            D_list[i] = F.normalize((D_list[i] - Doc_Mean) / Doc_Stdev, p=2, dim=2)
        order = sorted(range(len(D_list)), key=lambda i: lengths[i])
        chunks = []
        for s in range(0, len(order), args.chunk):
            idx = order[s: s + args.chunk]
            L = max(lengths[i] for i in idx)
            parts = []
            for i in idx:
                D = D_list[i]
                if D.size(1) < L:  # repeat the last token: a repeated token cannot change a max
                    D = torch.cat([D, D[:, -1:, :].expand(1, L - D.size(1), D.size(2))], dim=1)
                parts.append(D)
                D_list[i] = None
            chunks.append((torch.tensor(idx, device=device), torch.cat(parts, dim=0)))
        del D_list
        sub = args.chunk if metric == "cosine" else 1  # the l2 formula needs far more memory per passage

        def score_all(Q):  # their model.pairwise_score, then their division by Q.size(1)
            s = torch.empty(len(pids), device=device)
            for idx, Dc in chunks:
                for b in range(0, Dc.size(0), sub):
                    s[idx[b: b + sub]] = model.pairwise_score(Q, Dc[b: b + sub]).view(-1)
            return s / Q.size(1)

        check = (score_all(Q0)[:200] - ref).abs().max().item()
        print(f"check vs their one-by-one loop (200 passages): max abs difference {check:.2e}", flush=True)
        if check > 1e-4:
            sys.exit("chunked scores differ from the authors' loop; stopping")

        # --- rank and write: same rules and format as retrieve_bm25.py ------------------------------------
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for n, (qid, split, q) in enumerate(todo):
                scores = score_all(norm_query(qid))
                ranked = torch.sort(scores, descending=True, stable=True).indices.tolist()  # ties: corpus order, as in their sorted()
                top, used = [], set()
                for i in ranked:
                    p = meta[pids[i]]
                    doc = (p.get("source"), p["doc_id"])
                    if doc in used:  # one passage per article, as for BM25
                        continue
                    used.add(doc)
                    top.append({"pid": p["pid"], "source": p.get("source"), "title": p["title"],
                                "text": p["text"], "score": round(float(scores[i]), 4)})
                    if len(top) == args.keep:
                        break
                f.write(json.dumps({"question": q["question"], "split": split, "id": q["id"],
                                    "subject": q["meta"].get("subject"), "passages": top},
                                   ensure_ascii=False) + "\n")
                if n % 100 == 0:
                    print(f"  questions done: {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)

    record = {
        "output": str(out.relative_to(REPO)) if out.is_relative_to(REPO) else str(out),
        "output_sha256": sha256(out),
        "questions_written": len(todo), "keep": args.keep, "limit": args.limit, "chunk": args.chunk,
        "similarity_metric": metric, "colbert_embedding_dim": margs.colbert_embedding_dim,
        "check_max_abs_diff_vs_authors_loop": check,
        "checkpoint": ckpt.name, "checkpoint_sha256": sha256(ckpt),
        "inputs_sha256": {k: sha256(v) for k, v in files.items()} | {"passages.jsonl": sha256(passages_file)},
        "torch": torch.__version__, "seconds": round(time.time() - t0),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {len(todo)} questions -> {out} ({record['seconds']}s)")


if __name__ == "__main__":
    main()
    