"""Robust wins (A), robust "lost in Kinyarwanda" (B) and retrieval flips with McNemar tests (D), AfriMMLU.

Builds on error_analysis.py: its loader and parser are imported, not rewritten.

"Robust" means the same outcome under all 5 IrokoBench prompts. With 4 options a model guesses right 25% of the
time, so a question right under one prompt can be luck; right under 5 of 5 (and wrong under 5 of 5 for the
other models) is not.

Writes to results/analysis/ (AfriMMLU text only; safe to commit):
  robust_groups.csv            AfriMMLU (kin): questions solved robustly by all / none / only one model
  robust_wins_cases.csv        the questions behind robust_groups, with each model's k/5 and answer
  robust_lost_in_kin.csv       per model: right under 5/5 prompts in English, wrong under 5/5 in Kinyarwanda
  robust_lost_in_kin_cases.csv those questions (Kinyarwanda and English text)
  rag_flips.csv                per retrieval run and prompt: no-retrieval vs retrieval, flips, McNemar p
  rag_flips_by_subject.csv     per retrieval run and subject: mean accuracy over the 5 prompts, both ways
Writes to results/retrieval/analysis/ (gitignored: contains retrieved KINNEWS text):
  rag_flip_cases_<run>.csv     robust hurts and helps, with the passages that were in the prompt

    python analysis/robust_and_rag.py --models gemma2-27b gemma4-31b qwen38-27b
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import re
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_analysis import OUT, ROOT, acc, best_prompt, load_runs, parse, write_csv  # noqa: E402

RAG = ROOT / "results" / "raw" / "rag"
CASES_OUT = ROOT / "results" / "retrieval" / "analysis"
RAG_RE = re.compile(r"samples_afrimmlu_rag_(kin|eng)_prompt_(\d)_")
LETTERS = "ABCD"


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test: b, c = the two kinds of disagreement."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def k_of_5(prompts: dict) -> list[int]:
    """Per question: under how many prompts it was answered correctly."""
    n = len(next(iter(prompts.values())))
    return [sum(prompts[p][i]["correct"] for p in prompts) for i in range(n)]


def check_complete(name: str, prompts: dict, n: int | None = None) -> int:
    assert sorted(prompts) == [1, 2, 3, 4, 5], f"{name}: prompts {sorted(prompts)}, expected 1-5"
    lens = {len(v) for v in prompts.values()}
    assert len(lens) == 1, f"{name}: prompts have different question counts {lens}"
    if n is not None:
        assert lens == {n}, f"{name}: {lens.pop()} questions, expected {n}"
    return lens.pop()


def load_rag_runs() -> dict:
    """rag[(setting, model, lang)][prompt] = records ordered by doc_id. Prefers the committed stripped samples."""
    found = defaultdict(dict)
    for f in sorted(glob.glob(str(RAG / "*" / "*" / "*" / "*" / "*samples_afrimmlu_rag_*.jsonl.gz"))):
        p = Path(f)
        m = RAG_RE.search(p.name)
        if not m:
            continue
        key = (p.parts[-5], p.parts[-4], m.group(1))
        prompt = int(m.group(2))
        if p.name.startswith("stripped_") or prompt not in found[key]:
            found[key][prompt] = f
    rag = {}
    for key, files in found.items():
        rag[key] = {}
        for prompt, f in files.items():
            recs = {}
            for line in gzip.open(f, "rt", encoding="utf-8"):
                s = json.loads(line)
                recs[s["doc_id"]] = parse("afrimmlu", s)
            rag[key][prompt] = [recs[k] for k in sorted(recs)]
    return rag


def question_row(r: dict) -> dict:
    return {"id": r["id"], "subject": r.get("subject"), "question": r["question"],
            "choices": " | ".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"])),
            "gold": LETTERS[r["gold"]]}


def robust_wins(runs, models) -> None:
    kin = {m: runs[(m, "kin", "afrimmlu")] for m in models}
    n = check_complete(models[0], kin[models[0]])
    for m in models:
        check_complete(m, kin[m], n)
    k = {m: k_of_5(kin[m]) for m in models}
    best = {m: best_prompt(kin[m]) for m in models}
    ref = kin[models[0]][1]
    for m in models:  # same questions in the same order for every model
        assert all(kin[m][1][i]["question"] == ref[i]["question"] for i in range(n)), f"{m}: question order differs"

    groups, cases = defaultdict(int), []
    for i in range(n):
        full = [m for m in models if k[m][i] == 5]
        zero = [m for m in models if k[m][i] == 0]
        if len(full) == len(models):
            g = "robust_all"
        elif len(zero) == len(models):
            g = "robust_none"
        elif len(full) == 1 and len(zero) == len(models) - 1:
            g = f"robust_only_{full[0]}"
        else:
            continue
        groups[g] += 1
        row = {"group": g} | question_row(ref[i])
        for m in models:
            row[f"{m}_k_of_5"] = k[m][i]
            row[f"{m}_answer_best_prompt"] = LETTERS[kin[m][best[m]][i]["pred"]]
        cases.append(row)
    order = ["robust_all", "robust_none"] + [f"robust_only_{m}" for m in models]
    write_csv("robust_groups.csv", [{"group": g, "n": groups[g], "of": n} for g in order])
    write_csv("robust_wins_cases.csv", sorted(cases, key=lambda r: (order.index(r["group"]), r["id"])))


def robust_lost_in_kin(runs, models) -> None:
    summary, cases = [], []
    for m in models:
        kin, eng = runs.get((m, "kin", "afrimmlu")), runs.get((m, "eng", "afrimmlu"))
        if not kin or not eng:
            print(f"skip {m}: needs both kin and eng AfriMMLU samples")
            continue
        n = check_complete(f"{m} kin", kin)
        check_complete(f"{m} eng", eng, n)
        # the two languages are parallel translations: the gold answer and subject must agree question by question
        agree = sum(kin[1][i]["gold"] == eng[1][i]["gold"] and kin[1][i].get("subject") == eng[1][i].get("subject")
                    for i in range(n))
        assert agree == n, f"{m}: kin and eng questions not aligned ({agree}/{n} agree on gold and subject)"
        kk, ke = k_of_5(kin), k_of_5(eng)
        lost = [i for i in range(n) if ke[i] == 5 and kk[i] == 0]
        gained = [i for i in range(n) if kk[i] == 5 and ke[i] == 0]
        summary.append({"model": m, "n": n, "eng_robust_correct": sum(x == 5 for x in ke),
                        "robust_lost_in_kin": len(lost), "robust_kin_only": len(gained),
                        "lost_pct_of_eng_robust": round(100 * len(lost) / max(1, sum(x == 5 for x in ke)), 1)})
        bk, be = best_prompt(kin), best_prompt(eng)
        for i in lost:
            row = {"model": m} | question_row(kin[1][i])
            row |= {"question_eng": eng[1][i]["question"],
                    "choices_eng": " | ".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(eng[1][i]["choices"])),
                    "kin_answer_best_prompt": LETTERS[kin[bk][i]["pred"]]}
            cases.append(row)
    write_csv("robust_lost_in_kin.csv", summary)
    write_csv("robust_lost_in_kin_cases.csv", cases)


def load_retrieval(setting: str) -> dict:
    """question text -> stored passages, from the retrieval file the run used (results/retrieval/<retriever>/)."""
    retriever = setting.split("-k")[0]
    f = ROOT / "results" / "retrieval" / retriever / "afrimmlu_kin.jsonl"
    if not f.exists():
        print(f"  (no {f.relative_to(ROOT)}: cases written without passages)")
        return {}
    out = {}
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if r.get("split") == "test":
            out[r["question"]] = r["passages"]
    return out


def rag_flips(runs, rag) -> None:
    flips, by_subject = [], []
    CASES_OUT.mkdir(parents=True, exist_ok=True)
    for (setting, model, lang) in sorted(rag):
        base = runs.get((model, lang, "afrimmlu"))
        if not base:
            print(f"skip {setting}/{model}: no no-retrieval baseline samples")
            continue
        r = rag[(setting, model, lang)]
        n = check_complete(f"{setting} {model}", r)
        check_complete(f"{model} baseline", base, n)
        assert all(base[1][i]["question"] == r[1][i]["question"] for i in range(n)), \
            f"{setting}: questions not aligned with the baseline"
        name = f"{setting}/{model}/{lang}"
        for p in range(1, 6):
            b = sum(base[p][i]["correct"] and not r[p][i]["correct"] for i in range(n))
            c = sum(r[p][i]["correct"] and not base[p][i]["correct"] for i in range(n))
            flips.append({"run": name, "prompt": p, "n": n, "acc_no_retrieval": round(acc(base[p]), 1),
                          "acc_retrieval": round(acc(r[p]), 1), "right_only_without": b, "right_only_with": c,
                          "mcnemar_p": round(mcnemar_p(b, c), 4), "robust_hurt": "", "robust_helped": ""})
        kb, kr = k_of_5(base), k_of_5(r)
        # pooled over prompts: every (question, prompt) pair, for one overall test per run
        b = sum(base[p][i]["correct"] and not r[p][i]["correct"] for p in range(1, 6) for i in range(n))
        c = sum(r[p][i]["correct"] and not base[p][i]["correct"] for p in range(1, 6) for i in range(n))
        hurt = [i for i in range(n) if kb[i] >= 4 and kr[i] <= 1]
        helped = [i for i in range(n) if kr[i] >= 4 and kb[i] <= 1]
        flips.append({"run": name, "prompt": "all (pooled)", "n": 5 * n,
                      "acc_no_retrieval": round(sum(kb) / (5 * n) * 100, 1),
                      "acc_retrieval": round(sum(kr) / (5 * n) * 100, 1),
                      "right_only_without": b, "right_only_with": c, "mcnemar_p": round(mcnemar_p(b, c), 4),
                      "robust_hurt": len(hurt), "robust_helped": len(helped)})
        subj = defaultdict(list)
        for i in range(n):
            subj[base[1][i].get("subject")].append(i)
        for s, idx in sorted(subj.items()):
            by_subject.append({"run": name, "subject": s, "n": len(idx),
                               "acc_no_retrieval": round(100 * sum(kb[i] for i in idx) / (5 * len(idx)), 1),
                               "acc_retrieval": round(100 * sum(kr[i] for i in idx) / (5 * len(idx)), 1)})
        # cases, with the passages that were in the prompt (k of them; none for skipped subjects)
        k = int(re.search(r"-k(\d+)", setting).group(1))
        skipped = setting.split("-skip-")[1].split(",") if "-skip-" in setting else []
        passages = load_retrieval(setting)
        rows = []
        for kind, idx in (("hurt", hurt), ("helped", helped)):
            for i in idx:
                q = base[1][i]
                row = {"kind": kind} | question_row(q)
                row |= {"k_of_5_without": kb[i], "k_of_5_with": kr[i],
                        "answer_without_p1": LETTERS[base[1][i]["pred"]], "answer_with_p1": LETTERS[r[1][i]["pred"]]}
                ps = [] if q.get("subject") in skipped else passages.get(q["question"], [])[:k]
                for j in range(k):
                    row[f"passage_{j + 1}"] = f"[{ps[j]['source']}] {ps[j]['title']}: {ps[j]['text']}" if j < len(ps) else ""
                rows.append(row)
        f = CASES_OUT / f"rag_flip_cases_{setting}_{model}_{lang}.csv"
        with open(f, "w", newline="", encoding="utf-8") as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        print(f"wrote {f.relative_to(ROOT)} ({len(hurt)} hurt, {len(helped)} helped)")
    write_csv("rag_flips.csv", flips)
    write_csv("rag_flips_by_subject.csv", by_subject)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="models with complete sample logs")
    args = ap.parse_args()
    runs = load_runs(args.models)
    robust_wins(runs, args.models)
    robust_lost_in_kin(runs, args.models)
    rag_flips(runs, load_rag_runs())


if __name__ == "__main__":
    main()
