"""Question-level analysis of the baseline runs (performance breakdown + error analysis).

Reads the lm-eval sample logs (samples_*.jsonl.gz, committed with each run) under
results/raw/baseline/<model>/<nochat|nochat_eng>/ and writes CSV tables and a spreadsheet for manual
error coding to results/analysis/:

  summary_best_prompt.csv   best prompt per model/language/task, with accuracy and 95% CI
  afrimmlu_by_subject.csv   AfriMMLU accuracy per subject (best prompt per model and language)
  prompt_consistency.csv    share of questions solved under 5/5, some, or 0/5 prompts
  kin_vs_eng.csv            per model: solved in both, English only ("lost in Kinyarwanda"), ...
  answer_bias.csv           AfriMMLU: predicted vs gold letter distribution, shortest-choice rate
  model_overlap.csv         AfriMMLU (kin): questions solved by all / none / only one model
  afrimgsm_errors.csv       automatic AfriMGSM error categories (no number, repetition, wrong number)
  error_cases.xlsx          sampled errors with an empty `category` column to code by hand

    python analysis/error_analysis.py                      # all models found
    python analysis/error_analysis.py --models gemma2-27b gemma4-31b qwen38-27b
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import gzip
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "baseline"
OUT = ROOT / "results" / "analysis"
LANGS = {"nochat": "kin", "nochat_eng": "eng"}
TASK_RE = re.compile(r"samples_(afrimmlu|afrimgsm)(?:_direct)?_(kin|eng)_prompt_(\d)_")
MGSM_FILTER = "flexible-extract"


# ---------------------------------------------------------------- loading

def as_list(x):
    if isinstance(x, list):
        return x
    try:
        return list(ast.literal_eval(x))
    except (ValueError, SyntaxError):
        s = x.strip().strip("[]").strip()
        return [p.strip("'\"") for p in s.split("', '")]


def load_runs(models: list[str] | None) -> dict:
    """runs[(model, lang, task)][prompt] = list of per-question records, ordered by doc_id."""
    runs = defaultdict(dict)
    files = glob.glob(str(RAW / "*" / "*" / "*" / "samples_*.jsonl")) + \
        glob.glob(str(RAW / "*" / "*" / "*" / "samples_*.jsonl.gz"))
    for f in sorted(files):
        p = Path(f)
        model, mode = p.parts[-4], p.parts[-3]
        if mode not in LANGS or (models and model not in models):
            continue
        m = TASK_RE.search(p.name)
        if not m:
            continue
        task, lang, prompt = m.group(1), m.group(2), int(m.group(3))
        recs = {}
        opener = gzip.open if f.endswith(".gz") else open
        for line in opener(f, "rt", encoding="utf-8"):
            s = json.loads(line)
            if task == "afrimgsm" and s.get("filter", MGSM_FILTER) != MGSM_FILTER:
                continue
            recs[s["doc_id"]] = parse(task, s)
        runs[(model, lang, task)][prompt] = [recs[k] for k in sorted(recs)]
    if not runs:
        raise SystemExit(f"no samples_*.jsonl found under {RAW} (rerun the baselines to create them)")
    return runs


def parse(task: str, s: dict) -> dict:
    doc = s["doc"]
    if task == "afrimmlu":
        lls = [float(r[0]) for r in s["filtered_resps"]]
        pred = max(range(len(lls)), key=lls.__getitem__)
        choices = as_list(doc["choices"])
        gold = int(s["target"]) if str(s["target"]).isdigit() else "ABCD".index(str(doc["answer"]).strip())
        return {"id": s["doc_id"], "subject": doc.get("subject"), "question": doc["question"],
                "choices": choices, "gold": gold, "pred": pred, "correct": float(s.get("acc", pred == gold)) == 1.0}
    raw = s["resps"][0][0] if s.get("resps") else ""
    extracted = s["filtered_resps"][0] if s.get("filtered_resps") else ""
    return {"id": s["doc_id"], "question": doc["question"], "gold": str(s["target"]).strip(),
            "output": raw, "extracted": extracted, "correct": float(s.get("exact_match", 0)) == 1.0}


# ---------------------------------------------------------------- helpers

def acc(recs) -> float:
    return 100 * sum(r["correct"] for r in recs) / max(1, len(recs))


def ci95(recs) -> float:
    p, n = acc(recs) / 100, max(1, len(recs))
    return 100 * 1.96 * math.sqrt(p * (1 - p) / n)


def best_prompt(prompts: dict) -> int:
    return max(prompts, key=lambda k: acc(prompts[k]))


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))  # union of keys, first-seen order
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval=0)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT / name}")


def to_number(x: str):
    x = str(x).replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(x)
    except ValueError:
        return None


def mgsm_category(r: dict) -> str:
    if r["correct"]:
        return "correct"
    out = r["output"] or ""
    words = out.split()
    if not out.strip():
        return "empty_output"
    if len(words) >= 20 and len(set(words)) / len(words) < 0.2:
        return "degenerate_repetition"
    pred, gold = to_number(r["extracted"]), to_number(r["gold"])
    if pred is None:
        return "no_number_found"
    if gold is not None and gold != 0 and abs(pred - gold) / abs(gold) <= 0.1:
        return "wrong_number_close"   # within 10%: likely an arithmetic slip
    if not re.search(r"\d.*\D.*\d", out.strip()):
        return "wrong_bare_number"    # only a number, no visible working
    return "wrong_with_working"


# ---------------------------------------------------------------- analyses

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--cases-per-subject", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    runs = load_runs(args.models)
    best = {key: best_prompt(prompts) for key, prompts in runs.items()}

    write_csv("summary_best_prompt.csv", [
        {"model": m, "lang": l, "task": t, "best_prompt": best[(m, l, t)], "n": len(runs[(m, l, t)][best[(m, l, t)]]),
         "acc": round(acc(runs[(m, l, t)][best[(m, l, t)]]), 1),
         "ci95": round(ci95(runs[(m, l, t)][best[(m, l, t)]]), 1),
         "mean_over_prompts": round(sum(acc(v) for v in runs[(m, l, t)].values()) / len(runs[(m, l, t)]), 1)}
        for (m, l, t) in sorted(runs)])

    rows = []
    for (m, l, t), prompts in sorted(runs.items()):
        if t != "afrimmlu":
            continue
        by_sub = defaultdict(list)
        for r in prompts[best[(m, l, t)]]:
            by_sub[r["subject"]].append(r)
        rows += [{"model": m, "lang": l, "subject": s, "n": len(v), "acc": round(acc(v), 1)} for s, v in sorted(by_sub.items())]
    write_csv("afrimmlu_by_subject.csv", rows)

    rows = []
    for (m, l, t), prompts in sorted(runs.items()):
        solved = Counter()
        for recs in prompts.values():
            for r in recs:
                solved[r["id"]] += r["correct"]
        n_q, k = len(next(iter(prompts.values()))), len(prompts)
        rows.append({"model": m, "lang": l, "task": t, "prompts": k,
                     "all_prompts_pct": round(100 * sum(v == k for v in solved.values()) / n_q, 1),
                     "some_prompts_pct": round(100 * sum(0 < v < k for v in solved.values()) / n_q, 1),
                     "no_prompt_pct": round(100 * sum(v == 0 for v in solved.values()) / n_q, 1)})
    write_csv("prompt_consistency.csv", rows)

    rows = []
    for (m, l, t) in sorted(runs):
        if l != "kin" or (m, "eng", t) not in runs:
            continue
        kin = {r["id"]: r["correct"] for r in runs[(m, "kin", t)][best[(m, "kin", t)]]}
        eng = {r["id"]: r["correct"] for r in runs[(m, "eng", t)][best[(m, "eng", t)]]}
        ids = kin.keys() & eng.keys()
        c = Counter((eng[i], kin[i]) for i in ids)
        rows.append({"model": m, "task": t, "n": len(ids), "both_correct": c[(True, True)],
                     "eng_only_lost_in_kin": c[(True, False)], "kin_only": c[(False, True)],
                     "both_wrong": c[(False, False)],
                     "lost_pct_of_eng_correct": round(100 * c[(True, False)] / max(1, c[(True, True)] + c[(True, False)]), 1)})
    write_csv("kin_vs_eng.csv", rows)

    rows = []
    for (m, l, t), prompts in sorted(runs.items()):
        if t != "afrimmlu":
            continue
        recs = prompts[best[(m, l, t)]]
        pred, gold = Counter(r["pred"] for r in recs), Counter(r["gold"] for r in recs)
        shortest = [min(range(len(r["choices"])), key=lambda j: len(r["choices"][j])) for r in recs]
        row = {"model": m, "lang": l}
        for j, letter in enumerate("ABCD"):
            row[f"pred_{letter}_pct"] = round(100 * pred[j] / len(recs), 1)
            row[f"gold_{letter}_pct"] = round(100 * gold[j] / len(recs), 1)
        row["pred_shortest_choice_pct"] = round(100 * sum(r["pred"] == s for r, s in zip(recs, shortest)) / len(recs), 1)
        row["gold_shortest_choice_pct"] = round(100 * sum(r["gold"] == s for r, s in zip(recs, shortest)) / len(recs), 1)
        rows.append(row)
    write_csv("answer_bias.csv", rows)

    kin_mmlu = {m: {r["id"]: r for r in runs[(m, l, t)][best[(m, l, t)]]}
                for (m, l, t) in runs if l == "kin" and t == "afrimmlu"}
    if len(kin_mmlu) > 1:
        ids = set.intersection(*(set(v) for v in kin_mmlu.values()))
        solvers = {i: [m for m in kin_mmlu if kin_mmlu[m][i]["correct"]] for i in ids}
        rows = [{"group": "solved_by_all", "n": sum(len(s) == len(kin_mmlu) for s in solvers.values())},
                {"group": "solved_by_none", "n": sum(len(s) == 0 for s in solvers.values())}]
        rows += [{"group": f"only_{m}", "n": sum(s == [m] for s in solvers.values())} for m in kin_mmlu]
        write_csv("model_overlap.csv", rows)

    rows = []
    for (m, l, t), prompts in sorted(runs.items()):
        if t != "afrimgsm":
            continue
        c = Counter(mgsm_category(r) for r in prompts[best[(m, l, t)]])
        rows.append({"model": m, "lang": l, "best_prompt": best[(m, l, t)], **dict(sorted(c.items()))})
    write_csv("afrimgsm_errors.csv", rows)

    write_cases(runs, best, args.cases_per_subject, args.seed)


def write_cases(runs, best, per_subject: int, seed: int) -> None:
    """Errors to code by hand: AfriMMLU questions a model answers in English but not in Kinyarwanda
    (per_subject per subject per model), and wrong AfriMGSM answers (10 per model)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    rng = random.Random(seed)
    sheets = {
        "afrimmlu_lost_in_kin": (["model", "id", "subject", "question", "choices", "gold", "pred", "category", "source", "notes"],
                                 "translation_or_term,misread_question,distractor_or_ambiguity,cultural_or_local,dataset_error,scoring_artifact,other"),
        "afrimgsm_errors": (["model", "lang", "id", "question", "gold", "extracted", "output", "auto_category", "category", "source", "notes"],
                            "number_word_misread,misread_question,arithmetic_slip,missing_reasoning,format_or_extraction,degenerate_output,dataset_error,other"),
    }
    data = {name: [] for name in sheets}
    for (m, l, t) in sorted(runs):
        if t == "afrimmlu" and l == "kin" and (m, "eng", t) in runs:
            eng = {r["id"]: r["correct"] for r in runs[(m, "eng", t)][best[(m, "eng", t)]]}
            lost = [r for r in runs[(m, l, t)][best[(m, l, t)]] if not r["correct"] and eng.get(r["id"])]
            by_sub = defaultdict(list)
            for r in lost:
                by_sub[r["subject"]].append(r)
            for sub, rs in sorted(by_sub.items()):
                for r in rng.sample(rs, min(per_subject, len(rs))):
                    data["afrimmlu_lost_in_kin"].append([m, r["id"], sub, r["question"],
                        "\n".join(f"{'ABCD'[j]}) {c}" for j, c in enumerate(r["choices"])), "ABCD"[r["gold"]], "ABCD"[r["pred"]]])
        if t == "afrimgsm":
            wrong = [r for r in runs[(m, l, t)][best[(m, l, t)]] if not r["correct"]]
            for r in rng.sample(wrong, min(10, len(wrong))):
                data["afrimgsm_errors"].append([m, l, r["id"], r["question"], r["gold"], r["extracted"],
                                                (r["output"] or "")[:500], mgsm_category(r)])

    wb = Workbook()
    wb.remove(wb.active)
    for name, (cols, options) in sheets.items():
        ws = wb.create_sheet(name)
        ws.append(cols)
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for row in data[name]:
            ws.append(row + [""] * (len(cols) - len(row)))
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        for col, width in zip("ABCDEFGHIJK", [14, 8, 18, 60, 40, 12, 12, 22, 12, 30, 30]):
            ws.column_dimensions[col].width = width
        n = len(data[name]) + 1
        cat_col = "ABCDEFGHIJK"[cols.index("category")]
        src_col = "ABCDEFGHIJK"[cols.index("source")]
        dv = DataValidation(type="list", formula1=f'"{options}"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{cat_col}2:{cat_col}{n}")
        dv2 = DataValidation(type="list", formula1='"model,data,task"', allow_blank=True)
        ws.add_data_validation(dv2)
        dv2.add(f"{src_col}2:{src_col}{n}")
        ws.freeze_panes = "D2"
    OUT.mkdir(parents=True, exist_ok=True)
    wb.save(OUT / "error_cases.xlsx")
    print(f"wrote {OUT / 'error_cases.xlsx'}")


if __name__ == "__main__":
    main()