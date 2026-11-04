"""Summarize lm-eval results for the IrokoBench Kinyarwanda baseline.

Usage:
    python summarize_baseline.py /local/scratch/$USER/results/baseline/gemma2-27b/nochat

Prints, per task: accuracy for each of the 5 prompts, the best prompt (as in IrokoBench's
main tables) and mean +/- sd over prompts (as in their prompt ablation), next to the
published in-language score.
"""
import glob
import json
import re
import statistics
import sys
from pathlib import Path

PUBLISHED = {  # Adelani et al. 2025, in-language, Kinyarwanda
    "gemma2-27b": {"afrimmlu": 40.2, "afrimgsm": 32.4},
    "llama31-70b": {"afrimmlu": 38.6, "afrimgsm": 26.4},
}
# AfriMMLU: log-likelihood accuracy. AfriMGSM: exact match after extracting the last number.
METRIC = {"afrimmlu": "acc,none", "afrimgsm": "exact_match,flexible-extract"}


def load_results(out_dir: Path) -> dict:
    merged = {}
    for f in sorted(glob.glob(str(out_dir / "**" / "results_*.json"), recursive=True)):
        merged.update(json.load(open(f))["results"])
    if not merged:
        sys.exit(f"No results_*.json under {out_dir}")
    return merged


def main(out_dir: str) -> None:
    out = Path(out_dir)
    results = load_results(out)
    model = next((m for m in PUBLISHED if m in str(out)), None)

    for task, metric in METRIC.items():
        scores = {}
        for name, vals in results.items():
            m = re.fullmatch(rf"{task}(?:_direct)?_kin_prompt_(\d)", name)
            if m and metric in vals:
                scores[int(m.group(1))] = 100 * vals[metric]
        if not scores:
            print(f"{task}: no results found for metric {metric}")
            continue
        best_p = max(scores, key=scores.get)
        vals = list(scores.values())
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        pub = PUBLISHED.get(model, {}).get(task)
        print(f"\n{task} (kin), metric = {metric}")
        for p in sorted(scores):
            print(f"  prompt {p}: {scores[p]:5.1f}")
        print(f"  best      : {scores[best_p]:5.1f} (prompt {best_p})"
              + (f"   published: {pub}   diff: {scores[best_p] - pub:+.1f}" if pub else ""))
        print(f"  mean ± sd : {statistics.mean(vals):5.1f} ± {sd:.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
