"""Figures for the progress update / write-up, from the analysis tables and run summaries.

Inputs (all produced by earlier scripts, nothing typed in by hand):
  results/raw/baseline/<model>/<nochat|nochat_eng>/summary.txt    best and mean accuracy, all 5 models
  results/analysis/robust_lost_in_kin.csv, model_overlap.csv, robust_groups.csv   (error_analysis.py,
  results/analysis/rag_flips.csv, rag_flips_by_subject.csv                          robust_and_rag.py)
Writes PNG (300 dpi, for slides) and SVG (editable) to results/figures/.
Retrieval runs are found automatically, so the KinyaColBERT runs appear once their rows are in rag_flips.csv.

    python analysis/error_analysis.py --models gemma2-27b gemma4-31b qwen38-27b
    python analysis/robust_and_rag.py --models gemma2-27b gemma4-31b qwen38-27b
    python analysis/make_figures.py
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "baseline"
TABLES = ROOT / "results" / "analysis"
OUT = ROOT / "results" / "figures"

# reference palette (dataviz skill, light mode): categorical slots in fixed order, text inks, surface
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEG = "#e34948"                      # diverging pair: blue (+) / red (-)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

MODELS = {  # folder -> display name, in a fixed order (color follows the model, never its rank)
    "gemma2-27b": "Gemma 2 27B", "gemma4-31b": "Gemma 4 31B", "qwen38-27b": "Qwen 3.8 27B",
    "gptoss-20b": "gpt-oss-20b", "gptoss-120b": "gpt-oss-120b",
}
TASKS = {"afrimmlu": "AfriMMLU", "afrimgsm": "AfriMGSM"}
LANGS = {"nochat": "kin", "nochat_eng": "eng"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def read_csv(name: str) -> list[dict]:
    f = TABLES / name
    if not f.exists():
        print(f"  skip: {f.relative_to(ROOT)} not found")
        return []
    return list(csv.DictReader(open(f, encoding="utf-8")))


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote results/figures/{name}.png/.svg")


def value_grid(ax, axis="y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def read_summaries() -> dict:
    """acc[(model, lang, task)] = (best, mean, best_prompt). The language comes from the folder, not the label."""
    acc = {}
    for model in MODELS:
        for mode, lang in LANGS.items():
            f = RAW / model / mode / "summary.txt"
            if not f.exists():
                continue
            task = None
            for line in open(f, encoding="utf-8"):
                m = re.match(r"(afrimmlu|afrimgsm)\b", line.strip())
                if m:
                    task = m.group(1)
                    best = mean = prompt = None
                    continue
                if task and line.strip().startswith("best"):
                    m = re.search(r":\s*([\d.]+)\s*\(prompt (\d)\)", line)
                    best, prompt = float(m.group(1)), int(m.group(2))
                if task and line.strip().startswith("mean"):
                    mean = float(re.search(r":\s*([\d.]+)", line).group(1))
                    acc[(model, lang, task)] = (best, mean, prompt)
    return acc


# ------------------------------------------------------------------ 6.1 language gap
def fig_language_gap(acc) -> None:
    models = [m for m in MODELS if any(k[0] == m for k in acc)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    w = 0.38
    for ax, task in zip(axes, TASKS):
        for j, (lang, color, label) in enumerate([("kin", BLUE, "Kinyarwanda"), ("eng", ORANGE, "English")]):
            xs = [i + (j - 0.5) * w for i in range(len(models))]
            ys = [acc.get((m, lang, task), (0,))[0] for m in models]
            ax.bar(xs, ys, width=w - 0.04, color=color, label=label, zorder=2)
            for x, y in zip(xs, ys):
                ax.text(x, y + 1, f"{y:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(range(len(models)), [MODELS[m] for m in models], rotation=20, ha="right")
        ax.set_title(TASKS[task])
        value_grid(ax)
    axes[0].set_ylabel("Accuracy, best of 5 prompts (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="upper right")
    fig.suptitle("Every model loses much of its accuracy in Kinyarwanda", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "01_language_gap")


# ------------------------------------------------------------------ 6.2 lost in Kinyarwanda
def fig_lost_in_kin() -> None:
    rows = [r for r in read_csv("robust_lost_in_kin.csv") if r["model"] in MODELS]
    if not rows:
        return
    rows.sort(key=lambda r: list(MODELS).index(r["model"]), reverse=True)
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.55 * len(rows)))
    for i, r in enumerate(rows):
        eng, lost = int(r["eng_robust_correct"]), int(r["robust_lost_in_kin"])
        ax.barh(i, eng - lost, color=BLUE, height=0.6, label="Also right in Kinyarwanda (some prompts)" if i == 0 else None)
        ax.barh(i, lost, left=eng - lost + 2, color=ORANGE, height=0.6,
                label="Wrong in Kinyarwanda under all 5 prompts" if i == 0 else None)
        ax.text(eng + 6, i, f"{lost} of {eng} lost ({float(r['lost_pct_of_eng_robust']):.0f}%)",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [MODELS[r["model"]] for r in rows])
    ax.set_xlabel("AfriMMLU questions answered right in English under all 5 prompts")
    ax.set_xlim(0, max(int(r["eng_robust_correct"]) for r in rows) * 1.35)
    value_grid(ax, "x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    ax.set_title("What models reliably know in English, they often reliably miss in Kinyarwanda")
    save(fig, "02_lost_in_kinyarwanda")


# ------------------------------------------------------------------ 6.3 robust vs loose
def fig_robust_vs_loose() -> None:
    loose = {r["group"]: int(r["n"]) for r in read_csv("model_overlap.csv")}
    robust = {r["group"]: int(r["n"]) for r in read_csv("robust_groups.csv")}
    if not loose or not robust:
        return
    models = [m for m in MODELS if f"robust_only_{m}" in robust]
    groups = [("Solved by all", "solved_by_all", "robust_all"), ("Solved by none", "solved_by_none", "robust_none")]
    groups += [(f"Only {MODELS[m]}", f"only_{m}", f"robust_only_{m}") for m in models]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.38
    for j, (src, color, label) in enumerate([(loose, INK2 + "80", "Under at least one prompt (loose)"),
                                             (robust, BLUE, "Under all 5 prompts (robust)")]):
        key = 1 if j == 0 else 2
        xs = [i + (j - 0.5) * w for i in range(len(groups))]
        ys = [src.get(g[key], 0) for g in groups]
        ax.bar(xs, ys, width=w - 0.04, color=color if j else "#b9b8b3", label=label, zorder=2)
        for x, y in zip(xs, ys):
            ax.text(x, y + 2, str(y), ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], rotation=15, ha="right")
    ax.set_ylabel("AfriMMLU (kin) questions, of 500")
    value_grid(ax)
    ax.legend(loc="upper right")
    ax.set_title("Most apparent 'only this model' wins disappear when all 5 prompts must agree")
    save(fig, "03_robust_vs_loose")


# ------------------------------------------------------------------ 6.5 prompt sensitivity
def fig_prompt_sensitivity(acc) -> None:
    models = [m for m in MODELS if any(k[0] == m for k in acc)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.8 + 0.5 * 2 * len(models)), sharey=True)
    labels = [f"{MODELS[m]} ({lang})" for m in models for lang in ("kin", "eng")]
    for ax, task in zip(axes, TASKS):
        for i, (m, lang) in enumerate((m, lang) for m in models for lang in ("kin", "eng")):
            if (m, lang, task) not in acc:
                continue
            best, mean, _ = acc[(m, lang, task)]
            y = len(labels) - 1 - i
            color = BLUE if lang == "kin" else ORANGE
            ax.plot([mean, best], [y, y], color=GRID, linewidth=2, zorder=1)
            ax.scatter([mean], [y], s=40, color=SURFACE, edgecolor=color, linewidth=2, zorder=2,
                       label="Mean of 5 prompts" if i == 0 else None)
            ax.scatter([best], [y], s=40, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label="Best prompt" if i == 0 else None)
            ax.text(best + 1.2, y, f"+{best - mean:.1f}", va="center", fontsize=8, color=INK2)
        ax.set_yticks(range(len(labels)), labels[::-1])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Accuracy (%)")
        ax.set_title(TASKS[task])
        value_grid(ax, "x")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("The best prompt can beat the average prompt by several points", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "04_prompt_sensitivity")


# ------------------------------------------------------------------ retrieval runs
def run_label(run: str) -> str:
    setting = run.split("/")[0]
    retriever, rest = setting.split("-k", 1)
    k = rest.split("-")[0]
    name = {"bm25": "BM25", "kinyacolbert": "KinyaColBERT", "afrie5": "AfriE5"}.get(retriever, retriever)
    return f"{name} k={k}" + (", no math" if "-skip-" in setting else "")


def run_order(run: str):
    setting = run.split("/")[0]
    retriever, rest = setting.split("-k", 1)
    return (["bm25", "afrie5", "kinyacolbert"].index(retriever) if retriever in ("bm25", "afrie5", "kinyacolbert")
            else 9, -int(rest.split("-")[0]), "-skip-" in setting)


def fig_rag_results() -> None:
    rows = read_csv("rag_flips.csv")
    if not rows:
        return
    runs = sorted({r["run"] for r in rows}, key=run_order)
    per_prompt = defaultdict(list)
    pooled = {}
    for r in rows:
        if r["prompt"].startswith("all"):
            pooled[r["run"]] = r
        else:
            per_prompt[r["run"]].append(r)
    first = per_prompt[runs[0]]
    bars = [("No retrieval", max(float(r["acc_no_retrieval"]) for r in first),
             float(pooled[runs[0]]["acc_no_retrieval"]))]
    bars += [(run_label(x), max(float(r["acc_retrieval"]) for r in per_prompt[x]),
              float(pooled[x]["acc_retrieval"])) for x in runs]
    fig, ax = plt.subplots(figsize=(1.6 + 1.15 * len(bars), 3.8))
    w = 0.38
    for j, (idx, color, label) in enumerate([(1, BLUE, "Best of 5 prompts"), (2, ORANGE, "Mean of 5 prompts")]):
        xs = [i + (j - 0.5) * w for i in range(len(bars))]
        ys = [b[idx] for b in bars]
        ax.bar(xs, ys, width=w - 0.04, color=color, label=label, zorder=2)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.4, f"{y:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.axhline(bars[0][2], color=INK2, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.set_xticks(range(len(bars)), [b[0] for b in bars], rotation=15, ha="right")
    lo = min(min(b[1], b[2]) for b in bars)
    ax.set_ylim(max(0, lo - 10), max(b[1] for b in bars) + 4)
    ax.set_ylabel("AfriMMLU (kin) accuracy (%)")
    value_grid(ax)
    ax.legend(loc="upper right", ncol=2)
    model = runs[0].split("/")[1]
    ax.set_title(f"Retrieval-augmented {MODELS.get(model, model)}: no setting beats no retrieval"
                 if all(b[2] <= bars[0][2] for b in bars[1:]) else
                 f"Retrieval-augmented {MODELS.get(model, model)} on AfriMMLU (kin)")
    ax.text(0, -0.32, "Y axis does not start at 0. Dashed line: mean without retrieval.", transform=ax.transAxes,
            fontsize=8, color=INK2)
    save(fig, "05_rag_results")


def fig_rag_flips() -> None:
    rows = read_csv("rag_flips.csv")
    if not rows:
        return
    runs = sorted({r["run"] for r in rows}, key=run_order)
    cols = min(2, len(runs))
    nrows = (len(runs) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(5 * cols, 2.4 * nrows), sharex=True, squeeze=False)
    lim = max(max(int(r["right_only_without"]), int(r["right_only_with"])) for r in rows
              if not r["prompt"].startswith("all")) * 1.25
    for ax, run in zip(axes.flat, runs):
        rs = sorted((r for r in rows if r["run"] == run and not r["prompt"].startswith("all")),
                    key=lambda r: int(r["prompt"]))
        pooled = next(r for r in rows if r["run"] == run and r["prompt"].startswith("all"))
        ys = [len(rs) - 1 - i for i in range(len(rs))]
        ax.barh(ys, [int(r["right_only_with"]) for r in rs], color=BLUE, height=0.6,
                label="Right only with retrieval")
        ax.barh(ys, [-int(r["right_only_without"]) for r in rs], color=NEG, height=0.6,
                label="Right only without retrieval")
        for y, r in zip(ys, rs):
            ax.text(int(r["right_only_with"]) + 1.5, y, r["right_only_with"], va="center", fontsize=8, color=INK2)
            ax.text(-int(r["right_only_without"]) - 1.5, y, r["right_only_without"], va="center", ha="right",
                    fontsize=8, color=INK2)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_yticks(ys, [f"prompt {r['prompt']}" for r in rs])
        ax.set_xlim(-lim, lim)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
        p = float(pooled["mcnemar_p"])
        ax.set_title(f"{run_label(run)}   (pooled McNemar p={p:.3f})", fontsize=10)
        value_grid(ax, "x")
    for ax in axes.flat[len(runs):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Retrieval changes many answers in both directions, and the losses slightly outweigh the gains",
                 x=0.01, ha="left", fontweight="bold")
    fig.supxlabel("Questions (of 500) whose answer flips", fontsize=9, color=INK2, y=0.03)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "06_rag_flips")


def fig_rag_by_subject() -> None:
    rows = read_csv("rag_flips_by_subject.csv")
    if not rows:
        return
    runs = sorted({r["run"] for r in rows}, key=run_order)
    if len(runs) > len(SERIES):
        runs = runs[: len(SERIES)]
    subjects = sorted({r["subject"] for r in rows})
    delta = {(r["run"], r["subject"]): float(r["acc_retrieval"]) - float(r["acc_no_retrieval"]) for r in rows}
    fig, ax = plt.subplots(figsize=(8.5, 0.8 + 0.42 * len(subjects) * max(1, len(runs)) / 2))
    h = 0.8 / len(runs)
    for j, run in enumerate(runs):
        ys = [len(subjects) - 1 - i + (len(runs) / 2 - j - 0.5) * h for i in range(len(subjects))]
        ax.barh(ys, [delta.get((run, s), 0) for s in subjects], height=h - 0.03, color=SERIES[j],
                label=run_label(run), zorder=2)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(subjects)), [s.replace("_", " ") for s in subjects][::-1])
    ax.set_xlabel("Change in accuracy with retrieval, mean of 5 prompts (points)")
    value_grid(ax, "x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(4, len(runs)), fontsize=8)
    ax.set_title("Effect of retrieval by subject")
    save(fig, "07_rag_by_subject")


def main() -> None:
    acc = read_summaries()
    print(f"read {len(acc)} model/language/task summaries")
    fig_language_gap(acc)
    fig_lost_in_kin()
    fig_robust_vs_loose()
    fig_prompt_sensitivity(acc)
    fig_rag_results()
    fig_rag_flips()
    fig_rag_by_subject()


if __name__ == "__main__":
    main()
