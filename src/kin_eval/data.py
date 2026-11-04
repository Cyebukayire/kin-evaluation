"""Dataset registry and loader.

Every dataset in configs/datasets.yaml is downloaded once (scripts/download_data.py) and saved
under data_root as one .jsonl file per split, with every example in the same schema:

    {"id": str, "question": str, "choices": list[str] | None, "answer": str,
     "answer_index": int | None, "image": None, "meta": dict}

so later code (retrieval, prompting, evaluation) never needs to know where a dataset came from.

    from kin_eval.data import load
    test = load("afrimmlu", "test")
"""
from __future__ import annotations

import ast
import json
import os
import urllib.request
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "datasets.yaml"
LETTERS = "ABCD"


# ---------------------------------------------------------------- config

def load_config(path: Path | str = CONFIG) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["data_root"] = Path(os.path.expandvars(cfg["data_root"]))
    cfg["hf_cache"] = Path(os.path.expandvars(cfg["hf_cache"]))
    return cfg


def split_path(name: str, split: str, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return cfg["data_root"] / name / f"{split}.jsonl"


# ---------------------------------------------------------------- loading

def load(name: str, split: str, cfg: dict | None = None) -> list[dict]:
    """Read a downloaded split in the common schema."""
    path = split_path(name, split, cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run: python scripts/download_data.py {name}")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------- download

def download(name: str, cfg: dict | None = None) -> dict[str, int]:
    """Fetch every split of one dataset, normalize it and write .jsonl files. Returns split sizes."""
    cfg = cfg or load_config()
    spec = cfg["datasets"][name]
    sizes = {}
    for split in spec["splits"]:
        rows = _fetch(spec, split, cfg)
        examples = [NORMALIZE[name](row, i) for i, row in enumerate(rows)]
        out = split_path(name, split, cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        sizes[split] = len(examples)
    return sizes


def _fetch(spec: dict, split: str, cfg: dict) -> list[dict]:
    if spec["source"] == "huggingface":
        from datasets import load_dataset  # only needed for Hub datasets

        config = spec.get("config", "").format(split=split) or None
        hub_split = spec.get("hub_split", split)
        ds = load_dataset(spec["path"], config, split=hub_split, cache_dir=str(cfg["hf_cache"] / "datasets"))
        return [dict(r) for r in ds]
    if spec["source"] == "url":
        with urllib.request.urlopen(spec["url"].format(split=split)) as r:
            text = r.read().decode("utf-8")
        return _parse_json_or_jsonl(text)
    raise ValueError(f"unknown source {spec['source']}")


def _parse_json_or_jsonl(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _as_list(x) -> list:
    """Some datasets store lists as their Python string form, e.g. "['p = 4', 'p = 8']"."""
    if isinstance(x, list):
        return x
    try:
        return list(ast.literal_eval(x))
    except (ValueError, SyntaxError):
        # Unescaped apostrophes (e.g. "n'Uburusiya") break literal_eval: split the bracketed list by hand.
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
            s = s[1:-1]
        return [part for part in s.split("', '")] if s else []


# ---------------------------------------------------------------- normalizers (one per dataset)

def _afrimmlu(r: dict, i: int) -> dict:
    choices = [c.strip() for c in _as_list(r["choices"])]
    idx = LETTERS.index(r["answer"].strip())
    return {"id": f"afrimmlu-kin-{i}", "question": r["question"].strip(), "choices": choices,
            "answer": choices[idx], "answer_index": idx, "image": None,
            "meta": {"subject": r.get("subject"), "answer_letter": r["answer"].strip()}}


def _afrimgsm(r: dict, i: int) -> dict:
    num = r.get("answer_number")
    answer = str(num) if num is not None else str(r["answer"]).strip()
    if answer.endswith(".0"):
        answer = answer[:-2]
    return {"id": f"afrimgsm-kin-{i}", "question": r["question"].strip(), "choices": None,
            "answer": answer, "answer_index": None, "image": None, "meta": {}}


def _afriqa(r: dict, i: int) -> dict:
    answers = _as_list(r["answers"])  # empty for about a third of train/dev questions (no answer found)
    return {"id": f"afriqa-kin-{r.get('id', i)}", "question": r["question"].strip(), "choices": None,
            "answer": answers[0] if answers else "", "answer_index": None, "image": None,
            "meta": {"all_answers": answers, "answerable": bool(answers),
                     "translated_question": r.get("translated_question"),
                     "translated_answer": r.get("translated_answer")}}


def _afrimcqa(r: dict, i: int) -> dict:
    # Images are kept on the Hub cache; store the row ID so the image can be fetched when needed.
    options = [r["correct_native"], r["wrong_native_o1"], r["wrong_native_o2"], r["wrong_native_o3"]]
    return {"id": f"afrimcqa-kin-{r.get('ID', i)}", "question": r["native_question"].strip(),
            "choices": options, "answer": r["correct_native"], "answer_index": 0, "image": r.get("ID"),
            "meta": {"category": r.get("Category"), "eng_question": r.get("eng_question"),
                     "note": "correct option is listed first; shuffle before prompting"}}


NORMALIZE = {"afrimmlu": _afrimmlu, "afrimgsm": _afrimgsm, "afriqa": _afriqa, "afrimcqa": _afrimcqa}
