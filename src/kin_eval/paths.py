"""External paths from configs/paths.env (and configs/paths.local.env, if present), for Python scripts.

The .env files are read by bash, exactly as the shell scripts read them, so $SCRATCH, $USER and
the like expand the same way in both. Relative paths (e.g. WIKI_DIR=Wikipedia_rw) are relative to
the repo root.

    from kin_eval.paths import get_path
    corpus = get_path("CORPUS_DIR") / "passages.jsonl"
"""
from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "paths.env"
LOCAL = ROOT / "configs" / "paths.local.env"


@lru_cache(maxsize=1)
def load_paths() -> dict[str, str]:
    names = []
    for f in (CONFIG, LOCAL):
        if f.exists():
            names += re.findall(r"^([A-Z_][A-Z0-9_]*)=", f.read_text(), flags=re.M)
    script = "set -a; source configs/paths.env; [ -f configs/paths.local.env ] && source configs/paths.local.env; env -0"
    out = subprocess.run(["bash", "-c", script], cwd=ROOT, capture_output=True, check=True).stdout
    env = dict(item.split("=", 1) for item in out.decode().split("\0") if "=" in item)
    return {n: env[n] for n in dict.fromkeys(names) if n in env}


def get_path(name: str) -> Path:
    paths = load_paths()
    if name not in paths:
        raise KeyError(f"{name} is not defined in {CONFIG.relative_to(ROOT)}")
    p = Path(paths[name])
    return p if p.is_absolute() else ROOT / p
