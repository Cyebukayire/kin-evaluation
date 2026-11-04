"""Download datasets listed in configs/datasets.yaml to data_root (outside the repo).

    python scripts/download_data.py afrimmlu afrimgsm     # specific datasets
    python scripts/download_data.py --all                 # every dataset with enabled != false

Hub datasets are also cached in hf_cache, which lm-eval reuses, so baseline jobs can run offline.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kin_eval.data import download, load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="dataset names from configs/datasets.yaml")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    names = [n for n, s in cfg["datasets"].items() if s.get("enabled", True)] if args.all else args.names
    if not names:
        parser.error("name at least one dataset, or use --all")
    unknown = [n for n in names if n not in cfg["datasets"]]
    if unknown:
        parser.error(f"unknown dataset(s) {unknown}; choose from {list(cfg['datasets'])}")

    print(f"data_root: {cfg['data_root']}")
    for name in names:
        sizes = download(name, cfg)
        print(f"  {name}: " + ", ".join(f"{s}={n}" for s, n in sizes.items()))


if __name__ == "__main__":
    main()
