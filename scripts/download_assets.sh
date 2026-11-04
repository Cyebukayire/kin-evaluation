#!/bin/bash
# Download model weights and datasets ahead of time, so Slurm jobs can run offline.
# Run once from a node with internet access, from the repo root:
#   bash scripts/download_assets.sh gemma2-27b
set -euo pipefail

WHICH=${1:-gemma2-27b}
case "$WHICH" in
  gemma2-27b)  MODEL=google/gemma-2-27b-it ;;               # access: https://huggingface.co/google/gemma-2-27b-it
  llama31-70b) MODEL=meta-llama/Llama-3.1-70B-Instruct ;;   # access: https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct
  *) echo "unknown model $WHICH"; exit 1 ;;
esac

SCRATCH=/local/scratch/$USER
export HF_HOME=$SCRATCH/hf_cache
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kin-eval
conda activate kin-eval


hf auth whoami
hf download "$MODEL" --exclude "original/*"

python - <<'EOF'
from datasets import load_dataset
for name in ["masakhane/afrimmlu", "masakhane/afrimgsm"]:
    ds = load_dataset(name, "kin")
    print(name, {split: len(d) for split, d in ds.items()})
EOF
echo "Cached in $HF_HOME"
