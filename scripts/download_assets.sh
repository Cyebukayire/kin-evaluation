#!/bin/bash
# Download model weights and datasets ahead of time, so Slurm jobs can run offline.
# Run once from a node with internet access, from the repo root:
#   bash scripts/download_assets.sh gemma2-27b
set -euo pipefail

WHICH=${1:-gemma2-27b}
case "$WHICH" in
  gemma2-27b)  MODEL=google/gemma-2-27b-it ;;               # access: https://huggingface.co/google/gemma-2-27b-it
  gemma4-31b)  MODEL=google/gemma-4-31B-it ;;
  qwen38-27b)  MODEL=Qwen/Qwen3.8-27B ;;
  gptoss-20b)  MODEL=openai/gpt-oss-20b ;;
  gptoss-120b) MODEL=openai/gpt-oss-120b ;;
  *) echo "unknown model $WHICH"; exit 1 ;;
esac

SCRATCH=/local/scratch/$USER
export HF_HOME=$SCRATCH/hf_cache
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kin-eval
conda activate kin-eval


hf auth whoami
hf download "$MODEL" --exclude "original/*"

if [ "$WHICH" = "gptoss-120b" ]; then
  python -c "from kernels import get_kernel; get_kernel('kernels-community/gpt-oss-triton-kernels', version=1)" || true
fi

python - <<'EOF'
from datasets import load_dataset
for name in ["masakhane/afrimmlu", "masakhane/afrimgsm"]:
    ds = load_dataset(name, "kin")
    print(name, {split: len(d) for split, d in ds.items()})
EOF
echo "Cached in $HF_HOME"
