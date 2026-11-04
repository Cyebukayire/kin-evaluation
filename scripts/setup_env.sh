#!/bin/bash
# One-time environment setup on the cluster. Run from the repo root, on a compute node:
#   srun --pty -c 8 --mem 16g bash
#   bash scripts/setup_env.sh
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kin-eval
pip install --upgrade pip

# PyTorch first, built for the cluster's CUDA. Check with `nvidia-smi` (top right: CUDA Version)
# and pick the matching index URL from https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

echo
echo "Installed in conda env: kin-eval"
echo "Next: huggingface-cli login   (if 'huggingface-cli whoami' does not show your username)"