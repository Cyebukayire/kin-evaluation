#!/bin/bash
# Baseline (Algorithm 1): closed-book LLM on IrokoBench AfriMMLU and AfriMGSM, Kinyarwanda,
# zero-shot, all 5 prompts.
#
# Run via Slurm:
#   sbatch slurm/run.sbatch scripts/run_baseline.sh gemma2-27b nochat
#   sbatch --gres=gpu:2 --time=24:00:00 slurm/run.sbatch scripts/run_baseline.sh llama31-70b nochat
#
# Published in-language kin scores (Adelani et al., 2025):
#   Gemma 2 27B-it    AfriMMLU 40.2   AfriMGSM 32.4
#   LLaMA 3.1 70B-it  AfriMMLU 38.6   AfriMGSM 26.4
set -euo pipefail

WHICH=${1:-gemma2-27b}
MODE=${2:-nochat}
LIMIT=${3:-}

case "$WHICH" in
  gemma2-27b)  MODEL=google/gemma-2-27b-it;             EXTRA=",attn_implementation=eager" ;;
  llama31-70b) MODEL=meta-llama/Llama-3.1-70B-Instruct; EXTRA=",parallelize=True" ;;
  *) echo "unknown model $WHICH"; exit 1 ;;
esac

if [ -n "$LIMIT" ]; then
  OUT=results/raw/smoke/$WHICH/$MODE
  LIMIT_FLAG="--limit $LIMIT"
else
  OUT=results/raw/baseline/$WHICH/$MODE
  LIMIT_FLAG=""
fi
mkdir -p "$OUT"

CHAT_FLAG=""
[ "$MODE" = "chat" ] && CHAT_FLAG="--apply_chat_template"

TASKS=$(echo afrimmlu_direct_kin_prompt_{1..5} afrimgsm_kin_prompt_{1..5} | tr ' ' ',')

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
lm_eval \
  --model hf \
  --model_args pretrained=$MODEL,dtype=bfloat16$EXTRA \
  --tasks "$TASKS" \
  --num_fewshot 0 \
  --batch_size auto \
  $CHAT_FLAG \
  $LIMIT_FLAG \
  --log_samples \
  --output_path "$OUT"

pip freeze | grep -iE "^(lm.eval|transformers|torch|accelerate)==" > "$OUT/versions.txt" || true
python analysis/summarize_baseline.py "$OUT" | tee "$OUT/summary.txt"
