#!/bin/bash
# Baseline (Algorithm 1): closed-book LLM on IrokoBench AfriMMLU and AfriMGSM, Kinyarwanda,
# zero-shot, all 5 prompts.
#
# Run via Slurm:
#   sbatch slurm/run.sbatch scripts/run_baseline.sh gemma2-27b nochat
#
# Published in-language kin scores (Adelani et al., 2025):
#   Gemma 2 27B-it    AfriMMLU 40.2   AfriMGSM 32.4

set -euo pipefail
{
WHICH=${1:-gemma2-27b}
MODE=${2:-nochat}
LIMIT=${3:-}

EVAL_LANG=${EVAL_LANG:-kin}            # kin (default) or eng
SUFFIX=""
if [ "$EVAL_LANG" != "kin" ]; then SUFFIX="_$EVAL_LANG"; fi

case "$WHICH" in
  gemma2-27b)  MODEL=google/gemma-2-27b-it;             EXTRA=",attn_implementation=eager" ;;
  gemma4-31b)  MODEL=google/gemma-4-31B-it;             EXTRA=",add_bos_token=True" ;;
  qwen38-27b)  MODEL=Qwen/Qwen3.8-27B;                  EXTRA="" ;;
  gptoss-20b)  MODEL=openai/gpt-oss-20b;                EXTRA="" ;;
  gptoss-120b) MODEL=openai/gpt-oss-120b; EXTRA="" ;;
  *) echo "unknown model $WHICH"; exit 1 ;;
esac

if [ -n "$LIMIT" ]; then
  OUT=results/raw/smoke/$WHICH/$MODE$SUFFIX
  LIMIT_FLAG="--limit $LIMIT"
else
  OUT=results/raw/baseline/$WHICH/$MODE$SUFFIX
  LIMIT_FLAG=""
fi
rm -rf "$OUT" && mkdir -p "$OUT"

CHAT_FLAG=""
[ "$MODE" = "chat" ] && CHAT_FLAG="--apply_chat_template"

TASKS=$(echo afrimmlu_direct_${EVAL_LANG}_prompt_{1..5} afrimgsm_${EVAL_LANG}_prompt_{1..5} | tr ' ' ',')

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
echo "git_commit=$(git rev-parse HEAD)" >> "$OUT/versions.txt"
python analysis/summarize_baseline.py "$OUT" | tee "$OUT/summary.txt"
exit
}
