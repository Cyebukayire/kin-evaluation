#!/bin/bash
# Baseline (Algorithm 1): closed-book LLM on IrokoBench AfriMMLU and AfriMGSM, Kinyarwanda,
# zero-shot, all 5 prompts.
#
# Run via Slurm:
#   sbatch slurm/run.sbatch scripts/run_eval.sh gemma2-27b nochat
#
# Published in-language kin scores (Adelani et al., 2025):
#   Gemma 2 27B-it    AfriMMLU 40.2   AfriMGSM 32.4
#
# Retrieval-augmented AfriMMLU (Algorithm 2): set RETRIEVER (and optionally RAG_K, RAG_SKIP_SUBJECTS):
#   RETRIEVER=bm25 RAG_K=3 sbatch slurm/run.sbatch scripts/run_eval.sh gemma2-27b nochat
# reads results/retrieval/$RETRIEVER/afrimmlu_kin.jsonl and writes results/raw/rag/$RETRIEVER-k$RAG_K/...

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

OFFLINE_ENV="env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
if [ "$WHICH" = "gptoss-120b" ]; then OFFLINE_ENV="env HF_DATASETS_OFFLINE=1"; fi   # model side online for its GPU code lookup

if [ -n "$LIMIT" ]; then
  OUT=results/raw/smoke/$WHICH/$MODE$SUFFIX
  LIMIT_FLAG="--limit $LIMIT"
else
  OUT=results/raw/baseline/$WHICH/$MODE$SUFFIX
  LIMIT_FLAG=""
fi
RETRIEVER=${RETRIEVER:-}
INCLUDE_FLAG=""
if [ -n "$RETRIEVER" ]; then
  export AFRIMMLU_RAG_FILE=results/retrieval/$RETRIEVER/afrimmlu_${EVAL_LANG}.jsonl
  export RAG_K=${RAG_K:-3}
  export RAG_SKIP_SUBJECTS=${RAG_SKIP_SUBJECTS:-}
  [ -f "$AFRIMMLU_RAG_FILE" ] || { echo "missing $AFRIMMLU_RAG_FILE"; exit 1; }
  RUN=$RETRIEVER-k$RAG_K
  if [ -n "$RAG_SKIP_SUBJECTS" ]; then RUN=$RUN-skip-$(echo "$RAG_SKIP_SUBJECTS" | tr ',' '+'); fi
  if [ -n "$LIMIT" ]; then OUT=results/raw/smoke/rag/$RUN/$WHICH/$MODE$SUFFIX; else OUT=results/raw/rag/$RUN/$WHICH/$MODE$SUFFIX; fi
  INCLUDE_FLAG="--include_path tasks"
fi
echo "run: model=$WHICH mode=$MODE lang=$EVAL_LANG limit=${LIMIT:-none} retriever=${RETRIEVER:-none} k=${RAG_K:-} out=$OUT"
if [ -z "$LIMIT" ] && [ -d "$OUT" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "refusing to overwrite existing results in $OUT (rerun with FORCE=1 to replace them)"; exit 1
fi
rm -rf "$OUT" && mkdir -p "$OUT"

CHAT_FLAG=""
[ "$MODE" = "chat" ] && CHAT_FLAG="--apply_chat_template"

TASKS=$(echo afrimmlu_direct_${EVAL_LANG}_prompt_{1..5} afrimgsm_${EVAL_LANG}_prompt_{1..5} | tr ' ' ',')
if [ -n "$RETRIEVER" ]; then TASKS=$(echo afrimmlu_rag_${EVAL_LANG}_prompt_{1..5} | tr ' ' ','); fi

$OFFLINE_ENV \
lm_eval \
  --model hf \
  --model_args pretrained=$MODEL,dtype=bfloat16$EXTRA \
  --tasks "$TASKS" \
  --num_fewshot 0 \
  --batch_size ${BATCH_SIZE:-auto} \
  $CHAT_FLAG \
  $LIMIT_FLAG \
  --log_samples \
  $INCLUDE_FLAG \
  --output_path "$OUT"

pip freeze | grep -iE "^(lm.eval|transformers|torch|accelerate)==" > "$OUT/versions.txt" || true
if [ -n "$RETRIEVER" ]; then
  echo "retrieval_file=$AFRIMMLU_RAG_FILE sha256=$(sha256sum "$AFRIMMLU_RAG_FILE" | cut -c1-16) k=$RAG_K skip=$RAG_SKIP_SUBJECTS" >> "$OUT/versions.txt"
fi
gzip -f "$OUT"/*/samples_*.jsonl     # per-question answers, compressed (committed for closed-book runs)
if [ -n "$RETRIEVER" ]; then
  python scripts/strip_samples.py "$OUT"   # text-free copies for git; full logs hold KINNEWS passages
fi
ARCHIVE=/local/scratch/$USER/archive/$OUT  # second copy outside the repo, safe from re-clones and git clean
mkdir -p "$ARCHIVE" && cp "$OUT"/*/samples_*.jsonl.gz "$ARCHIVE"/
echo "git_commit=$(git rev-parse HEAD)" >> "$OUT/versions.txt"
python analysis/summarize_baseline.py "$OUT" | tee "$OUT/summary.txt"
exit
}