#!/bin/bash
# For each model: a smoke test, then the full run, which starts only if its smoke test passed.
#   bash scripts/submit_all.sh gemma2-27b gemma4-31b qwen38-27b gptoss-20b
set -euo pipefail
MODE=${MODE:-nochat}
mkdir -p logs
for M in "$@"; do
  S=$(sbatch --parsable --job-name=$M-smoke --partition=h100hour --time=00:50:00 \
      slurm/run.sbatch scripts/run_eval.sh $M $MODE 10)
  F=$(sbatch --parsable --job-name=$M-full --partition=h100day --time=12:00:00 \
      --dependency=afterok:$S slurm/run.sbatch scripts/run_eval.sh $M $MODE)
  echo "$M: smoke $S -> full $F"
done
