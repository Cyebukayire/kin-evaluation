#!/bin/bash
#SBATCH --job-name=job
#SBATCH --partition=h100day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /local/scratch/pcyebuk/env.sh
conda activate kinya-rag

echo "host: $(hostname)  job: $SLURM_JOB_ID  start: $(date)"
python "$@"