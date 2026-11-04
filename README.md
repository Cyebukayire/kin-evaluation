# kin-evaluation

Evaluating LLMs on Kinyarwanda question answering, and testing whether morphology-aware retrieval (KinyaColBERT) improves them.
CS 371W, Emory University — Peace Cyebukayire, Miya Fu, Ramlah Amer, Jinho D. Choi.

## Layout

```
Datasets/            local copies for inspection (lm-eval downloads its own copies from Hugging Face)
slurm/
  run.sbatch         generic Slurm runner: resources only, then runs the script you pass
scripts/
  setup_env.sh       one-time Python environment on the cluster
  download_assets.sh cache model weights + datasets for offline Slurm jobs
  run_baseline.sh    baseline: closed-book LLM on AfriMMLU + AfriMGSM (Kinyarwanda)
analysis/
  summarize_baseline.py  per-prompt scores, best prompt vs. published, mean ± sd
results/raw/         lm-eval outputs (not committed)
logs/                Slurm logs (not committed)
tests/               exploratory scripts and notebooks

configs/datasets.yaml     one entry per dataset: source, splits, where it's stored
src/kin_eval/data.py      downloads datasets and loads them in one common format
scripts/download_data.py  python scripts/download_data.py afrimmlu afrimgsm afriqa  
```

## Experiment status

| # | Experiment | Status |
|---|---|---|
| 0 | Reproduce KinyaColBERT retrieval (agricultural benchmark) | done |
| 1 | Baseline: Gemma 2 27B-it, AfriMMLU + AfriMGSM (kin) | in progress |
| 2 | Baseline: LLaMA 3.1 70B-it, AfriMMLU + AfriMGSM (kin) | after 1 |
| 3 | Retrieval augmentation with KinyaColBERT | needs MorphoKIN + corpus |
| 4 | Class-aware tier 1 | needs affix inventory from MORPHODATA |

## Baseline: reproduce IrokoBench

Published in-language Kinyarwanda scores (Adelani et al., 2025):

| Model | AfriMMLU | AfriMGSM |
|---|---|---|
| Gemma 2 27B-it | 40.2 | 32.4 |
| LLaMA 3.1 70B-it | 38.6 | 26.4 |

Reproduced if the best prompt is within about 2 points.

### 1. One-time setup

Request model access on Hugging Face (logged in):
- Gemma 2: https://huggingface.co/google/gemma-2-27b-it
- LLaMA 3.1: https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct

On the cluster:

```bash
cd /local/scratch/$USER/code
git clone https://github.com/Cyebukayire/kin-evaluation.git
cd kin-evaluation
mkdir -p logs

srun --pty -c 8 --mem 16g bash      # compute-node shell
bash scripts/setup_env.sh
conda activate kin-eval
huggingface-cli whoami || huggingface-cli login
bash scripts/download_assets.sh gemma2-27b
exit
```

`download_assets.sh` needs internet; if the compute node has none, run it on the access node instead.

### 2. Run

From the repo root:

```bash
sbatch slurm/run.sbatch scripts/run_baseline.sh gemma2-27b nochat
squeue -u $USER                       # check the job
tail -f logs/kin-eval-<jobid>.out
```

LLaMA 3.1 70B needs two GPUs and more time; override the defaults on the command line:

```bash
sbatch --gres=gpu:2 --time=24:00:00 slurm/run.sbatch scripts/run_baseline.sh llama31-70b nochat
```

Run `chat` (Gemma's chat template) only if `nochat` does not reproduce the published scores.
Whichever matches is the setting for every later experiment.

### 3. Results

The job prints a summary at the end and saves it as `results/raw/baseline/<model>/<mode>/summary.txt`.
To re-run it:

```bash
python analysis/summarize_baseline.py results/raw/baseline/gemma2-27b/nochat
```

Per-question outputs (`samples_*.jsonl`) are kept for significance tests against later runs.

## Notes

- AfriMMLU is scored by log-likelihood over A–D; AfriMGSM by greedy generation, taking the last number in the answer.
- Gemma 2 runs in bfloat16 with eager attention (float16 is unstable for Gemma 2).
- LLaMA 3.1 70B needs about 140 GB in bf16, hence `--gres=gpu:2`. Do not quantize; it changes the scores.
- Jobs run offline from the cache made by `download_assets.sh`. If compute nodes have internet and you skip that step, remove `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` from `scripts/run_baseline.sh`.
- Never commit the MorphoKIN license (`*.dat`), MORPHODATA, model weights or Hugging Face tokens.
