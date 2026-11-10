# kin-evaluation

Evaluating LLMs on Kinyarwanda question answering, and testing whether morphology-aware retrieval (KinyaColBERT) improves them.
CS 371W, Emory University — Peace Cyebukayire, Miya Fu, Ramlah Amer, Jinho D. Choi.

## Layout

```
configs/
  datasets.yaml        dataset registry: source, splits, storage location (outside the repo)
src/kin_eval/
  data.py              downloads datasets and loads them in one common format
slurm/
  run.sbatch           generic Slurm runner: resources only, then runs the script you pass
scripts/
  setup_env.sh         one-time package install into the kin-eval conda env
  download_assets.sh   cache model weights + baseline datasets for offline Slurm jobs
  download_data.py     download datasets listed in configs/datasets.yaml
  run_baseline.sh      baseline: closed-book LLM on AfriMMLU + AfriMGSM (Kinyarwanda)
analysis/
  summarize_baseline.py  per-prompt scores, best prompt vs. published, mean ± sd
results/raw/           lm-eval outputs (not committed)
logs/                  Slurm logs (not committed)
```

## Experiment status

| # | Experiment | Status |
|---|---|---|
| 0 | Reproduce KinyaColBERT retrieval (agricultural benchmark) | done |
| 1 | Baseline: Gemma 2 27B-it, AfriMMLU + AfriMGSM (kin), nochat and chat | in progress |
| 2 | Baseline: LLaMA 3.1 70B-it, AfriMMLU + AfriMGSM (kin) | after 1; needs 2 GPUs |
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

The cluster's default shell is ksh; conda needs bash. Start `bash` first in any new terminal or tmux window.

```bash
bash
cd /local/scratch/$USER/code
git clone https://github.com/Cyebukayire/kin-evaluation.git
cd kin-evaluation
mkdir -p logs                                   # Slurm cannot create it

bash scripts/setup_env.sh                       # installs requirements into the kin-eval conda env

# Hugging Face login, saved where the scripts look for it
echo 'export HF_HOME=/local/scratch/$USER/hf_cache' >> ~/.bashrc
export HF_HOME=/local/scratch/$USER/hf_cache
hf auth login                                   # read token from https://huggingface.co/settings/tokens
hf auth whoami
```

Download the model and datasets (about 55 GB for Gemma 2), from the access node, which has internet:

```bash
tmux new -s download bash                       # detach: Ctrl+B then D; reattach: tmux attach -t download
conda activate kin-eval
bash scripts/download_assets.sh gemma2-27b
```

Slurm jobs run offline from this cache.

### 2. Run

Resources per job (defaults come from `slurm/run.sbatch`; the commands below override partition and time):

| Job | Partition | GPUs | Memory | Time limit | Expected |
|---|---|---|---|---|---|
| Smoke test (10 questions per prompt) | `h100hour` | 1 | 128G | 50 min | ~10–20 min |
| Full run, Gemma 2 27B | `h100day` | 1 | 128G | 12 h (default) | a few hours |
| Full run, LLaMA 3.1 70B | `h100day` | 2 | 128G | 12 h (default) | longer; not run yet |

From the repo root.

```bash
SMOKE_NOCHAT=$(sbatch --parsable --job-name=g2-smoke-nochat --partition=h100hour --time=00:50:00 \
  slurm/run.sbatch scripts/run_baseline.sh gemma2-27b nochat 10)
SMOKE_CHAT=$(sbatch --parsable --job-name=g2-smoke-chat --partition=h100hour --time=00:50:00 \
  slurm/run.sbatch scripts/run_baseline.sh gemma2-27b chat 10)

sbatch --job-name=g2-full-nochat --partition=h100day --dependency=afterok:$SMOKE_NOCHAT \
  slurm/run.sbatch scripts/run_baseline.sh gemma2-27b nochat
sbatch --job-name=g2-full-chat --partition=h100day --dependency=afterok:$SMOKE_CHAT \
  slurm/run.sbatch scripts/run_baseline.sh gemma2-27b chat
```

Check and follow jobs:

```bash
squeue -u $USER -o "%.8i %.10P %.18j %.2t %.10M %.20E %R"   # full names, state, dependencies
tail -f logs/<job-name>-<jobid>.out                           # Ctrl+C stops watching; the job keeps running
scancel <jobid>                                               # e.g. full runs left with DependencyNeverSatisfied
```

`nochat` sends the raw prompt (lm-eval's default); `chat` applies Gemma's chat template. IrokoBench does not
say which it used, so we run both; whichever reproduces the published scores is the setting for every later
experiment.

LLaMA 3.1 70B needs two GPUs (about 140 GB in bf16):

```bash
sbatch --job-name=l3-full-nochat --partition=h100day --gres=gpu:2 \
  slurm/run.sbatch scripts/run_baseline.sh llama31-70b nochat
```

### 3. Results

Each job prints a summary at the end and saves it as `results/raw/<smoke|baseline>/<model>/<mode>/summary.txt`.
To re-run it:

```bash
python analysis/summarize_baseline.py results/raw/baseline/gemma2-27b/nochat
```

Smoke-test scores are meaningless (10 questions); only check that the summary exists.
Per-question outputs (`samples_*.jsonl`) are kept for significance tests against later runs.

## Data

Datasets are listed in `configs/datasets.yaml` and stored outside the repo under `data_root`
(default `/local/scratch/$USER/data`), one `.jsonl` file per split:

```bash
python scripts/download_data.py afrimmlu afrimgsm afriqa    # or --all
```

```python
import sys; sys.path.insert(0, "src")
from kin_eval.data import load
test = load("afrimmlu", "test")      # keys: id, question, choices, answer, answer_index, image, meta
```

The baseline does not read these files; lm-eval loads AfriMMLU and AfriMGSM from the Hugging Face cache.
They are for our own scripts (retrieval, prompting, analysis).

## Notes

- AfriMMLU is scored by log-likelihood over A–D; AfriMGSM by greedy generation, taking the last number in the answer.
- Gemma 2 runs in bfloat16 with eager attention (float16 is unstable for Gemma 2).
- Do not quantize LLaMA 3.1 70B to fit one GPU; it changes the scores.
- Never commit the MorphoKIN license (`*.dat`), MORPHODATA, model weights or Hugging Face tokens.
