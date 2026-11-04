#SBATCH --job-name=kin-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kin-eval
export HF_HOME=/local/scratch/$USER/hf_cache
cd "$SLURM_SUBMIT_DIR"
bash "$@"