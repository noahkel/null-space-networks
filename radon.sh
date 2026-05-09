#!/bin/bash
#SBATCH --job-name=radon
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=all
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=noah.keltsch@uibk.ac.at

REPO_DIR=/scratch/noah/Null-Space-Networks
DATA_DIR=/scratch/noah/data/ellipses_out
MODEL_DIR=/scratch/noah/models

cd $REPO_DIR
mkdir -p logs
export PYTHONPATH=/scratch/noah/InverseProblems:$PYTHONPATH

# ── Environment setup ────────────────────────────────────────────────────────
module purge
module load anaconda/anaconda3
module load cuda/12.5
source ~/.bashrc
conda activate data_prox2

# ── Diagnostics ──────────────────────────────────────────────────────────────
echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "GPU(s):        $CUDA_VISIBLE_DEVICES"
echo "Working dir:   $(pwd)"
echo "Start time:    $(date)"
echo "============================================"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo "============================================"


IMG_SIZE=128
NOISE=0.01
MIN_ANGLE=0
MAX_ANGLE=120
NUM_THETAS=180
N_SAMPLES=5000
TYPE="ellipses"

python -u test_radon.py --full
