#!/bin/bash
#SBATCH --job-name=matrixAtteck
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=a100
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


# ── Data Generation (MatrixRadonAdapter, matrix_mode=1) ──────────────────────
python -u create_ellipse_data.py \
    --img_size $IMG_SIZE \
    --noise $NOISE \
    --min_angle $MIN_ANGLE \
    --max_angle $MAX_ANGLE \
    --num_thetas $NUM_THETAS \
    --n_samples $N_SAMPLES \
    --matrix_mode 1

echo "Finished Data Generation at: $(date)"

BETA=$(python -c "import json;print(json.load(open('$DATA_DIR/summary.json'))['mean_norm_y_minus_y_delta'])")
echo "BETA=$BETA"


# ── Training (adapter chosen from summary.json matrix_mode) ──────────────────

python -u train.py --type $TYPE

echo "Finished Training at: $(date)"


# ── Adversarial Attacks ───────────────────────────────────────────────────────

python -u attack.py \
    --type $TYPE \
    --init fbp \
    --models resnet,nsn,dpnsn,dpnsn_res \
    --attacks adam \
    --norm l2 \
    --eps 1.0 \
    --alpha 0.5 \
    --steps 40 \
    --data-root $DATA_DIR \
    --model-dir $MODEL_DIR

echo "Finished Adversarial Attack at: $(date)"


# ── Done ─────────────────────────────────────────────────────────────────────
echo "Job finished at $(date)"
