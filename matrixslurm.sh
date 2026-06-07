#!/bin/bash
#SBATCH --job-name=matrix
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=all
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=noah.keltsch@uibk.ac.at

NTFY="c7021201_slurmjobs"
REPO_DIR=/scratch/noah/Null-Space-Networks
DATA_DIR=/scratch/noah/data/rectangles_out_matrices
DATA_DIR_NOISE=/scratch/noah/data/rectangles_out_matrices/0.0
MODEL_DIR=/scratch/noah/models_rectangles_matrices

cd $REPO_DIR
mkdir -p logs
export PYTHONPATH=/scratch/noah/Null-Space-Networks:$PYTHONPATH

curl -s -d "Job $SLURM_JOB_ID ($SLURM_JOB_NAME) started on $SLURMD_NODENAME" \
     "https://ntfy.sh/$NTFY" &

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


IMG_SIZE=128 #in Pixels
NOISE=0.01 #in percent of
MIN_ANGLE=0
MAX_ANGLE=120
NUM_THETAS=180
N_SAMPLES=5000
TYPE="ellipses"

echo "finished test_radon.py at: $(date)"

# ── Data Generation (MatrixRadonAdapter, matrix_mode=1) ──────────────────────

#python -u create_ellipse_data.py --img_size $IMG_SIZE --noise 0 --min_angle $MIN_ANGLE --max_angle $MAX_ANGLE --num_thetas $NUM_THETAS --n_samples $N_SAMPLES --matrix_mode 1 --out_dir $DATA_DIR
##python -u create_ellipse_data.py --img_size $IMG_SIZE --noise 1 --min_angle $MIN_ANGLE --max_angle $MAX_ANGLE --num_thetas $NUM_THETAS --n_samples $N_SAMPLES --matrix_mode 1 --out_dir $DATA_DIR
##python -u create_ellipse_data.py --img_size $IMG_SIZE --noise 2 --min_angle $MIN_ANGLE --max_angle $MAX_ANGLE --num_thetas $NUM_THETAS --n_samples $N_SAMPLES --matrix_mode 1 --out_dir $DATA_DIR

echo "Finished Data Generation at: $(date)"

# ── Training (adapter chosen from summary.json matrix_mode) ──────────────────

#python -u train.py --type $TYPE --out_dir $MODEL_DIR --data_dir $DATA_DIR_NOISE --models resnet,nsn

echo "Finished Training at: $(date)"

# ── Adversarial Attacks ───────────────────────────────────────────────────────

#python -u attack.py --type $TYPE --eps 1.0 --alpha 0.5 --steps 40 --data-root $DATA_DIR --model-dir $MODEL_DIR --models resnet,nsn --init pinv --attacks adam --norm l2

#python -u attack.py --type $TYPE --eps 1.0 --alpha 0.5 --steps 40 --data-root $DATA_DIR --model-dir $MODEL_DIR --models resnet,nsn --init fbp --attacks adam --norm l2

echo "Finished Adversarial Attack at: $(date)"

python -u test_radon.py --data-dir $DATA_DIR_NOISE --model-dir $MODEL_DIR

# ── Done ─────────────────────────────────────────────────────────────────────

echo "Job finished at $(date)"


curl -s -d "Job $SLURM_JOB_ID ($SLURM_JOB_NAME) finished at $(date)" \
     "https://ntfy.sh/$NTFY"
