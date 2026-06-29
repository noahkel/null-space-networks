#!/bin/bash
#SBATCH --job-name=ellips1%
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=all
##SBATCH --nodelist=mp-gpu4-a6000-4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=noah.keltsch@uibk.ac.at

NTFY="c7021201_slurmjobs"
REPO_DIR=/scratch/noah/Null-Space-Networks
# Base folders. Each noise level lives in its own subfolder named str(<noise>),
# exactly as create_phantom_data.py writes it (e.g. 0.0, 0.01, 0.02, 0.05):
#   data:   $DATA_BASE/<noise>/{gt,sino,summary.json}
#   models: $MODEL_BASE/<noise>/init_<init>/checkpoints/<model>_best.pt
DATA_BASE=/scratch/noah/data/ellipses_out_matrices
MODEL_BASE=/scratch/noah/models_ellipses_matrices

cd $REPO_DIR
mkdir -p logs
export PYTHONPATH=/scratch/noah/Null-Space-Networks:$PYTHONPATH

curl -s -d "Job $SLURM_JOB_ID ($SLURM_JOB_NAME) started on $SLURMD_NODENAME" \
     "https://ntfy.sh/$NTFY" &

# -- Environment setup --------------------------------------------------------
module purge
module load anaconda/anaconda3
module load cuda/12.5
source ~/.bashrc
conda activate data_prox2

# -- Diagnostics --------------------------------------------------------------
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
MIN_ANGLE=0
MAX_ANGLE=120
NUM_THETAS=180
N_SAMPLES=5000
TYPE="ellipses"

# -- Noise levels to evaluate -------------------------------------------------
# Must match the data already generated and the models already trained. The
# folder name is str(<noise>) from create_phantom_data.py, so use 0.0 (not 0.00).
NOISE_LEVELS="0.0 0.01 0.02 0.05"

# -- Data Generation + Training (run once per noise level; normally pre-done) --
#for NOISE in $NOISE_LEVELS; do
#   python -u create_ellipse_data.py --img_size $IMG_SIZE --noise $NOISE \
#     --min_angle $MIN_ANGLE --max_angle $MAX_ANGLE --num_thetas $NUM_THETAS \
#     --n_samples $N_SAMPLES --matrix_mode 1 --out_dir $DATA_BASE
#   python -u train.py --type $TYPE --out_dir $MODEL_BASE/$NOISE \
#     --data_dir $DATA_BASE/$NOISE --models resnet,nsn
#done
echo "Finished Data Generation / Training section at: $(date)"

# -- Adversarial Attacks (epsilon sweep, per matched noise level) -------------
# For each noise level we attack the data generated at that level with the
# models trained at that same level, so the evaluation is self-consistent.
# eps is scaled per-sample by ||y_i|| inside attack.py, so the same eps fraction
# is a comparable perturbation across noise levels. --tag encodes the noise level
# so runs never overwrite each other (-> attack_runs_ellipses_n<noise>/).
#
# Canonical fair attack: --objective null at full budget; the range/null error
# decomposition isolates the structural channel in the metric (see attack.py).
# --objective-matrix additionally reports the mse vs null channel comparison;
# --lipschitz reports the attack-free null-restricted Lipschitz of each model.
EPS="0.005,0.01,0.02,0.05"
INIT=pinv

for NOISE in $NOISE_LEVELS; do
  DATA_DIR_NOISE=$DATA_BASE/$NOISE
  MODEL_DIR_NOISE=$MODEL_BASE/$NOISE

  if [ ! -f "$DATA_DIR_NOISE/summary.json" ]; then
    echo "[skip] no data for noise $NOISE at $DATA_DIR_NOISE (summary.json missing)"
    continue
  fi

  echo "=== Attacking noise=$NOISE  data=$DATA_DIR_NOISE  models=$MODEL_DIR_NOISE  at $(date) ==="
  python -u attack.py --type $TYPE --init $INIT --eps $NOISE --steps 200 --objective null \
    --data-root $DATA_DIR_NOISE --model-dir $MODEL_DIR_NOISE --models nsn,resnet \
    --attacks adam --norm l2 --tag "ellipses_fast_n${NOISE}" \
    --objective-matrix mse,null \
    --lipschitz --lipschitz-samples 32 --lipschitz-iters 10
done
echo "Finished Adversarial Attacks at: $(date)"

# -- Done ---------------------------------------------------------------------
echo "Job finished at $(date)"


curl -s -d "Job $SLURM_JOB_ID ($SLURM_JOB_NAME) finished at $(date)" \
     "https://ntfy.sh/$NTFY"
