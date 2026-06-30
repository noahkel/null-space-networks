#!/bin/bash
#SBATCH --job-name=attackFeat
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=all
##SBATCH --nodelist=mp-gpu4-a6000-4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=noah.keltsch@uibk.ac.at



NTFY="c7021201_slurmjobs"
REPO_DIR=/scratch/noah/Null-Space-Networks
# Each noise level lives in its own subfolder named str(<noise>) exactly as
# create_phantom_data.py writes it (e.g. 0.0, 0.01, 0.02, 0.05):
#   data:   $DATA_BASE/<noise>/{gt,sino,summary.json,...}
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

# -- Common settings ----------------------------------------------------------
TYPE="ellipses"
INIT=pinv
MODELS="nsn,resnet"          # add dpnsn,dpnsn_res if those checkpoints exist
ATTACK="adam"
NORM="l2"
IMG_SIZE=128
MIN_ANGLE=0
MAX_ANGLE=120
NUM_THETAS=180
N_SAMPLES=5000
# Per-sample eps is scaled by ||y_i|| inside attack.py, so the same fraction is a
# comparable perturbation across noise levels.
EPS_SWEEP="0.005,0.01,0.02,0.05"   # robustness curve (sample budget, for speed)
EPS_SINGLE="0.02"                  # single budget for the full-test-set runs
STEPS=200

# Noise levels to evaluate (must match generated data + trained models). Folder
# name is str(<noise>) from create_phantom_data.py -> use 0.0 (not 0.00).
NOISE_LEVELS="0.0 0.01 0.02 0.05"
# -- Data Generation + Training (run once per noise level; normally pre-done) --
#for NOISE in $NOISE_LEVELS; do
#   python -u create_ellipse_data.py --img_size $IMG_SIZE --noise $NOISE \
#     --min_angle $MIN_ANGLE --max_angle $MAX_ANGLE --num_thetas $NUM_THETAS \
#     --n_samples $N_SAMPLES --matrix_mode 1 --out_dir $DATA_BASE
#   python -u train.py --type $TYPE --out_dir $MODEL_BASE/$NOISE \
#     --data_dir $DATA_BASE/$NOISE --models resnet,nsn
#done
for NOISE in $NOISE_LEVELS; do
  DATA_DIR=$DATA_BASE/$NOISE
  MODEL_DIR=$MODEL_BASE/$NOISE

  if [ ! -f "$DATA_DIR/summary.json" ]; then
    python -u create_ellipse_data.py --img_size $IMG_SIZE --noise $NOISE \
      --min_angle $MIN_ANGLE --max_angle $MAX_ANGLE --num_thetas $NUM_THETAS \
      --n_samples $N_SAMPLES --matrix_mode 1 --out_dir $DATA_DIR
    python -u train.py --type $TYPE --out_dir $MODEL_DIR \
      --data_dir $DATA_DIR --models resnet,nsn
    echo "Created data for noise $NOISE at $DATA_DIR and trained models $MODELS"
    continue
  fi


  echo "=================================================================="
  echo "=== noise=$NOISE  data=$DATA_DIR  models=$MODEL_DIR  $(date) ==="
  echo "=================================================================="

  # ------------------------------------------------------------------------
  # -- robustness eps-sweep (null objective) on a sample budget, and in
  # the SAME process: FEATURE 1 (range vs null vs mse channel matrix), the
  # shift-weight sweep, the null-restricted Lipschitz, FEATURE 2 (ghosts),
  # FEATURE 3 (post-hoc consistency, full test set), and FEATURE 4 (the
  # consolidated error_report.csv/.md over these runs).
  # ------------------------------------------------------------------------
  echo "--- robustness sweep + range matrix + ghosts + consistency ---"
  python -u attack.py --type $TYPE --init $INIT --eps "$EPS_SWEEP" --steps $STEPS \
    --objective null --models $MODELS --attacks $ATTACK --norm $NORM \
    --data-root "$DATA_DIR" --model-dir "$MODEL_DIR" --tag "feat_main_n${NOISE}" \
    --max-samples 128 \
    --objective-matrix mse,null,range \
    --shift-weight-sweep 0,0.25,1,4 \
    --lipschitz --lipschitz-samples 32 --lipschitz-iters 10 \
    --ghost --ghost-alpha 0.5,1.0,2.0 --ghost-source roll --ghost-max-samples 64 \
    --check-consistency --consistency-max-samples -1

  # ------------------------------------------------------------------------
  # -- the range / support component as a PRIMARY attack
  # objective, documented over the whole test set (--max-samples -1) or large test set.
  # ------------------------------------------------------------------------
  echo "--- range/support-objective attack over full test set ---"
  python -u attack.py --type $TYPE --init $INIT --eps "$NOISE" --steps $STEPS \
    --objective range --models $MODELS --attacks $ATTACK --norm $NORM \
    --data-root "$DATA_DIR" --model-dir "$MODEL_DIR" --tag "feat_range_n${NOISE}" \
    --max-samples 500

  # ------------------------------------------------------------------------
  # -- targeted attack driving the reconstruction towards the
  # all-zero image, the target-distance metrics (adv_to_target_rel, target_drop_frac) quantify how
  # close to "everything 0" the attack gets each model.
  # ------------------------------------------------------------------------
  echo "--- targeted 0-image attack over test set ---"
  python -u attack.py --type $TYPE --init $INIT --eps "$EPS_SINGLE" --steps 300 \
    --objective targeted --target zero --models $MODELS --attacks $ATTACK --norm $NORM \
    --data-root "$DATA_DIR" --model-dir "$MODEL_DIR" --tag "feat_targeted_n${NOISE}" \
    --max-samples 50

  echo "=== finished noise=$NOISE at $(date) ==="
done

# -- Optional: aggregate every run's summary.json into one CSV/plots -----------
# extract_errors.py walks attack_runs_*/.../summary.json across all tags above.
echo "--- aggregating all runs with extract_errors.py ---"
python -u extract_errors.py --root "$REPO_DIR" --out error_summary_features.csv --no-plot || true

echo "Job finished at $(date)"
curl -s -d "Job $SLURM_JOB_ID ($SLURM_JOB_NAME) finished at $(date)" \
     "https://ntfy.sh/$NTFY"
