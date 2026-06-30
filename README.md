# Null-Space-Networks
This project investigates the robustness of Null Space Networks (NSNs) for CT image reconstruction. NSNs are neural network architectures that enforce data consistency by decomposing their output into range-space and null-space components of the Radon transform, ensuring reconstructions are always consistent with the measured sinogram. The codebase trains and compares four models — a standard ResNet, a vanilla NSN, and two data-consistent variants (DPNSN and DPNSN_RES) — against adversarial attacks in the sinogram domain.

## Scripts

**`create_ellipse_data.py` / `create_rectangle_data.py`** — Generate the training/test dataset of ellipse or rectangle phantoms (thin wrappers around the shared pipeline in `create_phantom_data.py`). Simulates CT measurements by applying the Radon transform to ground-truth images, adding relative noise to the sinograms, and saving the ground-truth, FBP, pinv, pinv_full, and sinogram data to disk.

```bash
python -u create_ellipse_data.py --img_size 128 --noise 0.05 --min_angle 0 --max_angle 90 --num_thetas 180 --n_samples 1000 --matrix_mode 1
python -u create_rectangle_data.py --img_size 128 --noise 0.05 --min_angle 0 --max_angle 90 --num_thetas 180 --n_samples 1000 --matrix_mode 1
```

**`train.py`** — Trains each of the four models (ResNet, NSN, DPNSN, DPNSN_RES) on the generated dataset, minimizing MSE against ground-truth reconstructions and saving model checkpoints.

```bash
python -u train.py --type "ellipses"
```

**`attack.py`** — Evaluates model robustness by running adversarial attacks (e.g. Adam-based PGD) on the sinogram measurements under L2 or L-inf norm constraints. Reports PSNR, SSIM, and relative L2 error, and decomposes reconstruction errors into range- and null-space components.

```bash
python -u attack.py --type "ellipses" --init fbp --models resnet,nsn,dpnsn,dpnsn_res --attacks adam --norm l2 --eps 5.0 --alpha 0.5 --steps 500 --data-root "/scratch/noah/data/ellipses_out" --model-dir "/scratch/noah/models"
```
## New attack.py capabilities

Five capabilities were added to `attack.py` (all opt-in; the existing commands keep working). `attackFeaturesSlurm.sh` runs all of them across the noise-level sweep on the HPC.

1. **Range / support objective focus** — in addition to the null-space objectives, the attack can now focus on the **range (data-consistent / "support") component**: `--objective range` (also `range_shift`, `range_hybrid`, or the aliases `support` / `support_shift` / `support_hybrid`). Useful as the mirror of the null objectives and inside `--objective-matrix mse,null,range`.

2. **Ghosts: hallucinations without an adversarial attack** — `--ghost` builds a null-space ghost `x2 = P_NS(x1)` and forms `x_corrupt = x_gt + alpha*x2`. Because the ghost lives in `null(A_la)` the measurement is essentially unchanged, so injecting it exposes the network's hallucination/instability directly. Controls: `--ghost-alpha 0.5,1.0,2.0`, `--ghost-source {roll,perm,randn}`, `--ghost-roll`, `--ghost-max-samples`. Outputs: `ghost_analysis/{ghost_report.csv,ghost_summary.json,ghost_response.png,ghost_*.png}`.

3. **Post-hoc data-consistency check via the Radon transform** — `--check-consistency` forward-projects every network reconstruction and compares it to the measured sinogram: `||A_la x_hat - y|| / ||y||` (measured angles) plus the unmeasured-angle extrapolation error. Outputs: `consistency/{consistency_report.csv,consistency_summary.json,consistency_bar.png}`. Budget: `--consistency-max-samples` (`-1` = whole test split).

4. **Cleaner error documentation over the whole test set** — `--max-samples -1` evaluates the entire test split, and a consolidated `error_report.csv` + `error_report.md` (clean vs adversarial rel-L2/PSNR/SSIM/MAE/NRMSE/max-err, null/range fractions, consistency, success rates) is written per run-root. Disable with `--no-error-report`.

5. **Targeted attack toward the 0-image** — `--objective targeted --target zero` drives the reconstruction as close as possible to the all-zero image (`--target` also accepts `gt`/`clean`). The targeted run logs `adv_to_target_rel` and `target_drop_frac` (how dark the attack makes the reconstruction).

```bash
# all five features, full test set, one noise level:
python -u attack.py --type ellipses --init pinv --models nsn,resnet --attacks adam --norm l2 \
  --eps 0.02 --steps 200 --objective range --max-samples -1 \
  --objective-matrix mse,null,range --ghost --check-consistency \
  --data-root "$DATA_DIR" --model-dir "$MODEL_DIR" --tag demo
# targeted 0-image attack:
python -u attack.py --type ellipses --init pinv --models nsn,resnet --attacks adam --norm l2 \
  --eps 0.02 --steps 300 --objective targeted --target zero --max-samples -1 \
  --data-root "$DATA_DIR" --model-dir "$MODEL_DIR" --tag demo_zero
```

Run everything on the HPC with `sbatch attackFeaturesSlurm.sh`.
