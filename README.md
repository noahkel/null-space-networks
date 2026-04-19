# Null-Space-Networks
This project investigates the robustness of Null Space Networks (NSNs) for CT image reconstruction. NSNs are neural network architectures that enforce data consistency by decomposing their output into range-space and null-space components of the Radon transform, ensuring reconstructions are always consistent with the measured sinogram. The codebase trains and compares four models — a standard ResNet, a vanilla NSN, and two data-consistent variants (DPNSN and DPNSN_RES) — against adversarial attacks in the sinogram domain.

## Scripts

**`create_ellipse_data.py`** — Generates the training/test dataset of ellipse phantoms. Simulates CT measurements by applying the Radon transform to ground-truth images, adding relative noise to the sinograms, and saving the ground-truth, FBP, sinogram, TV, and Landweber reconstructions to disk.

```bash
python -u create_ellipse_data.py --img_size 128 --noise 0.05 --min_angle 0 --max_angle 90 --num_thetas 180 --n_samples 1000 --matrix_mode 1
```

**`train.py`** — Trains each of the four models (ResNet, NSN, DPNSN, DPNSN_RES) on the generated dataset, minimizing MSE against ground-truth reconstructions and saving model checkpoints.

```bash
python -u train.py --type "ellipses"
```

**`attack.py`** — Evaluates model robustness by running adversarial attacks (e.g. Adam-based PGD) on the sinogram measurements under L2 or L-inf norm constraints. Reports PSNR, SSIM, and relative L2 error, and decomposes reconstruction errors into range- and null-space components.

```bash
python -u attack.py --type "ellipses" --init fbp --models resnet,nsn,dpnsn,dpnsn_res --attacks adam --norm l2 --eps 5.0 --alpha 0.5 --steps 500 --data-root "/scratch/noah/data/ellipses_out" --model-dir "/scratch/noah/models"
```