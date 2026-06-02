from pathlib import Path
from typing import Tuple, List

from matplotlib import image
import numpy as np
import torch

from src.radon import AstraRadonAdapter
from src.radon_matrix import MatrixRadonAdapter
from dival.datasets import EllipsesDataset

from src.utils import ensure_dir, set_seed, rel_l2, save_image_with_colorbar, to_4d
from src.total_variation import tv_cp
from src.landweber import landweber
from odl.phantom import ellipsoid_phantom, cuboid

import argparse

#
#   Creates ellipse samples using radon adapters from astra:
#   for i in range(N_SAMPLES):
#       x_gt = torch.from_numpy(next(gen).data).to(DEVICE)
#
#       y = radon.forward_la(to_4d(x_gt))
#       noise = radon.proj_ran(torch.randn_like(y))
#       y_delta = y + NOISE_sigma_REL * y.abs().max() * noise
#
#       y_diff_norms.append(float(torch.linalg.norm((y - y_delta).reshape(-1))))
#
#       x_fbp = radon.fbp_la(y_delta).squeeze()
#
#       np.save(OUT_DIR / "gt" / f"{i:05d}.npy", x_gt.detach().cpu().numpy())
#       np.save(OUT_DIR / "fbp" / f"{i:05d}.npy", x_fbp.detach().cpu().numpy())
#       np.save(OUT_DIR / "sino" / f"{i:05d}.npy", y_delta.squeeze().detach().cpu().numpy())
#
#       samples.append((x_gt, y_delta))
#
#   Also saves TV and Landweber reconstructions for use during training and testing
#   And an example image of the generated dataset
#
def single_rectangle_generator(dataset, part='train'):
    seed = dataset.fixed_seeds.get(part)
    r = np.random.RandomState(seed)

    lo, hi = np.asarray(dataset.space.min_pt), np.asarray(dataset.space.max_pt)
    center, half = (hi + lo) / 2, (hi - lo) / 2
    to_abs = lambda pt: (center + np.asarray(pt) * half).tolist()

    min_area = 0.1
    for i in range(dataset.get_len(part=part)):
        while True:
            a1 = 0.2 * r.exponential(1.0)
            a2 = 0.2 * r.exponential(1.0)
            if a1*a2>=min_area:
                break
        x   = r.uniform(-0.3, 0.3)   # tighter center range
        y   = r.uniform(-0.3, 0.3)
        image = cuboid(dataset.space, to_abs([y -a2/2, x-a1/2]), to_abs([y+a2/2, x+a1/2]))
        yield image

def single_ellipse_generator(dataset, part='train'):
    """Generator yielding images with exactly one random ellipse, centered and contained."""
    seed = dataset.fixed_seeds.get(part)
    r = np.random.RandomState(seed)
    n = dataset.get_len(part=part)
    from itertools import repeat
    it = repeat(None, n) if n is not None else repeat(None)
    for _ in it:
        min_area = 0.1

        while True:
            a1 = 0.2 * r.exponential(1.0)
            a2 = 0.2 * r.exponential(1.0)
            if np.pi * a1 * a2 < min_area:
                continue

            v   = r.uniform(0.3, 1.0)
            x   = r.uniform(-0.3, 0.3)   # tighter center range
            y   = r.uniform(-0.3, 0.3)
            rot = r.uniform(0., 2 * np.pi)

            # max extent of rotated ellipse along each axis
            dx = np.sqrt((a1 * np.cos(rot))**2 + (a2 * np.sin(rot))**2)
            dy = np.sqrt((a1 * np.sin(rot))**2 + (a2 * np.cos(rot))**2)

            if abs(x) + dx <= 1.0 and abs(y) + dy <= 1.0:
                break

        ellipsoids = np.array([[v, a1, a2, x, y, rot]])
        image = ellipsoid_phantom(dataset.space, ellipsoids)
        yield image


def main():
    # Initialization of parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--matrix_mode", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--min_angle", type=float, default=-60)
    parser.add_argument("--max_angle", type=float, default=60)
    parser.add_argument("--num_thetas", type=int, default=180)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--out_dir", type=str, default="./")
    parser.add_argument("--svd_thresh", type=float, default=4e-3)
    args = parser.parse_args()
    OUT_DIR = Path(args.out_dir)
    N_SAMPLES = args.n_samples
    TV_SUBSET = 100
    print(args)
    IMG_SIZE = args.img_size
    NUM_ANGLES = args.num_thetas
    MIN_ANGLE = args.min_angle
    MAX_ANGLE = args.max_angle
    DET_COUNT = int(np.sqrt(2)*IMG_SIZE) + 1
    # 1 To use matrix, 0 to use "exact" RadonOperator
    MATRIX_MODE = args.matrix_mode
    
    NOISE_sigma_REL = args.noise

    TV_ALPHA_GRID = np.logspace(np.log10(0.001), np.log10(1.0), 20)#np.logspace(-5, -1, 10)#[0.002, 0.005, 0.01, 0.02, 0.04]
    TV_ITERS_SELECT = 200
    TV_ITERS_FINAL = 200
    
    LW_ITERS = 200
    LW_OMEGA_FACTOR = 1.0
    SVD_THRESH = float(args.svd_thresh)

    THETA = 1.0

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(0)
    print("Generating on device " + str(DEVICE))
    # output structure
    ensure_dir(OUT_DIR)
    ensure_dir(OUT_DIR / f"gt{NOISE_sigma_REL}")
    ensure_dir(OUT_DIR / f"fbp{NOISE_sigma_REL}")
    #ensure_dir(OUT_DIR / "tv")
    #ensure_dir(OUT_DIR / "lw")
    ensure_dir(OUT_DIR / f"sino{NOISE_sigma_REL}")
    ensure_dir(OUT_DIR / f"pinv{NOISE_sigma_REL}")
    ensure_dir(OUT_DIR / f"pinv_full{NOISE_sigma_REL}")

    # dataset
    dataset = EllipsesDataset(image_size=IMG_SIZE)
    gen = single_ellipse_generator(dataset, 'train') #dataset.generator('train')   #single_ellipse_generator(dataset, 'train')
    gen = single_rectangle_generator(dataset, 'train')
    # radon
    dx = 1.0
    angles = np.linspace(0, 180, NUM_ANGLES, endpoint=False) * np.pi / 180
    phi = (MIN_ANGLE * np.pi / 180, MAX_ANGLE * np.pi / 180)

    if MATRIX_MODE == 0:
        radon = AstraRadonAdapter(
            resolution=IMG_SIZE,
            angles=angles,
            det_count=DET_COUNT,
            clip_to_circle=False,
            dx=dx,
            phi=phi
        )
    else:
        radon = MatrixRadonAdapter(
            resolution=IMG_SIZE,
            angles=angles,
            det_count=DET_COUNT,
            dx=dx,
            phi=phi,
            device=DEVICE,
            cache_dir="radon_cache",
            svd_threshold=SVD_THRESH
        )
        radon_full = MatrixRadonAdapter(
            resolution=IMG_SIZE,
            angles=angles,
            det_count=DET_COUNT,
            dx=dx,
            phi=phi,
            device=DEVICE,
            cache_dir="radon_cache",
            svd_threshold=1e-15
        )
    L = radon.norm_A2
    tau, sigma = 1/L, 1/L
    
    omega = (LW_OMEGA_FACTOR / radon.norm_A2)

    y_diff_norms: List[float] = []
    y_norms: List[float] = []
    subset = []

    print("Generating data...")
    samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for i in range(N_SAMPLES):

        x_gt = torch.from_numpy(next(gen).data).to(DEVICE)
        print(f"groundtruth: {torch.count_nonzero(x_gt)}")
        y = radon_full.forward_la(to_4d(x_gt))
        noise = radon_full.proj_ran(torch.randn_like(y))
        add_noise = NOISE_sigma_REL * y.abs().max() * noise / 100
        y_delta = y + add_noise

        y_norms.append(float(torch.linalg.norm(y.reshape(-1))))
        y_diff_norms.append(float(torch.linalg.norm((add_noise).reshape(-1))))

        x_fbp = radon_full.fbp_la(y_delta).squeeze()

        x_pinv = radon.backward_la(y_delta).squeeze()
        x_pinv_full = radon_full.backward_la(y_delta).squeeze()
        np.save(OUT_DIR / f"gt{NOISE_sigma_REL}" / f"{i:05d}.npy", x_gt.detach().cpu().numpy())
        np.save(OUT_DIR / f"fbp{NOISE_sigma_REL}" / f"{i:05d}.npy", x_fbp.detach().cpu().numpy())
        np.save(OUT_DIR / f"pinv{NOISE_sigma_REL}" / f"{i:05d}.npy", x_pinv.detach().cpu().numpy())
        np.save(OUT_DIR / f"pinv_full{NOISE_sigma_REL}" / f"{i:05d}.npy", x_pinv_full.detach().cpu().numpy())
        np.save(OUT_DIR / f"sino{NOISE_sigma_REL}" / f"{i:05d}.npy", y_delta.squeeze().detach().cpu().numpy())

        samples.append((x_gt, y_delta))

    y_diff_norms = np.array(y_diff_norms)

    np.save(OUT_DIR / f"y_diff_norms{NOISE_sigma_REL}.npy", y_diff_norms)

    subset = samples[:TV_SUBSET]

    print("Selecting TV alpha...")
    alpha_errors = {}

    #for alpha in TV_ALPHA_GRID:
     #   errs = []
      #  for x_gt, y_delta in subset:
       #     x0 = radon.fbp(y_delta)
        #    x_tv = tv_cp(
         #       x0=x0,
          #      A=radon.forward_la,
           #     AT=radon.backward_la,
            #    g=y_delta,
             #   alpha=alpha,
              #  tau=tau,
               # sigma=sigma,
                #theta=THETA,
                #Niter=TV_ITERS_SELECT,
                #print_flag=False,
                # grad_scale=0
            #).squeeze()
            #errs.append(rel_l2(x_tv, x_gt))
        #alpha_errors[alpha] = float(np.mean(errs))
        #print(f"alpha={alpha}: mean rel L2 = {alpha_errors[alpha]:.4e}")

    #best_alpha = min(alpha_errors, key=alpha_errors.get)
    #print(f"Best alpha: {best_alpha}")

    #print("Running final TV reconstructions...")
    #for i, (x_gt, y_delta) in enumerate(samples):
     #   x0 = radon.fbp(y_delta)
      #  x_tv = tv_cp(
       #     x0=x0,
        #    A=radon.forward_la,
         #   AT=radon.backward_la,
          #  g=y_delta,
           # alpha=best_alpha,
            #tau=tau,
            #sigma=sigma,
            #theta=THETA,
            #Niter=TV_ITERS_FINAL,
            #print_flag=False,
            # grad_scale=0
        #).squeeze()
        
        #x_lw = landweber(
         #   A=radon.forward_la,
          #  AT=radon.backward_la,
           # g=y_delta,
            #x0=x0,
            #omega=omega,
            #n_iter=LW_ITERS,
        #).squeeze()

        #np.save(OUT_DIR / "tv" / f"{i:05d}.npy", x_tv.detach().cpu().numpy())
        #np.save(OUT_DIR / "lw" / f"{i:05d}.npy", x_lw.detach().cpu().numpy())

    summary = {
        "dataset": "ellipse",
        "part": None,
        "n_samples": N_SAMPLES,
        "img_size": int(IMG_SIZE),
        "num_angles": int(NUM_ANGLES),
        "det_count": int(DET_COUNT),
        "angles": angles.tolist(),
        "dx": float(dx),
        "phi": list(phi),
        "phi_deg": [float(MIN_ANGLE), float(MAX_ANGLE)],
        "device": DEVICE,
        "add_noise": None,
        "noise_sigma_rel": float(NOISE_sigma_REL),
        "mean_norm_y": float(np.array(y_norms).mean()),
        "mean_norm_y_minus_y_delta": float(y_diff_norms.mean()),
        "tv_alpha_grid": [float(a) for a in TV_ALPHA_GRID.tolist()],
        "tv_alpha_errors_subset": alpha_errors,
        "tv_best_alpha": None, # float(best_alpha), if TV loop is running!
        "tv_iters_select": int(TV_ITERS_SELECT),
        "tv_iters_final": int(TV_ITERS_FINAL),
        "lw_iters": int(LW_ITERS),
        "lw_omega": float(omega),
        "lw_omega_factor": float(LW_OMEGA_FACTOR),
        "operator_norm_A2": float(L),
        "matrix_mode": int(MATRIX_MODE),
        "svd_threshold": SVD_THRESH
    }

    with open(OUT_DIR / f"summary{NOISE_sigma_REL}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done. Data saved to:", OUT_DIR.resolve())



if __name__ == "__main__":
    import json
    main()

