import numpy as np
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from datasets.registry import get_dataset
from models.model import get_model
from models.tikhonov import TikhonovInverter
from models.lasso import LassoInverter
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm
import time

def gaussian_smooth(wl, vals, lam_cont):
    """
    Compute Gaussian-basis smoothing: sum vals * exp(-0.5*((lam_cont - wl_i)/sigma)**2)
    with sigma = (high - low)/len(wl).
    """
    sigma = (lam_cont[-1] - lam_cont[0]) / len(wl)
    cont = np.zeros_like(lam_cont)
    for mu, v in zip(wl, vals):
        cont += v * (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * ((lam_cont - mu) / sigma) ** 2)
    return cont

def MSE(S, S_gt, low_idx, high_idx):
    diff_squared = [(S[i] - S_gt[i])**2 for i in range(low_idx, high_idx)]
    return np.mean(diff_squared)

def dataset_to_numpy(ds):
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    x_tensor, y_tensor = next(iter(loader))
    X = x_tensor.detach().cpu().numpy()
    Y = y_tensor.detach().cpu().numpy()
    return X, Y

def main():

    # Argument Settings

    p = argparse.ArgumentParser(description="Compare methods with gaussian-basis reconstruction at multiple resolutions")
    p.add_argument("--dataset", "-d", required=True)
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--checkpoint", "-c", required=True)
    p.add_argument("--root", type=str, default="data/spectra_data/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-size", type=int, default=200)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--alpha-tikh", type=float, default=0.1)
    p.add_argument("--alpha-lasso", type=float, default=0.1)
    p.add_argument("--downsample-factors", type=int, nargs='+', default=[1])
    p.add_argument("--out-dir", type=str, default="./results/experiment_1")
    p.add_argument("--normalize", type=bool, default=True)
    args = p.parse_args()

    # Device setup
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Retrieving dataset...")
    # Dataset
    DS = get_dataset(args.dataset)
    full_ds = DS(root=args.root)
    total = len(full_ds)
    train_n = total - args.val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds,
        [train_n, args.val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    I_val, S_val = dataset_to_numpy(val_ds)

    # Model load
    print("Loading model...")
    ModelClass = get_model(args.model)
    x0, y0 = full_ds[0]
    model = ModelClass(input_dim=len(x0), output_dim=len(y0)).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    
    # Discrete to Continuous
    num_cont_bins = 10000
    x_low, x_high = 1e-6, 9.5e-6
    x_meas_low, x_meas_high = 2e-6, 9e-6
    cont_vals = np.linspace(x_low, x_high, num_cont_bins)
    rnge = x_high - x_low
    low_idx = int(((x_meas_low - x_low) / rnge)*num_cont_bins) + 1 #inclusive
    high_idx = int(((x_meas_high - x_low) / rnge)*num_cont_bins) #exclusive


    alpha_lasso = args.alpha_lasso
    alpha_tikh = args.alpha_tikh

    wl_full = np.linspace(1e-6, 9.5e-6, 1000)
    downsample_arr = args.downsample_factors

    R = np.load("data/responsivity_data/processed/responsivity.npy")
    R = R.T

    for downsample in downsample_arr:
        
        wl_new = wl_full[::downsample]
        R_new = R[:, ::downsample]

        tik = TikhonovInverter(alpha=alpha_tikh); tik.set_matrix(R_new)
        las = LassoInverter(alpha=alpha_lasso)

        # create output dir for this resolution
        out_dir_ds = Path(args.out_dir) / f"downsample_{downsample}"
        out_dir_ds.mkdir(parents=True, exist_ok=True)

        # --- time the three reconstructions over the whole val set ---
        x_all = torch.from_numpy(I_val).float().to(device)
        model.eval()

        t0 = time.time()
        with torch.no_grad():
            y_all = model(x_all)
        recon_mod_ds = y_all.cpu().numpy()
        total_time_mod = time.time() - t0

        t0 = time.time()
        recon_tik_ds = np.stack(
            [tik.solve(b) for b in tqdm.tqdm(I_val, desc="Tikhonov solve")],
            axis=0
        )
        total_time_tik = time.time() - t0

        t0 = time.time()
        recon_las_ds = np.stack(
            [las.solve(R_new, b) for b in tqdm.tqdm(I_val, desc="Lasso solve")],
            axis=0
        )
        total_time_las = time.time() - t0

        # reset accumulators
        mse_m_total = mse_t_total = mse_l_total = 0.0
        sse_m = sse_t = sse_l = 0.0
        sst_total = 0.0
        

        for i in range(len(S_val)):
            y_m_g = gaussian_smooth(wl_full, recon_mod_ds[i], cont_vals)
            y_t_g = gaussian_smooth(wl_new, recon_tik_ds[i], cont_vals)
            y_l_g = gaussian_smooth(wl_new, recon_las_ds[i], cont_vals)
            y_gt_g = gaussian_smooth(wl_full, S_val[i], cont_vals)

            if args.normalize:
                y_m_g /= np.mean(y_m_g)
                y_t_g /= np.mean(y_t_g)
                y_l_g /= np.mean(y_l_g)
                y_gt_g /= np.mean(y_gt_g)

            # Plot and save
            fig, ax = plt.subplots()
            ax.plot(cont_vals, y_gt_g, label="Ground Truth")
            ax.plot(cont_vals, y_t_g, label="Tikhonov")
            ax.plot(cont_vals, y_l_g, label="Lasso")
            ax.plot(cont_vals, y_m_g, label="Model")
            ax.set_title(f"Sample {i} — downsample ×{downsample}")
            ax.set_xlabel("Wavelength (m)")
            ax.set_ylabel("Normalized Gaussian-smoothed response")
            ax.legend()
            fig.savefig(out_dir_ds / f"sample_{i}.png")
            plt.close(fig)

            # Compute MSE
            mse_m = MSE(y_gt_g, y_m_g, low_idx, high_idx)
            mse_t = MSE(y_gt_g, y_t_g, low_idx, high_idx)
            mse_l = MSE(y_gt_g, y_l_g, low_idx, high_idx)
            mse_m_total += mse_m
            mse_t_total += mse_t
            mse_l_total += mse_l

            # Accumulate for R^2
            resid_m = y_m_g[low_idx:high_idx] - y_gt_g[low_idx:high_idx]
            resid_t = y_t_g[low_idx:high_idx] - y_gt_g[low_idx:high_idx]
            resid_l = y_l_g[low_idx:high_idx] - y_gt_g[low_idx:high_idx]
            sse_m += np.sum(resid_m**2)
            sse_t += np.sum(resid_t**2)
            sse_l += np.sum(resid_l**2)

            diff_gt = y_gt_g[low_idx:high_idx] - np.mean(y_gt_g[low_idx:high_idx])
            sst_total += np.sum(diff_gt**2)

        # Summary
        n = len(S_val)

        avg_mse_m = mse_m_total / n
        avg_mse_t = mse_t_total / n
        avg_mse_l = mse_l_total / n

        r2_m = 1 - (sse_m / sst_total)
        r2_t = 1 - (sse_t / sst_total)
        r2_l = 1 - (sse_l / sst_total)

        avg_time_mod = total_time_mod / n
        avg_time_tik = total_time_tik / n
        avg_time_las = total_time_las / n

        print(f"\n=== Summary for resolution {len(S_val[0])} ===")
        print(f"Avg MSE → Model: {avg_mse_m:.4e}, Tikhonov: {avg_mse_t:.4e}, Lasso: {avg_mse_l:.4e}")
        print(f"    R² → Model: {r2_m:.4f},   Tikhonov: {r2_t:.4f},   Lasso: {r2_l:.4f}")
        print(f"Avg runtime per sample (s) → Model: {avg_time_mod:.4f}, Tikhonov: {avg_time_tik:.4f}, Lasso: {avg_time_las:.4f}\n")

    #TODO: write summary, add in range of alpha values!
    print("=== FINAL SUMMARY ===")
    print(f"Tikhonov: alpha = {alpha_tikh}, Lasso: alpha = {alpha_lasso}")
    print(f"Model: {args.model}, Dataset: {args.dataset}")
    print(f"Data normalized: {args.normalize}")
    print(f"Validation size: {args.val_size}, seed: {args.seed}")
    print(f"Plots saved to {args.out_dir}")
    
if __name__ == "__main__":
    main()
