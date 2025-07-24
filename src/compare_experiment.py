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
    p.add_argument("--val-size", type=int, default=10)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--alpha-tikh", type=float, default=0.1)
    p.add_argument("--alpha-lasso", type=float, default=0.1)
    p.add_argument("--downsample-factors", type=int, nargs='+', default=[1])
    p.add_argument("--out-dir", type=str, default=".")
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
    downsample_arr = [1, 2, 5, 10, 12, 15, 20, 25, 30]
    R = np.load("data/responsivity_data/processed/responsivity.npy")
    R = R.T

    for downsample in downsample_arr:
        
        wl_new = wl_full[::downsample]
        R_new = R[:, ::downsample]    
        print(f"RESOLUTION: {len(wl_new)} POINTS")

        tik = TikhonovInverter(alpha=alpha_tikh); tik.set_matrix(R_new)
        las = LassoInverter(alpha=alpha_lasso)

        x_all = torch.from_numpy(I_val).float().to(device)
        model.eval()
        with torch.no_grad():
            y_all = model(x_all)
        recon_mod_ds = y_all.cpu().numpy()

        recon_tik_ds = np.stack(
            [tik.solve(b) for b in tqdm.tqdm(I_val, desc="Tikhonov solve")],
            axis=0
        )
        recon_las_ds = np.stack(
            [las.solve(R_new, b) for b in tqdm.tqdm(I_val, desc="Lasso solve")],
            axis=0
        )

        mse_m_total = 0
        mse_t_total = 0
        mse_l_total = 0
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

            mse_m = MSE(y_gt_g, y_m_g, low_idx, high_idx)
            mse_t = MSE(y_gt_g, y_t_g, low_idx, high_idx)
            mse_l = MSE(y_gt_g, y_l_g, low_idx, high_idx)
            mse_m_total += mse_m
            mse_t_total += mse_t
            mse_l_total += mse_l


        print(f"TOTAL MSE: Model={mse_m_total/len(recon_mod_ds)}, Tikhonov={mse_t_total/len(recon_tik_ds)}, Lasso={mse_l_total/len(recon_las_ds)}\n")

    
if __name__ == "__main__":
    main()