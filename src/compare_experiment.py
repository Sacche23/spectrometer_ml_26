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

def gaussian_smooth(wl, vals, lam_cont):
    """
    Compute Gaussian-basis smoothing: sum vals * exp(-0.5*((lam_cont - wl_i)/sigma)**2)
    with sigma = (high - low)/len(wl).
    """
    sigma = (lam_cont[-1] - lam_cont[0]) / len(wl)
    cont = np.zeros_like(lam_cont)
    # vals = np.pad(vals, pad_width=1, mode='edge')
    # wl = np.pad(wl, pad_width=1, mode='constant', constant_values=(wl[0]-sigma, wl[-1]+sigma))
    for mu, v in zip(wl, vals):
        cont += v * (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * ((lam_cont - mu) / sigma) ** 2)
    return cont

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute R² for each sample in a batch (on continuous grid)."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=1)
    mean_true = np.mean(y_true, axis=1, keepdims=True)
    ss_tot = np.sum((y_true - mean_true) ** 2, axis=1)
    return 1 - ss_res / ss_tot

def MSE(S, S_gt, low_idx, high_idx):
    diff_squared = [(S[i] - S_gt[i])**2 for i in range(low_idx, high_idx)]
    print(f"DIFF: length={len(diff_squared)}")
    return np.mean(diff_squared)

def test():
    
    num_cont_bins = 5000

    x_low = 1e-6
    x_high = 9.5e-6
    small = np.linspace(x_low, x_high, 50)
    large = np.linspace(x_low, x_high, 500)
    extra = np.linspace(x_low, x_high, 1000)

    cont_vals = np.linspace(x_low, x_high, num_cont_bins)

    x_meas_low = 2e-6
    x_meas_high = 9e-6

    range = x_high - x_low
    low_idx = int(((x_meas_low - x_low) / range)*num_cont_bins) + 1 #inclusive
    high_idx = int(((x_meas_high - x_low) / range)*num_cont_bins) #exclusive
    print(cont_vals[low_idx])
    print(cont_vals[high_idx])
    arr_new = cont_vals[low_idx:high_idx]
    print(len(arr_new), arr_new[-1])



    sin_small_d = np.sin(2*small/range) + 2
    sin_large_d = np.sin(2*large/range) + 2
    sin_extra_d = np.sin(2*extra/range) + 2

    sin_small_c = gaussian_smooth(small, sin_small_d, cont_vals)
    sin_large_c = gaussian_smooth(large, sin_large_d, cont_vals)
    sin_extra_c = gaussian_smooth(extra, sin_extra_d, cont_vals)
    


    plt.plot(cont_vals, sin_small_c)
    plt.ylabel("small")
    plt.plot(cont_vals, sin_large_c)
    plt.ylabel("large")
    plt.plot(cont_vals, sin_extra_c)
    plt.ylabel("extra")
    plt.plot(cont_vals, np.sin((2/range) * cont_vals) + 2)
    plt.ylabel("real")
    plt.show()

    plt.plot(arr_new, sin_small_c[low_idx:high_idx])
    plt.ylabel("large")
    plt.plot(arr_new, sin_large_c[low_idx:high_idx])
    plt.ylabel("large")
    plt.plot(arr_new, sin_extra_c[low_idx:high_idx])
    plt.ylabel("large")
    plt.plot(arr_new, np.sin(2*arr_new/range) + 1)
    plt.ylabel("real")
    plt.show()

    print(np.sum(sin_small_c)/10000)
    print(np.sum(sin_large_c)/10000)
    print(np.sum(sin_extra_c)/10000)


# downsample_arr = [0,1,2,3,4,5]

# wavelengths_total = np.load("data/responsivity_data/wavelengths.npy")
# responsivity_total = np.load("data/responsivity_data/responsivity.npy")




def main():



    p = argparse.ArgumentParser(description="Compare methods with gaussian-basis reconstruction at multiple resolutions")
    p.add_argument("--dataset", "-d", required=True)
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--checkpoint", "-c", required=True)
    p.add_argument("--root", type=str, default="data/spectra_data/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-size", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--alpha-tikh", type=float, default=0.1)
    p.add_argument("--alpha-lasso", type=float, default=0.1)
    p.add_argument("--n-plots", type=int, default=9)
    p.add_argument("--downsample-factors", type=int, nargs='+', default=[1])
    p.add_argument("--train-subset-size", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()


    # DEVICE SETUP
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # DATASETS
    DS = get_dataset(args.dataset)
    full_ds = DS(root=args.root)
    total = len(full_ds)
    train_n = total - args.val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds,
        [train_n, args.val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_eval = Subset(train_ds, range(args.train_subset_size)) if args.train_subset_size > 0 else train_ds

    # Discrete to Continuous

    num_cont_bins = 10000
    x_low, x_high = 1e-6, 9.5e-6
    x_meas_low, x_meas_high = 2e-6, 9e-6
    cont_vals = np.linspace(x_low, x_high, num_cont_bins)
    range = x_high - x_low
    low_idx = int(((x_meas_low - x_low) / range)*num_cont_bins) + 1 #inclusive
    high_idx = int(((x_meas_high - x_low) / range)*num_cont_bins) #exclusive
    print(high_idx, low_idx)


    wl = np.linspace(1e-6, 9.5e-6, 1000)

    x, y  = val_ds[0]
    x_np, y_np = x.numpy(), y.numpy()
    y_g = gaussian_smooth(wl, y_np, cont_vals)

    # print(y_np)

    print(MSE(y_g, y_g+0.01, low_idx, high_idx))

    # print(len(train_eval))
    # count = 0
    # for x, y in train_eval:
    #     print(f"x: {x[0]}, y:{y[0]}")
    #     count += 1
    # print(count)
    


    




if __name__ == "__main__":
    main()