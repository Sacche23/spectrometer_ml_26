#!/usr/bin/env python3
import argparse
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from src.models.lasso import LassoInverter
from src.models.tikhonov import TikhonovInverter

def main():
    p = argparse.ArgumentParser(description="Cross-validate alpha for Lasso and Tikhonov over multiple spectra.")
    p.add_argument("--resp", default="data/responsivity_data/processed/responsivity.npy",
                   help="Path to responsivity.npy (shape: m x n)")
    p.add_argument("--currents", default="data/spectra_data/processed/rand_sop/I_s42.npy",
                   help="Path to I.npy (shape: Nspec x m)")
    p.add_argument("--spectra", default="data/spectra_data/processed/rand_sop/S_s42.npy",
                   help="Path to S.npy (shape: Nspec x n) [optional for evaluation]")
    p.add_argument("--alphas", type=str, default="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1")
    p.add_argument("--folds", type=int, default=5, help="Number of folds")
    p.add_argument("--downsample", type=int, default=1,
                   help="Keep every Nth wavelength. 1 = full res, 2 = half res, etc.")
    p.add_argument("--subset", type=int, default=20,
                   help="Randomly choose this many spectra from the dataset. 0 = use all.")
    args = p.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]

    # Load data
    R = np.load(args.resp)       # (m, n)
    I = np.load(args.currents)   # (Nspec, m)
    try:
        S_all = np.load(args.spectra)  # (Nspec, n)
    except Exception:
        S_all = None

    # Orientation fix: ensure R rows == I.shape[1] (m)
    if R.shape[0] != I.shape[1]:
        if R.shape[1] == I.shape[1]:
            R = R.T
        else:
            raise ValueError(f"Incompatible shapes: R {R.shape}, I {I.shape}")

    # Downsample wavelength axis
    ds = max(1, int(args.downsample))
    if ds > 1:
        R = R[:, ::ds]
        if S_all is not None:
            S_all = S_all[:, ::ds]

    # Random subset selection
    Nspec = I.shape[0]
    if args.subset > 0 and args.subset < Nspec:
        idx = np.random.choice(Nspec, args.subset, replace=False, seed=42)
        I = I[idx]
        if S_all is not None:
            S_all = S_all[idx]
        Nspec = args.subset
        print(f"Using random subset of {Nspec} spectra")

    print(f"R shape after downsample: {R.shape}, I shape: {I.shape}, spectra: {None if S_all is None else S_all.shape}")

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=0)

    results = {}
    for method in ["lasso", "tikhonov"]:
        mse_scores = {a: [] for a in alphas}
        for train_idx, test_idx in kf.split(range(Nspec)):
            I_train, I_test = I[train_idx], I[test_idx]
            for alpha in alphas:
                if method == "lasso":
                    solver = LassoInverter(alpha=alpha)
                else:
                    solver = TikhonovInverter(alpha=alpha)
                    solver.set_matrix(R)

                preds = []
                for I_vec in I_test:
                    if method == "lasso":
                        S_hat = solver.solve(R, I_vec)
                    else:
                        S_hat = solver.solve(I_vec)
                    preds.append(S_hat)
                preds = np.array(preds)

                # Compare in current space
                I_pred = preds @ R.T
                mse = mean_squared_error(I_test, I_pred)
                mse_scores[alpha].append(mse)

        avg_mse = {a: float(np.mean(mse_scores[a])) for a in alphas}
        best_alpha = min(avg_mse, key=avg_mse.get)
        results[method] = (best_alpha, avg_mse)

    for method in results:
        best_alpha, avg_mse = results[method]
        print(f"\n=== {method.upper()} ===")
        for a in alphas:
            print(f"{a}\t{avg_mse[a]:.6e}")
        print(f"Best alpha: {best_alpha}")

if __name__ == "__main__":
    main()