import argparse
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from src.models.lasso import LassoInverter
from src.models.tikhonov import TikhonovInverter
import matplotlib.pyplot as plt
import os
from pathlib import Path

def rms_normalize(arr):
    """Normalize each row by its RMS value."""
    norm = np.sqrt(np.mean(arr**2, axis=1, keepdims=True))
    norm[norm == 0] = 1.0  # Do not divide by zero
    return arr / norm

def main():
    p = argparse.ArgumentParser(description="Cross-validate alpha for Lasso and Tikhonov over multiple spectra.")
    p.add_argument("--resp", default="data/responsivity_data/processed/responsivity.npy")
    p.add_argument("--currents", default="data/spectra_data/processed/rand_sop/I_s42.npy")
    p.add_argument("--spectra", default="data/spectra_data/processed/rand_sop/S_s42.npy")
    p.add_argument("--alphas", type=str, default="1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--downsample", type=int, default=1, help="Keep every Nth wavelength; scales R by N to preserve current scale.")
    p.add_argument("--subset", type=int, default=20)
    p.add_argument("--metric", choices=["spectra", "currents"], default=None,
                   help="Which space to compute MSE in. Default: spectra if S.npy exists, else currents.")
    p.add_argument("--plot", action="store_true", help="Plot predicted spectra for final best alpha (per method).")
    p.add_argument("--plot-n", type=int, default=5, help="How many examples to plot for the final result.")
    p.add_argument("--outdir", type=str, default="outputs/cross_val_images",
                   help="Where to save plots (top dir). No new dirs are created unless saving plots.")
    p.add_argument("--no-mkdir", action="store_true",
                   help="Do not create directories; show plots interactively instead of saving.")
    p.add_argument("--noise-std", type=float, default=0.0,
                   help="Std. dev. of Gaussian noise to add to currents in CV (0.0 = no noise).")
    p.add_argument("--normalize", action="store_true", default=True,
                   help="Normalize spectra before MSE scoring (RMS normalization).")
    args = p.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]

    # Load data
    R = np.load(args.resp)       # (m, n)
    I = np.load(args.currents)   # (Nspec, m)
    try:
        S_all = np.load(args.spectra)  # (Nspec, n)
    except Exception:
        S_all = None

    # Pick metric default
    if args.metric is None:
        args.metric = "spectra" if S_all is not None else "currents"
    print(f"Scoring metric: {args.metric}")

    # Orientation fix so that R rows == I.shape[1] (m)
    if R.shape[0] != I.shape[1]:
        if R.shape[1] == I.shape[1]:
            R = R.T
        else:
            raise ValueError(f"Incompatible shapes: R {R.shape}, I {I.shape}")

    # Downsample wavelength axis + SCALE R by ds to preserve current scale
    ds = max(1, int(args.downsample))
    if ds > 1:
        R = R[:, ::ds] * ds
        if S_all is not None:
            S_all = S_all[:, ::ds]
        print(f"Downsample factor {ds}: R columns sliced ::{ds} and R scaled by {ds} to keep I scale consistent.")

    # Subset selection
    Nspec = I.shape[0]
    np.random.seed(42)
    if args.subset > 0 and args.subset < Nspec:
        idx = np.random.choice(Nspec, args.subset, replace=False)
        I = I[idx]
        if S_all is not None:
            S_all = S_all[idx]
        Nspec = args.subset
        print(f"Using random subset of {Nspec} spectra")

    # CV loop
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=0)
    results = {}

    for method in ["lasso", "tikhonov"]:
        print(f"\nAnalyzing {method}:")
        mse_scores = {a: [] for a in alphas}

        for fold, (train_idx, test_idx) in enumerate(kf.split(range(Nspec)), start=1):
            I_train, I_test = I[train_idx], I[test_idx]
            S_train = S_all[train_idx] if S_all is not None else None
            S_test = S_all[test_idx] if S_all is not None else None

            # Add noise if requested
            if args.noise_std > 0:
                I_train = I_train + np.random.normal(0, args.noise_std, I_train.shape)
                I_test = I_test + np.random.normal(0, args.noise_std, I_test.shape)

            for alpha in alphas:
                if method == "lasso":
                    solver = LassoInverter(alpha=alpha)
                else:
                    solver = TikhonovInverter(alpha=alpha)
                    solver.set_matrix(R)

                preds = []
                for i_vec in I_test:
                    if method == "lasso":
                        S_hat = solver.solve(R, i_vec)
                    else:
                        S_hat = solver.solve(i_vec)
                    preds.append(S_hat)
                preds = np.array(preds)

                if args.metric == "spectra" and S_test is not None and args.normalize:
                    preds_for_score = rms_normalize(preds)
                    S_test_for_score = rms_normalize(S_test)
                else:
                    preds_for_score = preds
                    S_test_for_score = S_test

                if args.metric == "spectra" and S_test is not None:
                    mse = mean_squared_error(S_test_for_score, preds_for_score)
                else:
                    I_pred = preds @ R.T
                    mse = mean_squared_error(I_test, I_pred)
                mse_scores[alpha].append(mse)

        avg_mse = {a: float(np.mean(mse_scores[a])) for a in alphas}
        best_alpha = min(avg_mse, key=avg_mse.get)
        results[method] = (best_alpha, avg_mse)

    # Print results
    for method in results:
        best_alpha, avg_mse = results[method]
        print(f"\n=== {method.upper()} ===")
        for a in alphas:
            print(f"{a}\t{avg_mse[a]:.6e}")
        print(f"Best alpha: {best_alpha}")

    # Modified plotting section: plot first index for all alphas
    if args.plot and args.metric == "spectra" and S_all is not None:
        save_plots = not args.no_mkdir
        if save_plots:
            base_out_dir = Path(args.outdir)
            base_out_dir.mkdir(parents=True, exist_ok=True)
            existing_nums = [
                int(p.name.split("_")[1])
                for p in base_out_dir.glob("test_*")
                if p.name.startswith("test_") and p.name.split("_")[1].isdigit()
            ]
            next_num = max(existing_nums, default=0) + 1
            out_dir = base_out_dir / f"test_{next_num}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving plots to: {out_dir}")
        else:
            print("Showing plots interactively (no directories created).")

        i = 0  # always plot the first example
        for method in ["lasso", "tikhonov"]:
            for alpha in alphas:
                if method == "lasso":
                    solver = LassoInverter(alpha=alpha)
                    S_hat = solver.solve(R, I[i])
                else:
                    solver = TikhonovInverter(alpha=alpha)
                    solver.set_matrix(R)
                    S_hat = solver.solve(I[i])

                if args.normalize:
                    pred_vis = rms_normalize(S_hat[None, :])[0]
                    true_vis = rms_normalize(S_all[i][None, :])[0]
                else:
                    pred_vis = S_hat
                    true_vis = S_all[i]

                plt.figure()
                plt.plot(true_vis, label="True")
                plt.plot(pred_vis, label=f"{method.capitalize()} (alpha={alpha:g})")
                plt.legend()
                plt.title(f"{method} — example {i}")
                plt.xlabel("Wavelength index")
                plt.ylabel("Normalized amplitude" if args.normalize else "Amplitude")

                if save_plots:
                    plt.savefig(out_dir / f"{method}_ex{i}_alpha{alpha}.png")
                    plt.close()
                else:
                    plt.show()

if __name__ == "__main__":
    main()
