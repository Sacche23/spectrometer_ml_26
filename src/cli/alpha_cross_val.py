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
    p.add_argument("--plot", action="store_true", help="Plot predicted spectra for each alpha (test set).")
    p.add_argument("--noise-std", type=float, default=0.0,
                   help="Std. dev. of Gaussian noise to add to currents in CV (0.0 = no noise).")
    p.add_argument("--normalize", action="store_true", default=True,
                   help="Normalize spectra before MSE scoring (RMS normalization).")
    args = p.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]

    # Load data
    R = np.load(args.resp)
    I = np.load(args.currents)
    try:
        S_all = np.load(args.spectra)
    except Exception:
        S_all = None

    # Pick metric default
    if args.metric is None:
        args.metric = "spectra" if S_all is not None else "currents"
    print(f"Scoring metric: {args.metric}")

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

            # Add noise in current space. Scale is independent of downsample thanks to R scaling above.
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

                # Optional normalization for spectra-space scoring
                if args.metric == "spectra" and S_test is not None and args.normalize:
                    preds_for_score = rms_normalize(preds)
                    S_test_for_score = rms_normalize(S_test)
                else:
                    preds_for_score = preds
                    S_test_for_score = S_test

                # Base output dir
                base_out_dir = Path("outputs/cross_val_images")

                # Auto-increment test number
                existing_tests = sorted(base_out_dir.glob("test_*"))
                test_number = len(existing_tests) + 1
                this_test_dir = base_out_dir / f"test_{test_number}"
                this_test_dir.mkdir(parents=True, exist_ok=True)

                print(f"Saving plots to: {this_test_dir}")

                # Plot spectra if requested
                if args.plot and args.metric == "spectra" and S_test is not None:
                    for idx_plot in range(min(5, len(S_test))):  # plot up to 5 examples
                        plt.figure()
                        plt.plot(S_test_for_score[idx_plot], label="True")
                        plt.plot(preds_for_score[idx_plot], label=f"Pred alpha={alpha}")
                        plt.legend()
                        plt.savefig(this_test_dir / f"fold{fold}_ex{idx_plot}_alpha{alpha}.png")
                        plt.close()

                # Scoring
                if args.metric == "spectra" and S_test is not None:
                    mse = mean_squared_error(S_test_for_score, preds_for_score)
                else:
                    # Compare in current space (not done usually)
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

if __name__ == "__main__":
    main()