# compare_experiment
import numpy as np
import torch
import argparse
import random
import json
import csv
from torch.utils.data import DataLoader
from src.datasets import get_dataset
from src.models.model import get_model
import matplotlib.pyplot as plt
from pathlib import Path


# ====================================================================================
# Helper Functions
# ====================================================================================

def gaussian_smooth(wl, vals, lam_cont):
    """
    Convert a discrete spectrum vector back into a continuous curve by placing
    a small Gaussian "bump" at each wavelength point and summing them up.

    Think of it like this: instead of a bar chart with one bar per wavelength,
    we replace each bar with a smooth bell curve centered at that wavelength.
    Summing all the bell curves gives a smooth, continuous spectrum.

    wl       : 1D array of wavelength values (the x-axis positions of the bars)
    vals     : 1D array of spectrum values   (the heights of the bars)
    lam_cont : 1D array of fine wavelength grid to evaluate the smooth curve on
    """
    sigma = (lam_cont[-1] - lam_cont[0]) / len(wl)
    cont = np.zeros_like(lam_cont)
    for mu, v in zip(wl, vals):
        cont += v * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((lam_cont - mu) / sigma) ** 2)
    return cont


def MSE(pred, gt, low_idx, high_idx):
    """
    Mean squared error between pred and gt, computed only over the
    wavelength range [low_idx, high_idx]. Only score the model over the 
    2-9 µm operational range, not the full 1-9.5 µm grid where the 
    device has poor responsivity.
    """
    diff_squared = [(pred[i] - gt[i]) ** 2 for i in range(low_idx, high_idx)]
    return np.mean(diff_squared)


def spectral_angle(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Spectral Angle Mapper (SAM) — measures the angle between two
    spectrum vectors in degrees. 0° means perfect match regardless
    of scale; 90° means completely orthogonal (worst case).
    Works on raw numpy arrays.
    """
    pred = pred.flatten()
    gt   = gt.flatten()
    dot  = np.dot(pred, gt)
    norm = np.linalg.norm(pred) * np.linalg.norm(gt)
    if norm == 0:
        return 0.0
    cos_theta = np.clip(dot / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def peak_wavelength_error(pred: np.ndarray, gt: np.ndarray,
                          wl_grid: np.ndarray) -> float:
    """
    Finds the wavelength of the brightest peak in each spectrum
    and returns the absolute difference in µm.
    wl_grid: 1D array of wavelength values in metres,
             same length as pred and gt.
    """
    pred_peak_um = wl_grid[np.argmax(pred)] * 1e6   # convert m → µm
    gt_peak_um   = wl_grid[np.argmax(gt)]   * 1e6
    return float(abs(pred_peak_um - gt_peak_um))

def dataset_to_numpy(ds):
    """Load an entire dataset split into numpy arrays in one shot."""
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    x_tensor, y_tensor = next(iter(loader))
    return x_tensor.detach().cpu().numpy(), y_tensor.detach().cpu().numpy()

def smooth_batch(wl, spectra, lam_cont):
    """
    Apply gaussian_smooth() to every spectrum in a batch.

    Parameters:
        wl:       wavelength grid (output_dim,)
        spectra:  spectra array (N, output_dim)
        lam_cont: continuous wavelength grid

    Returns:
        smoothed spectra array (N, len(lam_cont))
    """
    return np.array([
        gaussian_smooth(wl, spectrum, lam_cont)
        for spectrum in spectra
    ])

def load_model(model_name, checkpoint_path, input_dim, output_dim, device):
    """
    Instantiate a model by name and load its weights from a checkpoint file.
    Returns the model in eval() mode (inference mode, dropout disabled).
    """
    ModelClass = get_model(model_name)
    model = ModelClass(input_dim=input_dim, output_dim=output_dim).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    # Checkpoints saved by train.py are dicts containing "model_state_dict".
    # Handle both that format and raw state dicts for compatibility.
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    print(f"  Loaded {model_name} from {checkpoint_path}")
    return model

def plot_loss_comparison(experiment_dir: Path, model_names: list, comparison_dir: Path):
    """
    Reads each model's loss_history.json and overlays their validation
    loss curves on ONE shared plot, so you can visually compare how fast
    and how far each architecture's training converged.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name in model_names:
        history_path = experiment_dir / model_name / "evaluation" / "loss_history.json"
        if not history_path.exists():
            print(f"  WARNING: no loss_history.json for '{model_name}', skipping in comparison plot.")
            continue

        with open(history_path) as f:
            history = json.load(f)

        ax.plot(history["epoch"], history["val_loss"], label=model_name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE Loss")
    ax.set_yscale("log")
    ax.set_title("Validation Loss Comparison Across Models")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    out_path = comparison_dir / "loss_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss comparison plot to {out_path}")

# ====================================================================================
# Main
# ====================================================================================

def main():

    p = argparse.ArgumentParser(
        description="Compare multiple NN models on the validation set. "
                    "Reads best.pth from <experiment-dir>/<model>/checkpoints/best.pth "
                    "and writes results to <experiment-dir>/_comparison/."
    )

    # Required arguments
    p.add_argument("--experiment-dir", "-e", required=True,
                   help="Path to the experiment directory, e.g. experiments/20260710_1519_rand_sop_seed42")
    p.add_argument("--models", "-m", nargs="+", required=True,
                   help="One or more registered model names to compare, e.g. --models dnn cnn2")
    p.add_argument("--dataset", "-d", required=True,
                   help="Registered dataset name, e.g. rand_sop")
    p.add_argument("--responsivity", required=True,
                   help="Path to responsivity matrix .npy file")

    # Optional arguments (match defaults used throughout the rest of the codebase)
    p.add_argument("--root", type=str, default="data/spectra_data/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-size", type=int, default=200)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--normalize", action="store_true", default=True,
                   help="Normalize each smoothed spectrum to unit mean before scoring (default: True)")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip saving per-sample plots (faster, useful on cluster)")
    p.add_argument("--noise-std", type=float, default=0.0,
                   help="Std dev of Gaussian noise added to evaluation currents. Default 0.0 (clean).")

    args = p.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    # Reproducibility — same seeds as training so the validation split
    # is identical to what was held out during training.
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    experiment_dir = Path(args.experiment_dir)
    comparison_dir = experiment_dir / "_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    print(f"\nLoading dataset '{args.dataset}'...")
    DS = get_dataset(args.dataset)
    full_ds = DS(root=args.root, seed=args.seed)
    total = len(full_ds)
    train_n = total - args.val_size

    _, val_ds = torch.utils.data.random_split(
        full_ds,
        [train_n, args.val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    I_val, S_val = dataset_to_numpy(val_ds)
    print(f"  Validation set: {len(S_val)} samples")
    print(f"  Input dim: {I_val.shape[1]}, Output dim: {S_val.shape[1]}")

    # Optionally inject noise into evaluation currents.
    # This tests how robust each model is to measurement noise.
    if args.noise_std > 0:
        noise = np.random.normal(0.0, args.noise_std, size=I_val.shape)
        I_eval = I_val + noise
        print(f"  Added Gaussian noise: std = {args.noise_std}")
    else:
        I_eval = I_val

    # ------------------------------------------------------------------
    # Load models and run inference
    # ------------------------------------------------------------------

    input_dim  = I_val.shape[1]
    output_dim = S_val.shape[1]
    x_all = torch.from_numpy(I_eval).float().to(device)

    # ------------------------------------------------------------------
    # Continuous wavelength grid for Gaussian-smoothed evaluation
    # ------------------------------------------------------------------

    num_cont_bins = 10000
    x_low,      x_high      = 1e-6, 9.5e-6   # full grid: 1 to 9.5 µm
    x_meas_low, x_meas_high = 2e-6, 9e-6     # scored range: 2 to 9 µm

    cont_vals = np.linspace(x_low, x_high, num_cont_bins)
    rnge      = x_high - x_low
    low_idx   = int(((x_meas_low  - x_low) / rnge) * num_cont_bins) + 1  # inclusive
    high_idx  = int(((x_meas_high - x_low) / rnge) * num_cont_bins)      # exclusive

    wl_full = np.linspace(x_low, x_high, output_dim)
    S_val_smoothed = smooth_batch(wl_full, S_val, cont_vals)

    # results dict: model_name -> {"predictions": np.array, "epoch": int}
    model_results = {}

    print("\nLoading models...")
    for model_name in args.models:
        ckpt_path = experiment_dir / model_name / "checkpoints" / "best.pth"

        if not ckpt_path.exists():
            print(f"  WARNING: checkpoint not found for '{model_name}' at {ckpt_path}. Skipping.")
            continue

        model = load_model(model_name, ckpt_path, input_dim, output_dim, device)

        # Load epoch number from checkpoint for reporting
        state = torch.load(ckpt_path, map_location=device)
        epoch = state.get("epoch", "?") if isinstance(state, dict) else "?"

        with torch.no_grad():
            preds = model(x_all).cpu().numpy()

        preds_smoothed = smooth_batch(wl_full, preds, cont_vals)

        model_results[model_name] = {
            "predictions": preds,
            "predictions_smoothed": preds_smoothed,
            "epoch": epoch,
        }

    if not model_results:
        raise RuntimeError("No models were loaded successfully. Check checkpoint paths.")

    loaded_models = list(model_results.keys())
    print(f"\nComparing: {loaded_models}")


    # ------------------------------------------------------------------
    # Per-sample scoring
    # ------------------------------------------------------------------

    # Accumulators: one entry per model
    mse_totals = {m: 0.0 for m in loaded_models}
    sse        = {m: 0.0 for m in loaded_models}
    sam_totals = {m: 0.0 for m in loaded_models}  
    pwe_totals = {m: 0.0 for m in loaded_models}  
    sst_total  = 0.0

    # Output folder for per-sample plots
    plots_dir = comparison_dir / "sample_plots"
    if not args.no_plot:
        plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nScoring {len(S_val)} validation samples...")

    for i in range(len(S_val)):

        if i % 50 == 0:
            print(f"Processed {i}/{len(S_val)} samples")

        # Ground truth smoothed
        y_gt_g = S_val_smoothed[i]

        # Each model's smoothed prediction
        smoothed = {}
        for model_name in loaded_models:
            smoothed[model_name] = model_results[model_name]["predictions_smoothed"][i]

        # Normalize all curves to unit mean over the scored range so that
        # MSE reflects shape similarity, not absolute scale differences.
        if args.normalize:
            y_gt_g = y_gt_g / np.mean(y_gt_g)
            for model_name in loaded_models:
                smoothed[model_name] = smoothed[model_name] / np.mean(smoothed[model_name])

        # Accumulate MSE and SSE for R²
        for model_name in loaded_models:
            mse_totals[model_name] += MSE(smoothed[model_name], y_gt_g, low_idx, high_idx)
            resid = smoothed[model_name][low_idx:high_idx] - y_gt_g[low_idx:high_idx]
            sse[model_name] += np.sum(resid ** 2)

            sam_totals[model_name] += spectral_angle(
            smoothed[model_name][low_idx:high_idx],
            y_gt_g[low_idx:high_idx]
            )
            
            scored_cont = cont_vals[low_idx:high_idx]
            pred_peak_um = scored_cont[np.argmax(smoothed[model_name][low_idx:high_idx])] * 1e6
            gt_peak_um   = scored_cont[np.argmax(y_gt_g[low_idx:high_idx])]               * 1e6
            pwe_totals[model_name] += abs(pred_peak_um - gt_peak_um)

        diff_gt = y_gt_g[low_idx:high_idx] - np.mean(y_gt_g[low_idx:high_idx])
        sst_total += np.sum(diff_gt ** 2)

        # Per-sample plot: ground truth + all models on one axes
        if not args.no_plot:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(cont_vals * 1e6, y_gt_g, label="Ground Truth", color="black", linewidth=1.5)
            for model_name in loaded_models:
                ax.plot(cont_vals * 1e6, smoothed[model_name], label=model_name, linewidth=1.2)
            ax.axvspan(x_meas_low * 1e6, x_meas_high * 1e6, alpha=0.07, color="blue",
                       label="Scored range (2–9 µm)")
            ax.set_xlabel("Wavelength (µm)")
            ax.set_ylabel("Normalized response")
            ax.set_title(f"Sample {i}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(plots_dir / f"sample_{i:04d}.png", dpi=80)
            plt.close(fig)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    n = len(S_val)
    avg_mse = {m: mse_totals[m] / n         for m in loaded_models}
    r2      = {m: 1 - sse[m] / sst_total    for m in loaded_models}
    avg_sam = {m: sam_totals[m] / n         for m in loaded_models}   
    avg_pwe = {m: pwe_totals[m] / n         for m in loaded_models}   

    print("\n" + "=" * 68)
    print("COMPARISON SUMMARY")
    print("=" * 68)
    print(f"{'Model':<12}  {'Avg MSE':>12}  {'R²':>8}  {'SAM (°)':>9}  {'PWE (µm)':>10}  {'Best Epoch':>10}")
    print("-" * 68)
    for model_name in loaded_models:
        print(f"{model_name:<12}  {avg_mse[model_name]:>12.4e}  "
            f"{r2[model_name]:>8.4f}  "
            f"{avg_sam[model_name]:>9.3f}  "
            f"{avg_pwe[model_name]:>10.4f}  "
            f"{model_results[model_name]['epoch']:>10}")
    print("=" * 68)

    # ------------------------------------------------------------------
    # Save metrics.json
    # ------------------------------------------------------------------

    metrics = {
        "experiment_dir": str(experiment_dir),
        "dataset": args.dataset,
        "seed": args.seed,
        "val_size": args.val_size,
        "noise_std": args.noise_std,
        "normalized": args.normalize,
        "scored_range_um": [x_meas_low * 1e6, x_meas_high * 1e6],
        "models": {
            model_name: {
                "best_epoch": model_results[model_name]["epoch"],
                "avg_mse": avg_mse[model_name],
                "r2": r2[model_name],
                "avg_sam_deg": avg_sam[model_name],
                "avg_pwe_um":  avg_pwe[model_name],
            }
            for model_name in loaded_models
        }
    }

    metrics_path = comparison_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved metrics to {metrics_path}")

    # ------------------------------------------------------------------
    # Save metrics.csv (one row per model, easy to open in Excel)
    # ------------------------------------------------------------------

    csv_path = comparison_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "avg_mse", "r2", "avg_sam_deg", "avg_pwe_um", "best_epoch"])        
        writer.writeheader()
        for model_name in loaded_models:
            writer.writerow({
                "model":       model_name,
                "avg_mse":     avg_mse[model_name],
                "r2":          r2[model_name],
                "avg_sam_deg": avg_sam[model_name],
                "avg_pwe_um":  avg_pwe[model_name],
                "best_epoch":  model_results[model_name]["epoch"],
            })
    print(f"Saved CSV to {csv_path}")

    # ------------------------------------------------------------------
    # Comparison bar chart
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    models_list = loaded_models
    mse_vals    = [avg_mse[m] for m in models_list]
    r2_vals     = [r2[m]      for m in models_list]
    sam_vals    = [avg_sam[m] for m in models_list]
    pwe_vals    = [avg_pwe[m] for m in models_list]
    colors      = plt.cm.tab10.colors[:len(models_list)]

    axes[0].bar(models_list, mse_vals, color=colors)
    axes[0].set_title("Avg MSE (lower is better)")
    axes[0].set_ylabel("MSE")
    axes[0].set_xlabel("Model")

    axes[1].bar(models_list, r2_vals, color=colors)
    axes[1].set_title("R² (higher is better)")
    axes[1].set_ylabel("R²")
    axes[1].set_xlabel("Model")
    axes[1].set_ylim(min(0, min(r2_vals)) - 0.05, 1.05)

    axes[2].bar(models_list, sam_vals, color=colors)
    axes[2].set_title("Avg SAM in degrees (lower is better)")
    axes[2].set_ylabel("SAM (°)")

    axes[3].bar(models_list, pwe_vals, color=colors)
    axes[3].set_title("Avg Peak Wavelength Error (lower is better)")
    axes[3].set_ylabel("PWE (µm)")

    fig.suptitle(f"Model Comparison — {args.dataset}, seed {args.seed}", fontsize=12)
    fig.tight_layout()
    chart_path = comparison_dir / "comparison.png"
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"Saved chart  to {chart_path}")

    # ------------------------------------------------------------------
    # Overlaid validation loss curves
    # ------------------------------------------------------------------
    plot_loss_comparison(experiment_dir, loaded_models, comparison_dir)

    # ------------------------------------------------------------------
    # summary.txt
    # ------------------------------------------------------------------

    summary_path = comparison_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Experiment : {experiment_dir}\n")
        f.write(f"Dataset    : {args.dataset}\n")
        f.write(f"Seed       : {args.seed}\n")
        f.write(f"Val size   : {args.val_size}\n")
        f.write(f"Noise std  : {args.noise_std}\n")
        f.write(f"Normalized : {args.normalize}\n")
        f.write(f"Scored range: {x_meas_low*1e6:.1f}-{x_meas_high*1e6:.1f} um\n\n")
        f.write(f"{'Model':<12}  {'Avg MSE':>12}  {'R^2':>8}  "
                f"{'SAM(deg)':>10}  {'PWE(um)':>9}  {'Best Epoch':>10}\n")
        f.write("-" * 70 + "\n")
        for model_name in loaded_models:
            f.write(f"{model_name:<12}  {avg_mse[model_name]:>12.4e}  "
                    f"{r2[model_name]:>8.4f}  "
                    f"{avg_sam[model_name]:>10.3f}  "
                    f"{avg_pwe[model_name]:>9.4f}  "
                    f"{model_results[model_name]['epoch']:>10}\n")
        best = min(loaded_models, key=lambda m: avg_mse[m])
        f.write(f"\nBest model by MSE: {best}\n")

    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()