# compare_experiment
import numpy as np
import torch
import argparse
import random
import time
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
    Mean squared error between pred and gt, computed only over
    [low_idx, high_idx] -- the wavelength band where the model is
    actually meant to be scored (typically the device's usable
    responsivity band, clipped to the dataset's coverage).
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

def benchmark_inference(model, input_dim, n_warmup=50, n_reps=100, n_threads=1):
    """
    Measures single-sample inference latency on CPU.

    Designed to reflect real-world deployment conditions for a miniaturized
    spectrometer, where the target platform is an embedded CPU or FPGA. 
    - CPU only
    - Batch size (1)
    - Fixed thread count (single processor, 1)
    - 50 warm-ups
    - 100 timed repetitions, report median
    Returns a dict with median, std, min, max latency in ms.
    """
    cpu = torch.device("cpu")

    # Move model to CPU for benchmarking — keep original device for inference
    model_cpu = model.to(cpu)
    model_cpu.eval()

    # Lock thread count — must be restored afterward
    original_threads = torch.get_num_threads()
    torch.set_num_threads(n_threads)

    # Single random sample — batch size 1
    dummy = torch.randn(1, input_dim, device=cpu)

    # Warm-up: triggers JIT, fills CPU instruction cache, warms memory allocator
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model_cpu(dummy)

    # Timed repetitions
    times_ms = []
    with torch.no_grad():
        for _ in range(n_reps):
            t0 = time.perf_counter()
            _ = model_cpu(dummy)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

    # Restore original thread count so the rest of the script is unaffected
    torch.set_num_threads(original_threads)

    times_ms = np.array(times_ms)

    return {
        "median_ms":  float(np.median(times_ms)),
        "q25_ms":     float(np.percentile(times_ms, 25)),
        "q75_ms":     float(np.percentile(times_ms, 75)),
        "std_ms":     float(np.std(times_ms)),
        "min_ms":     float(np.min(times_ms)),
        "max_ms":     float(np.max(times_ms)),
        "n_warmup":   n_warmup,
        "n_reps":     n_reps,
        "n_threads":  n_threads,
    }

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

def run_overfitting_analysis(experiment_dir: Path, model_names: list, comparison_dir: Path):
    """
    For each model, reads its metrics.json and loss_history.json and computes:
      - train/val MSE gap and ratio (how much worse val is than train)
      - best epoch (when val loss was lowest)
      - overfit % (how much val loss rose from its best to the final epoch)

    Prints a summary table and saves overfitting_summary to the comparison dir.
    """
    rows = []

    for model_name in model_names:
        model_dir = experiment_dir / model_name

        # ---- Read metrics.json for train/val MSE ----
        metrics_file = model_dir / "evaluation" / "metrics.json"
        if not metrics_file.exists():
            print(f"  WARNING: no metrics.json for '{model_name}', skipping.")
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        train_mse = metrics["train"]["mse"]
        val_mse   = metrics["validation"]["mse"]
        mse_gap   = val_mse - train_mse
        ratio     = val_mse / train_mse if train_mse > 0 else float("inf")

        # ---- Read loss_history.json for convergence info ----
        history_file = model_dir / "evaluation" / "loss_history.json"
        best_epoch = metrics.get("epoch", "?")
        overfit_pct = None

        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)

            val_loss = history["val_loss"]
            epochs   = history["epoch"]

            best_idx   = min(range(len(val_loss)), key=lambda i: val_loss[i])
            best_epoch = epochs[best_idx]
            best_val   = val_loss[best_idx]
            final_val  = val_loss[-1]

            if best_val > 0:
                overfit_pct = (final_val - best_val) / best_val * 100

        rows.append({
            "model":       model_name,
            "train_mse":   train_mse,
            "val_mse":     val_mse,
            "mse_gap":     mse_gap,
            "val/train":   ratio,
            "best_epoch":  best_epoch,
            "overfit_%":   overfit_pct,
        })

    if not rows:
        print("  No overfitting data available.")
        return

    # Sort by least overfitting (smallest gap)
    rows.sort(key=lambda r: r["mse_gap"])

    # ---- Print summary table ----
    print()
    print("=" * 80)
    print("OVERFITTING ANALYSIS")
    print("=" * 80)
    print(f"{'Model':<20}  {'Gap':>12}  {'Ratio':>8}  {'BestEp':>8}  {'Overfit %':>12}")
    print("-" * 80)
    for r in rows:
        overfit_str = f"{r['overfit_%']:>12.2f}" if r["overfit_%"] is not None else f"{'N/A':>12}"
        print(f"{r['model']:<20}  {r['mse_gap']:>12.4e}  "
              f"{r['val/train']:>8.3f}  {str(r['best_epoch']):>8}  {overfit_str}")
    print("=" * 80)

    # ---- Save CSV ----
    csv_path = comparison_dir / "overfitting_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved overfitting summary to {csv_path}")
    
    # ---- Save .txt ----

    txt_path = comparison_dir / "overfitting_summary.txt"

    with open(txt_path, "w") as f:

        f.write("=" * 110 + "\n")
        f.write("OVERFITTING ANALYSIS\n")
        f.write("=" * 110 + "\n\n")

        f.write(
            "Metrics:\n"
            "  MSE Gap             = Validation MSE - Training MSE\n"
            "  Validation/Train    = Validation MSE / Training MSE\n"
            "  Overfit %           = Increase in validation loss from best epoch to final epoch\n"
            "  Validation Degrad.  = Final validation loss - best validation loss\n"
            "  After Best          = Number of epochs trained after the best validation epoch\n\n"
        )

        # ---- Table ----

        f.write(
            f"{'Model':<20}  "
            f"{'Gap':>12}  "
            f"{'Ratio':>8}  "
            f"{'BestEp':>8}  "
            f"{'Overfit %':>12}  "
            f"{'Val Degrad.':>14}  "
            f"{'After Best':>12}\n"
        )

        f.write("-" * 110 + "\n")

        for r in rows:

            overfit_str = (
                f"{r['overfit_%']:.4f}"
                if r["overfit_%"] is not None
                else "N/A"
            )

            degradation_str = (
                f"{r['validation_degradation']:.6e}"
                if r["validation_degradation"] is not None
                else "N/A"
            )

            after_best_str = (
                str(r["epochs_after_best"])
                if r["epochs_after_best"] is not None
                else "N/A"
            )

            f.write(
                f"{r['model']:<20}  "
                f"{r['mse_gap']:>12.6e}  "
                f"{r['val/train']:>8.4f}  "
                f"{str(r['best_epoch']):>8}  "
                f"{overfit_str:>12}  "
                f"{degradation_str:>14}  "
                f"{after_best_str:>12}\n"
            )

        f.write("=" * 110 + "\n\n")

        # ---- Detailed model information ----

        f.write("DETAILED RESULTS\n")
        f.write("-" * 110 + "\n\n")

        for r in rows:
            f.write(f"Model: {r['model']}\n")
            f.write(f"  Training MSE:            {r['train_mse']:.6e}\n")
            f.write(f"  Validation MSE:          {r['val_mse']:.6e}\n")
            f.write(f"  MSE gap:                 {r['mse_gap']:.6e}\n")
            f.write(f"  Validation/train ratio:  {r['val/train']:.6f}\n")
            f.write(f"  Best epoch:              {r['best_epoch']}\n")

            if r["overfit_%"] is not None:
                f.write(f"  Overfit:                 {r['overfit_%']:.6f}%\n")
            else:
                f.write("  Overfit:                 N/A\n")

            if r["validation_degradation"] is not None:
                f.write(
                    f"  Validation degradation:  "
                    f"{r['validation_degradation']:.6e}\n"
                )
            else:
                f.write("  Validation degradation:  N/A\n")

            if r["epochs_after_best"] is not None:
                f.write(
                    f"  Epochs after best:       "
                    f"{r['epochs_after_best']}\n"
                )
            else:
                f.write("  Epochs after best:       N/A\n")

            f.write("\n")

    print(f"Saved overfitting summary to {txt_path}")
    
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

    # Pull the real wavelength range from dataset.json instead of assuming
    # every dataset spans 1.0-9.5 um. Falls back to the historical default
    # only if an older dataset.json (pre-metadata) is being used.
    meta = getattr(full_ds, "metadata", {})
    lam_min_um = meta.get("lam_min_um", 1.0)
    lam_max_um = meta.get("lam_max_um", 9.5)
    print(f"  Wavelength range from dataset.json: {lam_min_um}-{lam_max_um} um")

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
    # The BP sensor's known-good operational band (Yuan et al., 2021 
    DEVICE_SCORE_MIN_UM = 2.0
    DEVICE_SCORE_MAX_UM = 9.0

    x_low,  x_high  = lam_min_um * 1e-6, lam_max_um * 1e-6   # dataset's actual range

    # Score wherever the device's usable band and the dataset's coverage
    # overlap -- never score outside what the dataset can even provide.
    x_meas_low  = max(x_low,  DEVICE_SCORE_MIN_UM * 1e-6)
    x_meas_high = min(x_high, DEVICE_SCORE_MAX_UM * 1e-6)

    if x_meas_low >= x_meas_high:
        raise ValueError(
            f"Dataset range [{lam_min_um}, {lam_max_um}] um doesn't overlap "
            f"the device's scorable band [{DEVICE_SCORE_MIN_UM}, {DEVICE_SCORE_MAX_UM}] um."
        )

    rnge = x_high - x_low

    cont_vals = np.linspace(x_low, x_high, num_cont_bins)
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

        # --- Batch inference for predictions (use GPU if available, fast) ---

        with torch.no_grad():
            preds = model(x_all).cpu().numpy()

        # --- Single-sample CPU latency benchmark (deployment-relevant) ---
        timing = benchmark_inference(model, input_dim, n_warmup=50, n_reps=100, n_threads=1)
        print(f"  {model_name}: {timing['median_ms']:.3f} ms/sample "
            f"(IQR {timing['q25_ms']:.3f}–{timing['q75_ms']:.3f}, "
            f"min {timing['min_ms']:.3f}, max {timing['max_ms']:.3f}) "
            f"[{timing['n_reps']} reps, {timing['n_threads']} thread]")

        # Explicitly free up memory before loading next model
        del model
        torch.cuda.empty_cache()

        preds_smoothed = smooth_batch(wl_full, preds, cont_vals)

        model_results[model_name] = {
            "predictions":          preds,
            "predictions_smoothed": preds_smoothed,
            "epoch":                epoch,
            "timing":               timing,
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
            
            pwe_totals[model_name] += peak_wavelength_error(
                smoothed[model_name][low_idx:high_idx],
                y_gt_g[low_idx:high_idx],
                cont_vals[low_idx:high_idx]
            )
            
        diff_gt = y_gt_g[low_idx:high_idx] - np.mean(y_gt_g[low_idx:high_idx])
        sst_total += np.sum(diff_gt ** 2)

        # Per-sample plot: ground truth + all models on one axes
        if not args.no_plot:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(cont_vals * 1e6, y_gt_g, label="Ground Truth", color="black", linewidth=1.5)
            for model_name in loaded_models:
                ax.plot(cont_vals * 1e6, smoothed[model_name], label=model_name, linewidth=1.2)
            ax.axvspan(x_meas_low * 1e6, x_meas_high * 1e6, alpha=0.07, color="blue",
                       label=f"Scored range ({x_meas_low*1e6:.1f}–{x_meas_high*1e6:.1f} µm)")
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
    avg_mse  = {m: mse_totals[m] / n                            for m in loaded_models}
    r2       = {m: 1 - sse[m] / sst_total                       for m in loaded_models}
    avg_sam  = {m: sam_totals[m] / n                            for m in loaded_models}   
    avg_pwe  = {m: pwe_totals[m] / n                            for m in loaded_models}
    
    print("\n" + "=" * 110)
    print("COMPARISON SUMMARY")
    print("=" * 110)
    print(
        f"{'Model':<12}  "
        f"{'Avg MSE':>12}  "
        f"{'R²':>8}  "
        f"{'SAM (°)':>9}  "
        f"{'PWE (µm)':>10}  "
        f"{'Best Epoch':>10}  "
        f"{'Latency (ms/sample)':>14}"
    )
    print("-" * 110)
    for model_name in loaded_models:
        t = model_results[model_name]["timing"]
        print(
            f"{model_name:<12}  "
            f"{avg_mse[model_name]:>12.4e}  "
            f"{r2[model_name]:>8.4f}  "
            f"{avg_sam[model_name]:>9.3f}  "
            f"{avg_pwe[model_name]:>10.4f}  "
            f"{model_results[model_name]['epoch']:>10}  "
            f"{t['median_ms']:>12.3f} ±{t['std_ms']:.3f}"
            )
    print("=" * 110)

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
            "best_epoch":            model_results[model_name]["epoch"],
            "avg_mse":               avg_mse[model_name],
            "r2":                    r2[model_name],
            "avg_sam_deg":           avg_sam[model_name],
            "avg_pwe_um":            avg_pwe[model_name],
            "inference_median_ms":   model_results[model_name]["timing"]["median_ms"],
            "inference_std_ms":      model_results[model_name]["timing"]["std_ms"],
            "inference_min_ms":      model_results[model_name]["timing"]["min_ms"],
            "inference_max_ms":      model_results[model_name]["timing"]["max_ms"],
            "inference_n_reps":      model_results[model_name]["timing"]["n_reps"],
            "inference_n_threads":   model_results[model_name]["timing"]["n_threads"],
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
            "model", "avg_mse", "r2", "avg_sam_deg", "avg_pwe_um",
            "best_epoch", "latency_median_ms", "latency_std_ms",
            "latency_min_ms", "latency_max_ms"])
        writer.writeheader()
        for model_name in loaded_models:
            t = model_results[model_name]["timing"]
            writer.writerow({
                "model":            model_name,
                "avg_mse":          avg_mse[model_name],
                "r2":               r2[model_name],
                "avg_sam_deg":      avg_sam[model_name],
                "avg_pwe_um":       avg_pwe[model_name],
                "best_epoch":       model_results[model_name]["epoch"],
                "latency_median_ms": t["median_ms"],
                "latency_std_ms":   t["std_ms"],
                "latency_min_ms":   t["min_ms"],
                "latency_max_ms":   t["max_ms"],
            })
    print(f"Saved CSV to {csv_path}")

    # ------------------------------------------------------------------
    # Comparison bar chart
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    models_list = loaded_models
    mse_vals    = [avg_mse[m]               for m in models_list]
    r2_vals     = [r2[m]                    for m in models_list]
    sam_vals    = [avg_sam[m]               for m in models_list]
    pwe_vals    = [avg_pwe[m]               for m in models_list]
    inf_median  = [model_results[m]["timing"]["median_ms"] for m in models_list]
    inf_q25     = [model_results[m]["timing"]["q25_ms"]        for m in models_list]
    inf_q75     = [model_results[m]["timing"]["q75_ms"]        for m in models_list]    
    colors      = plt.cm.tab10.colors[:len(models_list)]

    axes[0].bar(models_list, mse_vals, color=colors)
    axes[0].set_title("Avg MSE")
    axes[0].set_ylabel("MSE")
    axes[0].set_xlabel("Model")

    axes[1].bar(models_list, r2_vals, color=colors)
    axes[1].set_title("R²")
    axes[1].set_ylabel("R²")
    axes[1].set_xlabel("Model")
    axes[1].set_ylim(min(0, min(r2_vals)) - 0.05, 1.05)

    axes[2].bar(models_list, sam_vals, color=colors)
    axes[2].set_title("Avg SAM in degrees")
    axes[2].set_ylabel("SAM (°)")
    axes[2].set_xlabel("Model")

    axes[3].bar(models_list, pwe_vals, color=colors)
    axes[3].set_title("Avg Peak Wavelength Error")
    axes[3].set_ylabel("PWE (µm)")
    axes[3].set_xlabel("Model")

    inf_err = [np.array(inf_median) - np.array(inf_q25), np.array(inf_q75) - np.array(inf_median)]
    axes[4].bar(models_list, inf_median, color=colors, yerr=inf_err)
    axes[4].set_title("Inference Latency — median (IQR)")
    axes[4].set_ylabel("Time (ms/sample, batch=1, 1 thread, CPU)")
    axes[4].set_xlabel("Model")

    axes[5].set_visible(False)

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
                f"{'SAM(deg)':>10}  {'PWE(um)':>9}  {'Best Epoch':>10}"
                f"{'Inference(ms/sample)':>22}\n")
        f.write("-" * 70 + "\n")
        for model_name in loaded_models:
            t = model_results[model_name]["timing"]
            f.write(f"{model_name:<12}  {avg_mse[model_name]:>12.4e}  "
                    f"{r2[model_name]:>8.4f}  "
                    f"{avg_sam[model_name]:>10.3f}  "
                    f"{avg_pwe[model_name]:>9.4f}  "
                    f"{model_results[model_name]['epoch']:>10}  "
                    f"{t['median_ms']:>10.3f} ±{t['std_ms']:.3f}\n")
        best = min(loaded_models, key=lambda m: avg_mse[m])
        f.write(f"\nBest model by MSE: {best}\n")

    print(f"Saved summary to {summary_path}")

    # ------------------------------------------------------------------
    # Overfitting analysis
    # ------------------------------------------------------------------
    run_overfitting_analysis(experiment_dir, loaded_models, comparison_dir)

if __name__ == "__main__":
    main()