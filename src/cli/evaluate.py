# evaluate.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.datasets.registry import get_dataset
from src.models.model import get_model
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def r2_batch(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred_flat = y_pred.view(-1)
    y_true_flat = y_true.view(-1)
    ss_res = torch.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = torch.sum((y_true_flat - torch.mean(y_true_flat)) ** 2)
    return (1 - ss_res / ss_tot).item()

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

def evaluate_split(model, loader, criterion, device, wl_grid: np.ndarray):
    """
    Returns avg MSE, R², SAM (degrees), and peak wavelength error (µm)
    over all samples in the loader.
    """
    model.eval()
    total_mse = 0.0
    total_r2 = 0.0
    total_sam = 0.0
    total_pwe = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            pred = model(x)
            mse = criterion(pred, y).item()
            total_mse += mse * x.size(0)
            total_r2  += r2_batch(pred, y) * x.size(0)
            pred_np = pred.cpu().numpy()
            y_np    = y.cpu().numpy()
            for j in range(len(pred_np)):
                total_sam += spectral_angle(pred_np[j], y_np[j])
                total_pwe += peak_wavelength_error(pred_np[j], y_np[j], wl_grid)
            n += x.size(0)

    return (total_mse / n,
            total_r2  / n,
            total_sam / n,
            total_pwe / n)

def plot_loss_curve(eval_dir: Path):
    """
    Reads loss_history.json (written by train.py) and plots train vs. val
    loss over epochs for THIS model only. Saves as loss_curve.png in the
    same evaluation folder as metrics.json.
    """
    history_path = eval_dir / "loss_history.json"
    if not history_path.exists():
        print(f"  No loss_history.json found at {history_path} — skipping plot. "
              f"(Was this checkpoint trained before the history-saving update?)")
        return

    with open(history_path) as f:
        history = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train Loss")
    ax.plot(history["epoch"], history["val_loss"],   label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_yscale("log")
    ax.set_title("Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = eval_dir / "loss_curve.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved loss curve to {out_path}")

def main():
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on train/val splits")
    p.add_argument("--dataset", "-d", required=True, help="Registered dataset name")
    p.add_argument("--model", "-m", required=True, help="Registered model name")
    p.add_argument("--checkpoint","-c", required=True, help="Path to .pth or checkpoint file")
    p.add_argument("--root", type=str, default="data/spectra_data/", help="Data root")
    p.add_argument("--seed", type=int, default=42, help="Split RNG seed")
    p.add_argument("--val-size", type=int, default=200, help="Number of val samples")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size for eval")
    p.add_argument("--num-workers", type=int,default=0, help="DataLoader workers")
    p.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = p.parse_args()

    # PRINT MODEL 

    print(f"Training model {args.model}")

    # DEVICE SETUP
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # DATASET AND DATALOADER
    DS = get_dataset(args.dataset)
    full_ds = DS(
        root=args.root,
        seed=args.seed
    )

    print("\nDataset information:")

    if hasattr(full_ds, "metadata"):
        for key, value in full_ds.metadata.items():
            print(f"  {key}: {value}")

    print(f"  Number of samples: {len(full_ds)}")

    total = len(full_ds)
    train_n = total - args.val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds,
        [train_n, args.val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=len(train_ds),
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=len(val_ds),
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=(device.type=="cuda"))

    # MODEL
    x0, y0 = full_ds[0]
    print(f"Input dimension : {len(x0)}")
    print(f"Output dimension: {len(y0)}")
    # Build wavelength grid in metres — used for peak wavelength error
    wl_grid = np.linspace(1e-6, 9.5e-6, len(y0))

    ModelClass = get_model(args.model)
    model = ModelClass(input_dim=len(x0), output_dim=len(y0)).to(device)

    # LOAD CHECKPOINT
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Evaluating epoch {ckpt.get('epoch')}")
    else:
        model.load_state_dict(ckpt)

    # LOSS
    criterion = nn.MSELoss()

    # EVALUATE
    train_mse, train_r2, train_sam, train_pwe = evaluate_split(
        model, train_loader, criterion, device, wl_grid)
    val_mse,   val_r2,   val_sam,   val_pwe   = evaluate_split(
        model, val_loader,   criterion, device, wl_grid)
    
    print("\nEvaluation Results")
    print("-"*40)
    print(f"Training")
    print(f"  MSE : {train_mse:.6f}")
    print(f"  R²  : {train_r2:.4f}")
    print(f"  SAM : {train_sam:.4f}°")
    print(f"  PWE : {train_pwe:.3f}µm")

    print()

    print(f"Validation")
    print(f"  MSE : {val_mse:.6f}")
    print(f"  R²  : {val_r2:.4f}")
    print(f"  SAM : {val_sam:.4f}°")
    print(f"  PWE : {val_pwe:.3f}µm")

    results = {
        "dataset": args.dataset,
        "model": args.model,
        "checkpoint": str(Path(args.checkpoint).name),
        "epoch": ckpt.get("epoch"),

        "train": {
            "mse": train_mse,
            "r2": train_r2,
            "SAM": train_sam,
            "Peak wavelentgh error (um)": train_pwe,
        },

        "validation": {
            "mse": val_mse,
            "r2": val_r2,
            "SAM": val_sam,
            "Peak wavelentgh error (um)": val_pwe,
        }
    }

    results_file = (
        Path(args.checkpoint)
        .parents[1]
        / "evaluation"
        / "metrics.json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved evaluation to {results_file}")
    
    # Generate the loss-over-epochs plot for this model
    plot_loss_curve(results_file.parent)


if __name__ == "__main__":
    main()