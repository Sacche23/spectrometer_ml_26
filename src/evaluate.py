#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from datasets.registry import get_dataset
from models.model import get_model

def r2_batch(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred_flat = y_pred.view(-1)
    y_true_flat = y_true.view(-1)
    ss_res = torch.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = torch.sum((y_true_flat - torch.mean(y_true_flat)) ** 2)
    return (1 - ss_res / ss_tot).item()

def evaluate_split(model, loader, criterion, device):
    model.eval()
    total_mse = 0.0
    total_r2 = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            pred = model(x)
            mse = criterion(pred, y).item()
            total_mse += mse * x.size(0)
            total_r2  += r2_batch(pred, y) * x.size(0)
            n += x.size(0)
    return total_mse / n, total_r2 / n

def main():
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on train/val splits")
    p.add_argument("--dataset", "-d", required=True, help="Registered dataset name")
    p.add_argument("--model", "-m", required=True, help="Registered model name")
    p.add_argument("--checkpoint","-c", required=True, help="Path to .pth or checkpoint file")
    p.add_argument("--root", type=str, default="data/spectra_data/", help="Data root")
    p.add_argument("--seed", type=int, default=42, help="Split RNG seed")
    p.add_argument("--val-size", type=int, default=200, help="Number of val samples")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for eval")
    p.add_argument("--num-workers", type=int,default=0, help="DataLoader workers")
    p.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = p.parse_args()

    # DEVICE SETUP
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # DATASET AND DATALOADER
    DS = get_dataset(args.dataset)
    full_ds = DS(root=args.root)
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
    ModelClass = get_model(args.model)
    model = ModelClass(input_dim=len(x0), output_dim=len(y0)).to(device)

    # LOAD CHECKPOINT
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    # LOSS
    criterion = nn.MSELoss()

    # EVALUATE
    train_mse, train_r2 = evaluate_split(model, train_loader, criterion, device)
    val_mse,   val_r2   = evaluate_split(model, val_loader,   criterion, device)

    print(f"Train → MSE: {train_mse:.6f},  R²: {train_r2:.4f}")
    print(f"Val   → MSE: {val_mse:.6f},  R²: {val_r2:.4f}")

if __name__ == "__main__":
    main()