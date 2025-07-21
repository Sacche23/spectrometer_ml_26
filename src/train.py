import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets.registry import get_dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from models.model import *


def train(
        dataset_name: str,
        seed: int,

        batch_size: int,
        num_epochs: int,

        learning_rate: float,
        learning_rate_decay: float,
        learning_rate_period: int,

        gaussian_noise: bool,
        gaussian_noise_std: float,

        validation_size: int,

        model_type: str,
        
        num_workers: int=0,
        device: torch.device=None):
    
    # Transformations

    # add stuff here...
    
    # Device setup

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and Dataloader

    DSClass = get_dataset(dataset_name)
    ds = DSClass(root="./data/spectra_data/")

    total = len(ds)
    val_size = validation_size
    train_size = total-val_size
    train_ds, val_ds = torch.utils.data.random_split(
        ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed) # for reproducibility
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type=="cuda"))

    # Model

    x0, y0 = ds[0]
    input_size, output_size = len(x0), len(y0)
    model = CNN2(input_dim=input_size, output_dim=output_size).to(device)
    print(f"Model is on device: {next(model.parameters()).device}")

    # Loss and Optimizer

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=learning_rate_period, gamma=learning_rate_decay) # multiply lr * gamma every N steps

    # Training Loop

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (currents, spectra) in enumerate(train_loader, start=1):
            currents = currents.to(device, non_blocking=True)
            currents = currents.to(device).float() #quick fix, must change!!!!!!!!!
            spectra = spectra.to(device, non_blocking=True)

            if gaussian_noise:
                currents = currents + torch.randn_like(currents) * gaussian_noise_std

            optimizer.zero_grad()
            outputs = model(currents)
            loss = criterion(outputs, spectra)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * currents.size(0)
        
        scheduler.step()

        train_loss = running_loss/train_size
        
        if epoch % 10 == 0:
            print(f"--> Epoch {epoch:4d} complete.  train_loss={train_loss:.6f}")


if __name__=="__main__":
    train(dataset_name="rand_sop", seed=42, batch_size=128, num_epochs=50, learning_rate=1e-3, learning_rate_decay=6e-1, learning_rate_period=20, gaussian_noise=True, gaussian_noise_std=1e-4, validation_size=200, model_type=None)