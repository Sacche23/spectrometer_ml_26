import os
import argparse
import torch
import numpy as np


def train(
        spectra_path: str,
        matrix_path: str,
        experiment_path: str, #path to run, if empty makes new timestamped one (or could also opt do do just if current name exists do this)

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
    
    # TRANSFORMATIONS
    # add stuff here...

    
    # DEVICE SETUP
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DATASET

    # DATALOADER

    # MODEL

    