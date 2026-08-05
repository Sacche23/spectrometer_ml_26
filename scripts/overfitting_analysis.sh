#!/bin/bash
#SBATCH --job-name=overfit
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00

python3 -m src.cli.overfitting_analysis \
    --experiment-dir experiments/unet_grid_analysis