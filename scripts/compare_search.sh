#!/bin/bash
#SBATCH --job-name=compare_search
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL

python3 -m src.cli.compare_search \
    --mlp-dir experiments/mlp_search \
    --unet-dir experiments/unet_search \
    --output-dir experiments/search_comparison
