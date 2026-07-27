#!/bin/bash
#SBATCH --job-name=eval_model
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=diego.sacchettoni@yale.edu


python3 -m src.cli.evaluate \
    --dataset rand_sop \
    --model cui_mlp_v2 \
    --checkpoint experiments/debug_mlp_2/cui_mlp_v2/checkpoints/best.pth \
    --root data/spectra_data/ \
    --seed 42 \
    --val-size 200 \
    --batch-size 32 \
    "$@"
    