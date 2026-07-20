#!/bin/bash
#SBATCH --job-name=eval_cnn2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err


python3 -m src.cli.evaluate \
    --dataset rand_sop \
    --model cui_mlp_v2 \
    --checkpoint experiments/debug_mlp_2/cui_mlp_v2/checkpoints/best.pth \
    --root data/spectra_data/ \
    --seed 42 \
    --val-size 200 \
    --batch-size 32 \
    "$@"
    