#!/bin/bash
#SBATCH --job-name=compare_exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

python3 -m src.cli.compare_experiment \
    --experiment-dir experiments/20260710_1519_rand_sop_seed42 \
    --models dnn cnn2 \
    --dataset rand_sop \
    --responsivity data/responsivity_data/processed/responsivity.npy \
    --seed 42 \
    --val-size 200 \
    --device cpu \
    "$@"