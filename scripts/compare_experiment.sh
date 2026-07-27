#!/bin/bash
#SBATCH --job-name=compare_model
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=diego.sacchettoni@yale.edu

python3 -m src.cli.compare_experiment \
    --experiment-dir experiments/20260710_1519_rand_sop_seed42 \
    --models dnn cnn2 \
    --dataset rand_sop \
    --responsivity data/responsivity_data/processed/responsivity.npy \
    --seed 42 \
    --val-size 200 \
    --device cpu \
    "$@"