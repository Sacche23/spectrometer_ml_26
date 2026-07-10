#!/bin/bash
#SBATCH --job-name=run_experiment
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=diego.sacchettoni@yale.edu

python3 -m src.cli.run_experiment \
    --models dnn cnn2 \
    --dataset rand_sop \
    --seed 42 \
    --batch-size 128 \
    --num-epochs 20 \
    --learning-rate 1e-3 \
    --learning-rate-decay 0.6 \
    --learning-rate-period 200 \
    --gaussian-noise True \
    --gaussian-noise-std 1e-4 \
    --validation-size 200 \
    --num-workers 0 \
    --device cpu \
    --responsivity data/responsivity_data/processed/responsivity.npy \
    "$@"
