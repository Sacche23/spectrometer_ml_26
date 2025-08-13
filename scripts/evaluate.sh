#!/bin/bash
#SBATCH --job-name=eval_cnn2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err


python3 src/evaluate.py \
    --dataset rand_sop \
    --model cnn2 \
    --checkpoint experiments/run_20250812_160809_rand_sop/checkpoints/epoch1000.pth \
    --batch-size 128
