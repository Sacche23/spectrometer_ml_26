#!/bin/bash
#SBATCH --job-name=my_ml_train
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # Number of GPUs
#SBATCH --cpus-per-task=8       # Number of CPU cores
#SBATCH --mem=32G               # RAM
#SBATCH --time=02:00:00         # HH:MM:SS
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=diego.sacchettoni@yale.edu

python3 -m src.cli.train \
    --experiment-dir experiments/debug_mlp_3 \
	--dataset rand_sop \
	--model wen_mlp \
	--seed 42 \
	--batch-size 32 \
	--num-epochs 20 \
	--learning-rate 1e-3 \
	--learning-rate-decay 0.1 \
	--learning-rate-period 60 \
	--gaussian-noise True \
	--gaussian-noise-std 1e-4 \
	--validation-size 200 \
	--num-workers 0 \
	--device cpu \
	"$@"
