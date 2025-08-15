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
#SBATCH --mail-user=leo.wylonis@yale.edu

python3 -m src.cli.train \
	--dataset rand_sop \
	--model cnn2 \
	--seed 42 \
	--batch-size 128 \
	--num-epochs 1000 \
	--learning-rate 1e-3 \
	--learning-rate-decay 0.6 \
	--learning-rate-period 200 \
	--gaussian-noise True \
	--gaussian-noise-std 1e-4 \
	--validation-size 200 \
	--num-workers 0 \
	--device cpu \
	"$@"
