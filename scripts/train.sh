#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=diego.sacchettoni@yale.edu

python3 -m src.cli.train \
    --experiment-dir experiments/debug_unet \
	--dataset rand_sop \
	--model unet \
	--seed 42 \
	--batch-size 32 \
	--num-epochs 100 \
	--learning-rate 1e-3 \
	--learning-rate-decay 0.6 \
	--learning-rate-period 100 \
	--gaussian-noise True \
	--gaussian-noise-std 1e-4 \
	--validation-size 200 \
	--num-workers 0 \
	--device cpu \
	"$@"
