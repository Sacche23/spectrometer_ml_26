#!/bin/bash
#SBATCH --job-name=compare_exp
#SBATCH --partition=gpu        # change to your CPU queue if no GPU needed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err


python3 src/compare_experiment.py \
	--dataset rand_sop \
	--model cnn2 \
	--checkpoint experiments/run_20250812_160809_rand_sop/checkpoints/epoch1000.pth \
	--normalize True \
	--downsample-factors 1 2 4 10 20 25 40 \
	--device cpu \
	--alpha-tikh 1e-5 1e-5 1e-5 1e-4 1e-4 1e-4 1e-2 \
	--alpha-lasso 1e-2 1e-2 1e-1 1e-1 1e-1 1e-1 1e-1
