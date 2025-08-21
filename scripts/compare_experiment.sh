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
	--dataset rand_sop_823 \
	--model cnn2 \
	--checkpoint training_runs/run_20250820_145540_rand_sop_823/checkpoints/epoch720.pth \
	--normalize True \
	--downsample-factors 4 \
	--device cpu \
	--alpha-tikh 1e-2 \
	--alpha-lasso 1e-1 \
	--responsivity data/responsivity_data/processed/cropped_2p5_9p5/responsivity_823.npy \
	"$@"
