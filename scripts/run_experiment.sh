#!/bin/bash
#SBATCH --job-name=run_experiment
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=diego.sacchettoni@yale.edu

module load miniconda
conda activate spectro

python3 -m src.cli.run_experiment \
    --models cui_mlp_v2_256_256 cui_mlp_v2_256_512 cui_mlp_v2_256_1024 \
             cui_mlp_v2_512_256 cui_mlp_v2_512_512 cui_mlp_v2_512_1024 \
             cui_mlp_v2_1024_256 cui_mlp_v2_1024_512 cui_mlp_v2_1024_1024 \
    --dataset rand_sop \
    --seed 42 \
    --batch-size 32 \
    --num-epochs 75 \
    --learning-rate 1e-3 \
    --learning-rate-decay 0.1    \
    --learning-rate-period 60 \
    --gaussian-noise True \
    --gaussian-noise-std 1e-4 \
    --validation-size 200 \
    --num-workers 0 \
    --device cpu \
    --responsivity data/responsivity_data/processed/responsivity.npy \
    "$@"
