#!/bin/bash
#SBATCH --job-name=final_3_rand_sop_nist
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=diego.sacchettoni@yale.edu

module reset

source ~/venvs/spectro/bin/activate

python3 -u -m src.cli.run_experiment \
    --models cui_mlp_v2 cnn2 unet_8_0.2 \
    --dataset rand_sop_nist \
    --seed 64 \
    --batch-size 64 128 64 \
    --num-epochs 1600 \
    --learning-rate 2e-4 1e-3 1.25e-3 \
    --learning-rate-decay 0.3 0.6 0.1 \
    --learning-rate-period 100 200 200 \
    --gaussian-noise True \
    --gaussian-noise-std 1e-4 \
    --validation-size 500 \
    --num-workers 4 \
    --device cuda \
    --experiment-dir experiments/mlp_cnn_unet_rand_sop_nist_FINALS \
    "@"
