#!/bin/bash
#SBATCH --job-name=unet_final_3
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

python3 -m src.cli.run_experiment \
    --models \
        unet_8_0.2 \
        unet_8_0.5 \
        unet_16_0.5 \
    --dataset rand_sop \
    --seed 42 \
    --batch-size 64 \
    --num-epochs 1500 \
    --learning-rate 5e-4 \
    --learning-rate-decay 0.8 \
    --learning-rate-period 200 \
    --gaussian-noise True \
    --gaussian-noise-std 1e-4 \
    --validation-size 500 \
    --num-workers 4 \
    --device cuda \
    --experiment-dir experiments/unet_final_3 \
    --responsivity data/responsivity_data/processed/responsivity.npy \
    "$@"
