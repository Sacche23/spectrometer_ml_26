#!/bin/bash
#SBATCH --job-name=run_experiment_unet
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=diego.sacchettoni@yale.edu

module reset
module load miniconda

conda activate cenv

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TORCH_NUM_THREADS=8   

python3 -m src.cli.run_experiment \
    --models unet \
    --dataset rand_sop \
    --seed 42 \
    --batch-size 64 \
    --num-epochs 100 \
    --learning-rate 1e-3 \
    --learning-rate-decay 0.6 \
    --learning-rate-period 50 \
    --gaussian-noise True \
    --gaussian-noise-std 1e-4 \
    --validation-size 200 \
    --num-workers 0 \
    --device cpu \
    --responsivity data/responsivity_data/processed/responsivity.npy \
    "$@"
