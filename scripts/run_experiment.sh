#!/bin/bash
#SBATCH --job-name=mlp_search
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

set -e

MODEL="cui_mlp_v2"
DATASET="rand_sop_nist"
SEED=23

NUM_EPOCHS=1

LEARNING_RATE_DECAY=0.1

GAUSSIAN_NOISE=True
GAUSSIAN_NOISE_STD=1e-4

VALIDATION_SIZE=500
NUM_WORKERS=4
DEVICE="cuda"

EXPERIMENT_DIR="experiments/mlp_search"

mkdir -p "$EXPERIMENT_DIR"
mkdir -p logs

# Learning rates to test
LEARNING_RATES=(3e-4 1e-3 3e-3)

# Batch sizes to test
BATCH_SIZES=(64 128 256)

# Learning-rate decay periods to test
LR_PERIODS=(60 100)


for LR in "${LEARNING_RATES[@]}"; do
    for BATCH in "${BATCH_SIZES[@]}"; do
        for PERIOD in "${LR_PERIODS[@]}"; do
            RUN_NAME="lr_${LR}_bs_${BATCH}_period_${PERIOD}"
            python3 -m src.cli.run_experiment \
                --models "$MODEL" \
                --dataset "$DATASET" \
                --seed "$SEED" \
                --batch-size "$BATCH" \
                --num-epochs "$NUM_EPOCHS" \
                --learning-rate "$LR" \
                --learning-rate-decay "$LEARNING_RATE_DECAY" \
                --learning-rate-period "$PERIOD" \
                --gaussian-noise "$GAUSSIAN_NOISE" \
                --gaussian-noise-std "$GAUSSIAN_NOISE_STD" \
                --validation-size "$VALIDATION_SIZE" \
                --num-workers "$NUM_WORKERS" \
                --device "$DEVICE" \
                --experiment-dir "$EXPERIMENT_DIR/$RUN_NAME"
        done
    done
done