#!/bin/bash
#SBATCH --job-name=unet_search_2
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

MODEL="unet_8_0.2"
DATASET="rand_sop_nist"
SEED=23

NUM_EPOCHS=800

LEARNING_RATE_DECAY=0.1
LEARNING_RATE_PERIOD=200

GAUSSIAN_NOISE=True
GAUSSIAN_NOISE_STD=1e-4

VALIDATION_SIZE=500
NUM_WORKERS=4
DEVICE="cuda"

EXPERIMENT_DIR="experiments/unet_search_2"

mkdir -p "$EXPERIMENT_DIR"
mkdir -p logs

LEARNING_RATES=(5e-4 7.5e-4 1e-3 1.25e-3 1.5e-3)
BATCH_SIZES=(16 32 64)

for LR in "${LEARNING_RATES[@]}"; do
    for BATCH in "${BATCH_SIZES[@]}"; do
        RUN_NAME="lr_${LR}_bs_${BATCH}_period_${LEARNING_RATE_PERIOD}"

        python3 -m src.cli.run_experiment \
            --models "$MODEL" \
            --dataset "$DATASET" \
            --seed "$SEED" \
            --batch-size "$BATCH" \
            --num-epochs "$NUM_EPOCHS" \
            --learning-rate "$LR" \
            --learning-rate-decay "$LEARNING_RATE_DECAY" \
            --learning-rate-period "$LEARNING_RATE_PERIOD" \
            --gaussian-noise "$GAUSSIAN_NOISE" \
            --gaussian-noise-std "$GAUSSIAN_NOISE_STD" \
            --validation-size "$VALIDATION_SIZE" \
            --num-workers "$NUM_WORKERS" \
            --device "$DEVICE" \
            --experiment-dir "$EXPERIMENT_DIR/$RUN_NAME"
    done
done