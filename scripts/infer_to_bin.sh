python3 -m src.cli.infer_to_bin \
    --data-dir data/spectra_data/processed/rand_sop \
    --model-path experiments/run_20250722_115422_rand_sop/checkpoints/epoch050.pth \
    --index 0 \
    --output-dir outputs/binary_in_out \
    "$@"
