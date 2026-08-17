# scripts/generate_spectra_nist.sh
python3 -m src.cli.generate_spectra \
    --seed 42 \
    --n-samples 50000 \
    --method NIST \
    "$@"