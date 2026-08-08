# scripts/generate_spectra_nist.sh
python3 -m src.cli.generate_spectra \
    --seed 42 \
    --n-samples 20000 \
    --method NIST \
    "$@"