python3 -m src.cli.estimate_noise \
    --resp data/responsivity_data/processed/responsivity.npy \
    --currents data/spectra_data/processed/rand_sop/I.npy \
    --spectra data/spectra_data/processed/rand_sop/S.npy \
    --subset 1000 \
    "$@"
