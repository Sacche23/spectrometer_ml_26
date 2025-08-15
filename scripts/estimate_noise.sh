python3 -m src.cli.estimate_noise \
    --resp data/responsivity_data/processed/responsivity.npy \
    --currents data/spectra_data/processed/rand_sop/I_s42.npy \
    --spectra data/spectra_data/processed/rand_sop/S_s42.npy \
    --subset 1000 \
    "$@"
