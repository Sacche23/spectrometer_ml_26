python3 -m src.cli.estimate_noise \
    --resp data/responsivity_data/processed/cropped_2p5_9p5/responsivity_823.npy \
    --currents data/spectra_data/processed/rand_sop/I_s42_1000x823.npy \
    --spectra data/spectra_data/processed/rand_sop/S_s42_1000x823.npy \
    --subset 1000 \
    "$@"
