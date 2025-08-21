python3 -m src.cli.alpha_cross_val \
    --resp data/responsivity_data/processed/cropped_2p5_9p5/responsivity_823.npy \
    --currents data/spectra_data/processed/NIST/I.npy \
    --spectra data/spectra_data/processed/NIST/S.npy \
    --alphas "1e-2,1e-1,1" \
    --folds 5 \
    --downsample 4 \
    --subset 20 \
    --metric spectra \
    --noise-std 0 \
    --normalize \
    --plot \
    "$@"