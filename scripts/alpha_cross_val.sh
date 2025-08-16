python3 -m src.cli.alpha_cross_val \
    --resp data/responsivity_data/processed/responsivity.npy \
    --currents data/spectra_data/processed/rand_sop/I_s42.npy \
    --spectra data/spectra_data/processed/rand_sop/S_s42.npy \
    --alphas "1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1" \
    --folds 5 \
    --downsample 100 \
    --subset 20 \
    --metric spectra \
    --noise-std 1e-3 \
    --normalize \
    --plot \
    "$@"