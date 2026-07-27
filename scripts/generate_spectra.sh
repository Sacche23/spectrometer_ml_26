python3 -m src.cli.generate_spectra \
	--seed 42 \
	--n-samples 20000 \
	--s-dim 1000 \
	--method rand_sop \
	--responsivity data/responsivity_data/processed/responsivity.npy
	"$@"