python3 -m src.cli.generate_spectra \
	--seed 42 \
	--n-samples 1 \
	--s-dim 1000 \
	--method NIST \
	--responsivity data/responsivity_data/processed/responsivity.npy
	"$@"