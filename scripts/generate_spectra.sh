python3 -m src.cli.generate_spectra \
	--seed 42 \
	--n-samples 20000 \
	--s-dim 823 \
	--method NIST \
	--responsivity data/responsivity_data/processed/cropped_2p5_9p5/responsivity_823.npy
	"$@"
