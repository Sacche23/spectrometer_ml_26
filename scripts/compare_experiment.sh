python3 src/compare_experiment.py \
	--dataset rand_sop \
	--model cnn2 \
	--checkpoint experiments/run_20250722_115422_rand_sop/checkpoints/epoch050.pth \
	--normalize True \
	--downsample-factors 25 30 50 \
	--no-plot \
	--alpha-tikh 1e-3 5e-4 2e-4
