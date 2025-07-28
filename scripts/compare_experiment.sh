python3 src/compare_experiment.py \
	--dataset rand_sop \
	--model cnn2 \
	--checkpoint experiments/run_20250725_112425_rand_sop/checkpoints/epoch1000.pth \
	--normalize True \
	--downsample-factors 4 10 12 16 20 22 23 24 25 30 50
