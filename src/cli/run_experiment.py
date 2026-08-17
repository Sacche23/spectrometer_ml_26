"""
run_experiment.py
-----------------
Mother script that runs a full experiment end-to-end:

  1. Creates one shared experiment directory
  2. For each model in --models:
       a. Trains the model (with per-model hyperparameters if provided)
       b. Evaluates the best checkpoint
  3. Prompts the user for confirmation (auto-skipped in batch jobs)
  4. Runs comparison across all trained models
     (works for 1 or more models)

Usage:
  python3 -m src.cli.run_experiment \
      --models dnn cnn2 \
      --dataset rand_sop \
      --seed 42 \
      ...

All training hyperparameters are passed through to train.py unchanged.
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ====================================================================================
# Helpers
# ====================================================================================
def resolve_per_model(values: list, n_models: int, param_name: str) -> list:
    """
    Broadcast or validate a list of hyperparameter values against the model count.
 
    Rules:
      - 1 value  → replicated for every model
      - N values == n_models → used as-is, one per model in order
      - Anything else → clear error message
 
    Examples:
      resolve_per_model([1e-3], 3, "learning-rate")     → [1e-3, 1e-3, 1e-3]
      resolve_per_model([1e-3, 5e-4, 1e-4], 3, ...)     → [1e-3, 5e-4, 1e-4]
      resolve_per_model([1e-3, 5e-4], 3, ...)            → ValueError
    """
    if len(values) == 1:
        return values * n_models
    elif len(values) == n_models:
        return values
    else:
        raise ValueError(
            f"\n  ERROR: --{param_name} has {len(values)} value(s) "
            f"but there are {n_models} models.\n"
            f"  Provide either 1 value (applied to all models) or exactly "
            f"{n_models} values (one per model in order).\n"
            f"  Got: {values}"
        )
    
def run(cmd: list[str], label: str) -> int:
    """
    Run a subprocess command, streaming its output live to the terminal.
    Returns the exit code. Prints a clear header/footer so you can see
    where each step begins and ends in the scrollback.

    cmd   : list of strings, e.g. ["python3", "-m", "src.cli.train", "--model", "dnn"]
    label : human-readable description shown in the header, e.g. "TRAINING dnn"
    """
    print()
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)

    # subprocess.run with no capture_output means stdout/stderr flow
    # directly to your terminal in real time — you see training logs
    # as they happen, not all at once at the end.
    result = subprocess.run(cmd)
    return result.returncode


def confirm(prompt: str) -> bool:
    """
    Ask the user a yes/no question. Keeps asking until they type y or n.
    Returns True for yes, False for no.
    """
    while True:
        answer = input(f"\n{prompt} [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please type y or n.")


# ====================================================================================
# Main
# ====================================================================================

def main():

    p = argparse.ArgumentParser(
        description="End-to-end experiment runner: train → evaluate → compare.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- What to run ----
    p.add_argument("--models", nargs="+", required=True,
                   help="One or more model names. Example: --models dnn cnn2 unet_8_0.2")
    p.add_argument("--dataset", required=True,
                   help="Registered dataset name, e.g. rand_sop")
 
    # ---- Experiment directory ----
    p.add_argument("--experiment-dir", type=str, default=None,
                   help="Path for the experiment directory. Auto-generated if omitted.")
 
    # ---- Per-model training hyperparameters ----
    # Each accepts either 1 value (broadcast) or N values (one per model).
    p.add_argument("--seed", type=int, default=42,
                   help="Global RNG seed (single value only).")
    p.add_argument("--batch-size", type=int, nargs="+", default=[128],
                   metavar="BS",
                   help="Batch size(s). 1 value or one per model.")
    p.add_argument("--num-epochs", type=int, nargs="+", default=[1000],
                   metavar="N",
                   help="Number of epochs. 1 value or one per model.")
    p.add_argument("--learning-rate", type=float, nargs="+", default=[1e-3],
                   metavar="LR",
                   help="Learning rate(s). 1 value or one per model.")
    p.add_argument("--learning-rate-decay", type=float, nargs="+", default=[0.8],
                   metavar="D",
                   help="LR decay multiplier(s). 1 value or one per model.")
    p.add_argument("--learning-rate-period", type=int, nargs="+", default=[200],
                   metavar="P",
                   help="Epochs between LR decay steps. 1 value or one per model.")
    p.add_argument("--gaussian-noise", type=bool, default=True,
                   help="Add Gaussian noise to photocurrents during training.")
    p.add_argument("--gaussian-noise-std", type=float, nargs="+", default=[1e-4],
                   metavar="STD",
                   help="Noise std dev(s). 1 value or one per model.")
    p.add_argument("--validation-size", type=int, default=500,
                   help="Number of validation samples (shared across all models).")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader worker count.")
    p.add_argument("--device", type=str, default=None,
                   help="Device: cpu or cuda.")
 
    # ---- Weight decay (regularization) ----
    p.add_argument("--weight-decay", type=float, nargs="+", default=[0.0],
                   metavar="WD",
                   help="Weight decay strength. 0 disables it. "
                        "1 value or one per model.")
    p.add_argument("--weight-decay-type", type=str, nargs="+", default=["l2"],
                   choices=["l1", "l2"],
                   metavar="TYPE",
                   help="l2: built into Adam (standard). "
                        "l1: manual penalty added to loss (encourages sparsity). "
                        "1 value or one per model.")


    # ---- Comparison settings ----
    p.add_argument("--no-plot", action="store_true",
                   help="Skip per-sample plots in the comparison step (faster).")
    p.add_argument("--auto-compare", action="store_true",
                   help="Skip the interactive y/n prompt and always run comparison. "
                        "Also triggers automatically when stdin is not a terminal "
                        "(e.g. running under sbatch on the HPC).")

    args = p.parse_args()
    n_models = len(args.models)


    # ------------------------------------------------------------------
    # Step 0 — Validate and resolve per-model hyperparameters
    # ------------------------------------------------------------------
    # All validation happens here before any training starts, so you get
    # a clear error immediately rather than halfway through a long run.
 
    try:
        batch_sizes     = resolve_per_model(args.batch_size,            n_models, "batch-size")
        num_epochs_list = resolve_per_model(args.num_epochs,            n_models, "num-epochs")
        lrs             = resolve_per_model(args.learning_rate,         n_models, "learning-rate")
        lr_decays       = resolve_per_model(args.learning_rate_decay,   n_models, "learning-rate-decay")
        lr_periods      = resolve_per_model(args.learning_rate_period,  n_models, "learning-rate-period")
        noise_stds      = resolve_per_model(args.gaussian_noise_std,    n_models, "gaussian-noise-std")
        weight_decays   = resolve_per_model(args.weight_decay,          n_models, "weight-decay")
        wd_types        = resolve_per_model(args.weight_decay_type,     n_models, "weight-decay-type")
    except ValueError as e:
        print(e)
        sys.exit(1)
 
    # ------------------------------------------------------------------
    # Step 1 — Resolve experiment directory
    # ------------------------------------------------------------------
 
    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_dir = Path("experiments") / f"{ts}_{args.dataset}_seed{args.seed}"
 
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExperiment directory: {experiment_dir}")
    print(f"Models to train: {args.models}")
 
    # Print per-model config summary so it's visible in SLURM logs
    print("\nPer-model hyperparameter assignments:")
    print(f"  {'Model':<20} {'LR':>8} {'BS':>6} {'Epochs':>7} "
          f"{'Decay':>7} {'Period':>7} {'WD':>8} {'WD-type':>8}")
    print(f"  {'-'*75}")
    for i, model_name in enumerate(args.models):
        print(f"  {model_name:<20} {lrs[i]:>8.1e} {batch_sizes[i]:>6} "
              f"{num_epochs_list[i]:>7} {lr_decays[i]:>7.2f} {lr_periods[i]:>7} "
              f"{weight_decays[i]:>8.1e} {wd_types[i]:>8}")
 
    # ------------------------------------------------------------------
    # Step 2 — Train + Evaluate each model in sequence
    # ------------------------------------------------------------------
 
    trained_models   = []
    evaluated_models = []
 
    for i, model_name in enumerate(args.models):
 
        # ---- 2a. Training ----
        train_cmd = [
            "python3", "-m", "src.cli.train",
            "--experiment-dir",       str(experiment_dir),
            "--dataset",              args.dataset,
            "--model",                model_name,
            "--seed",                 str(args.seed),
            "--batch-size",           str(batch_sizes[i]),
            "--num-epochs",           str(num_epochs_list[i]),
            "--learning-rate",        str(lrs[i]),
            "--learning-rate-decay",  str(lr_decays[i]),
            "--learning-rate-period", str(lr_periods[i]),
            "--gaussian-noise",       str(args.gaussian_noise),
            "--gaussian-noise-std",   str(noise_stds[i]),
            "--validation-size",      str(args.validation_size),
            "--num-workers",          str(args.num_workers),
            "--weight-decay",         str(weight_decays[i]),
            "--weight-decay-type",    wd_types[i],
        ]
        if args.device:
            train_cmd += ["--device", args.device]
 
        exit_code = run(train_cmd, f"TRAINING  {model_name}")
 
        if exit_code != 0:
            print(f"\n  ERROR: training failed for '{model_name}' (exit code {exit_code}).")
            print(  "  Skipping evaluation. Continuing with remaining models.")
            continue
 
        trained_models.append(model_name)
 
        # ---- 2b. Evaluation ----
        best_ckpt = experiment_dir / model_name / "checkpoints" / "best.pth"
 
        if not best_ckpt.exists():
            print(f"\n  WARNING: best.pth not found for '{model_name}' at {best_ckpt}.")
            print(  "  Skipping evaluation.")
            continue
 
        eval_cmd = [
            "python3", "-m", "src.cli.evaluate",
            "--dataset",    args.dataset,
            "--model",      model_name,
            "--checkpoint", str(best_ckpt),
            "--root",       "data/spectra_data/",
            "--seed",       str(args.seed),
            "--val-size",   str(args.validation_size),
            "--batch-size", str(batch_sizes[i]),
        ]
        if args.device:
            eval_cmd += ["--device", args.device]
 
        exit_code = run(eval_cmd, f"EVALUATING {model_name}")
 
        if exit_code != 0:
            print(f"\n  WARNING: evaluation failed for '{model_name}' (exit code {exit_code}).")
        else:
            evaluated_models.append(model_name)
 
    # ------------------------------------------------------------------
    # Step 3 — Summary
    # ------------------------------------------------------------------
 
    print()
    print("=" * 60)
    print("  TRAINING + EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Trained successfully:    {trained_models  or 'none'}")
    print(f"  Evaluated successfully:  {evaluated_models or 'none'}")
    print(f"  Experiment directory:    {experiment_dir}")
 
    if not trained_models:
        print("\n  No models trained successfully. Exiting.")
        sys.exit(1)

    if not evaluated_models:
        print("\n  No models evaluated successfully. Skipping comparison.")
        sys.exit(1)
 
    # ------------------------------------------------------------------
    # Step 4 — Ask whether to run comparison
    # ------------------------------------------------------------------
 
    running_non_interactively = not sys.stdin.isatty()
 
    if args.auto_compare or running_non_interactively:
        if running_non_interactively and not args.auto_compare:
            print("\n  (Non-interactive session detected — running comparison automatically.)")
        should_compare = True
    else:
        if len(evaluated_models) == 1:
            prompt = (
                f"Only one model ({evaluated_models[0]}) was evaluated. "
                f"Run physics-aware comparison (Gaussian-smoothed MSE + overfitting analysis)?"
            )
        else:
            prompt = (
                f"Run comparison across {evaluated_models}? "
                f"Produces MSE/R²/SAM/PWE charts, overfitting summary, metrics.csv, summary.txt."
            )
        should_compare = confirm(prompt)
 
    if not should_compare:
        print("\n  Skipping comparison. Run it later with:")
        print(f"\n    python3 -m src.cli.compare_experiment \\")
        print(f"        --experiment-dir {experiment_dir} \\")
        print(f"        --models {' '.join(evaluated_models)} \\")
        print(f"        --dataset {args.dataset} \\")
        print()
        sys.exit(0)
 
    # ------------------------------------------------------------------
    # Step 5 — Comparison
    # ------------------------------------------------------------------
 
    compare_cmd = [
        "python3", "-m", "src.cli.compare_experiment",
        "--experiment-dir", str(experiment_dir),
        "--models",         *evaluated_models,
        "--dataset",        args.dataset,
        "--seed",           str(args.seed),
        "--val-size",       str(args.validation_size),
        "--device",         "cpu",   # comparison is numpy-heavy; CPU avoids GPU memory spike
    ]
    if args.no_plot:
        compare_cmd += ["--no-plot"]
 
    exit_code = run(compare_cmd, f"COMPARING  {evaluated_models}")
 
    if exit_code != 0:
        print(f"\n  ERROR: comparison failed (exit code {exit_code}).")
        sys.exit(1)
 
    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
 
    print()
    print("=" * 60)
    print("  ALL DONE")
    print("=" * 60)
    print(f"  Results:     {experiment_dir}")
    print(f"  Comparison:  {experiment_dir / '_comparison'}")
    print()
 
 
if __name__ == "__main__":
    main()
