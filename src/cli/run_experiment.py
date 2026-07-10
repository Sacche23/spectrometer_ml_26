"""
run_experiment.py
-----------------
Mother script that runs a full experiment end-to-end:

  1. Creates one shared experiment directory
  2. For each model in --models:
       a. Trains the model
       b. Evaluates the best checkpoint
  3. Prompts the user for confirmation
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
        description="End-to-end experiment runner: train → evaluate → compare."
    )

    # ---- What to run ----
    p.add_argument("--models", nargs="+", required=True,
                   help="One or more model names to train and compare. "
                        "Example: --models dnn cnn2 unet")
    p.add_argument("--dataset", required=True,
                   help="Registered dataset name, e.g. rand_sop")

    # ---- Experiment directory ----
    p.add_argument("--experiment-dir", type=str, default=None,
                   help="Path to use for the experiment directory. "
                        "If omitted, one is created automatically under experiments/ "
                        "using the current timestamp, dataset, and seed.")

    # ---- Training hyperparameters (passed straight through to train.py) ----
    p.add_argument("--seed",                  type=int,   default=42)
    p.add_argument("--batch-size",            type=int,   default=128)
    p.add_argument("--num-epochs",            type=int,   default=1000)
    p.add_argument("--learning-rate",         type=float, default=1e-3)
    p.add_argument("--learning-rate-decay",   type=float, default=0.6)
    p.add_argument("--learning-rate-period",  type=int,   default=200)
    p.add_argument("--gaussian-noise",        type=bool,  default=True)
    p.add_argument("--gaussian-noise-std",    type=float, default=1e-4)
    p.add_argument("--validation-size",       type=int,   default=200)
    p.add_argument("--num-workers",           type=int,   default=0)
    p.add_argument("--device",                type=str,   default=None)

    # ---- Comparison settings ----
    p.add_argument("--responsivity", required=True,
                   help="Path to responsivity matrix .npy file, "
                        "e.g. data/responsivity_data/processed/responsivity.npy")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip per-sample plots in the comparison step (faster).")

    args = p.parse_args()

    # ------------------------------------------------------------------
    # Step 0 — Resolve the experiment directory
    # ------------------------------------------------------------------
    # All models in this run share one experiment directory so their
    # results land side by side under experiments/<name>/dnn/, /cnn2/, etc.

    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_dir = (
            Path("experiments")
            / f"{ts}_{args.dataset}_seed{args.seed}"
        )

    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExperiment directory: {experiment_dir}")
    print(f"Models to train: {args.models}")

    # ------------------------------------------------------------------
    # Step 1 — Train + Evaluate each model in sequence
    # ------------------------------------------------------------------

    trained_models  = []   # models that completed training successfully
    evaluated_models = []  # models that also completed evaluation

    for model_name in args.models:

        # ---- 1a. Training ----
        train_cmd = [
            "python3", "-m", "src.cli.train",
            "--experiment-dir",       str(experiment_dir),
            "--dataset",              args.dataset,
            "--model",                model_name,
            "--seed",                 str(args.seed),
            "--batch-size",           str(args.batch_size),
            "--num-epochs",           str(args.num_epochs),
            "--learning-rate",        str(args.learning_rate),
            "--learning-rate-decay",  str(args.learning_rate_decay),
            "--learning-rate-period", str(args.learning_rate_period),
            "--gaussian-noise",       str(args.gaussian_noise),
            "--gaussian-noise-std",   str(args.gaussian_noise_std),
            "--validation-size",      str(args.validation_size),
            "--num-workers",          str(args.num_workers),
        ]
        if args.device:
            train_cmd += ["--device", args.device]

        exit_code = run(train_cmd, f"TRAINING  {model_name}")

        if exit_code != 0:
            print(f"\n  ERROR: training failed for '{model_name}' (exit code {exit_code}).")
            print(  "  Skipping evaluation for this model.")
            print(  "  The experiment will continue with the remaining models.")
            continue  # move on to the next model rather than crashing the whole run

        trained_models.append(model_name)

        # ---- 1b. Evaluation ----
        # Checkpoint path is deterministic: train.py always saves best.pth here.
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
            "--batch-size", str(args.batch_size),
        ]
        if args.device:
            eval_cmd += ["--device", args.device]

        exit_code = run(eval_cmd, f"EVALUATING {model_name}")

        if exit_code != 0:
            print(f"\n  WARNING: evaluation failed for '{model_name}' (exit code {exit_code}).")
        else:
            evaluated_models.append(model_name)

    # ------------------------------------------------------------------
    # Step 2 — Summary of what finished
    # ------------------------------------------------------------------

    print()
    print("=" * 60)
    print("  TRAINING + EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Trained successfully : {trained_models  or 'none'}")
    print(f"  Evaluated successfully: {evaluated_models or 'none'}")
    print(f"  Experiment directory : {experiment_dir}")

    if not trained_models:
        print("\n  No models trained successfully. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3 — Ask whether to run comparison
    # ------------------------------------------------------------------

    if len(evaluated_models) == 1:
        prompt = (
            f"Only one model ({evaluated_models[0]}) was trained. "
            f"Run physics-aware comparison (Gaussian-smoothed MSE + sample plots)?"
        )
    else:
        prompt = (
            f"Run comparison across {evaluated_models}? "
            f"This produces MSE/R² charts, per-sample plots, metrics.csv, and summary.txt."
        )

    if not confirm(prompt):
        print("\n  Skipping comparison. You can run it later with:")
        print(f"\n    python3 -m src.cli.compare_experiment \\")
        print(f"        --experiment-dir {experiment_dir} \\")
        print(f"        --models {' '.join(evaluated_models)} \\")
        print(f"        --dataset {args.dataset} \\")
        print(f"        --responsivity {args.responsivity}")
        print()
        sys.exit(0)

    # ------------------------------------------------------------------
    # Step 4 — Comparison
    # ------------------------------------------------------------------
    # compare_experiment works correctly for both 1 and multiple models.
    # For a single model it still gives you Gaussian-smoothed MSE, R²,
    # per-sample plots, and metrics files — more informative than the
    # raw evaluate.py output.

    compare_cmd = [
        "python3", "-m", "src.cli.compare_experiment",
        "--experiment-dir", str(experiment_dir),
        "--models",         *evaluated_models,
        "--dataset",        args.dataset,
        "--responsivity",   args.responsivity,
        "--seed",           str(args.seed),
        "--val-size",       str(args.validation_size),
    ]
    if args.device:
        compare_cmd += ["--device", args.device]
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
    print(f"  Results saved to: {experiment_dir}")
    print(f"  Comparison:       {experiment_dir / '_comparison'}")
    print()


if __name__ == "__main__":
    main()
