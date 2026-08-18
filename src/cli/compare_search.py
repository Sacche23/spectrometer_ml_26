import argparse
import json
from pathlib import Path

import numpy as np


# ============================================================
# JSON helper
# ============================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# Find model directory
# ============================================================

def find_model_dir(run_dir):
    """
    Find the model directory inside one hyperparameter-search run.

    Expected structure:

        run_dir/
            cui_mlp_v2/
                evaluation/
                checkpoints/
                training_config.json

    or:

        run_dir/
            unet_8_0.2/
                ...
    """

    candidates = []

    for child in run_dir.iterdir():
        if not child.is_dir():
            continue

        if (child / "evaluation").exists():
            candidates.append(child)

    if not candidates:
        return None

    if len(candidates) > 1:
        print(
            f"WARNING: multiple model directories found in "
            f"{run_dir}: {[x.name for x in candidates]}"
        )

    return candidates[0]


# ============================================================
# Overfitting analysis
# ============================================================

def analyze_overfitting(model_dir):
    """
    Reproduce the overfitting analysis currently performed by
    compare_experiment.py, but return JSON-compatible data
    instead of writing a CSV.

    Uses:
        evaluation/metrics.json
        evaluation/loss_history.json
    """

    metrics_file = model_dir / "evaluation" / "metrics.json"

    if not metrics_file.exists():
        return {
            "available": False,
            "reason": "evaluation/metrics.json not found"
        }

    metrics = load_json(metrics_file)

    # --------------------------------------------------------
    # Train / validation MSE
    # --------------------------------------------------------

    train_mse = metrics["train"]["mse"]
    val_mse = metrics["validation"]["mse"]

    mse_gap = val_mse - train_mse

    if train_mse > 0:
        val_train_ratio = val_mse / train_mse
    else:
        val_train_ratio = None

    # --------------------------------------------------------
    # Loss history
    # --------------------------------------------------------

    history_file = (
        model_dir
        / "evaluation"
        / "loss_history.json"
    )

    if not history_file.exists():
        return {
            "available": True,

            "train_mse": train_mse,
            "val_mse": val_mse,
            "mse_gap": mse_gap,
            "val_train_ratio": val_train_ratio,

            "best_epoch": metrics.get("epoch"),
            "overfit_percent": None,

            "loss_history_available": False
        }

    history = load_json(history_file)

    val_loss = history["val_loss"]
    train_loss = history.get("train_loss")
    epochs = history["epoch"]

    if not val_loss:
        return {
            "available": True,

            "train_mse": train_mse,
            "val_mse": val_mse,
            "mse_gap": mse_gap,
            "val_train_ratio": val_train_ratio,

            "best_epoch": metrics.get("epoch"),
            "overfit_percent": None,

            "loss_history_available": False
        }

    # --------------------------------------------------------
    # Best validation epoch
    # --------------------------------------------------------

    best_idx = min(
        range(len(val_loss)),
        key=lambda i: val_loss[i]
    )

    best_epoch = epochs[best_idx]
    best_val_loss = val_loss[best_idx]
    final_val_loss = val_loss[-1]

    # --------------------------------------------------------
    # Existing compare_experiment overfit metric
    #
    # (final_val - best_val) / best_val * 100
    # --------------------------------------------------------

    if best_val_loss > 0:
        overfit_percent = (
            (final_val_loss - best_val_loss)
            / best_val_loss
            * 100
        )
    else:
        overfit_percent = None

    # --------------------------------------------------------
    # Additional useful overfitting information
    # --------------------------------------------------------

    if train_loss is not None:
        train_loss_at_best = train_loss[best_idx]
        final_train_loss = train_loss[-1]

        best_epoch_gap = (
            best_val_loss - train_loss_at_best
        )

        if train_loss_at_best > 0:
            best_epoch_gap_percent = (
                best_epoch_gap
                / train_loss_at_best
                * 100
            )
        else:
            best_epoch_gap_percent = None

        final_gap = (
            final_val_loss - final_train_loss
        )

        if final_train_loss > 0:
            final_gap_percent = (
                final_gap
                / final_train_loss
                * 100
            )
        else:
            final_gap_percent = None

    else:
        train_loss_at_best = None
        final_train_loss = None
        best_epoch_gap = None
        best_epoch_gap_percent = None
        final_gap = None
        final_gap_percent = None

    return {
        "available": True,

        # Same metrics currently reported by compare_experiment
        "train_mse": train_mse,
        "val_mse": val_mse,
        "mse_gap": mse_gap,
        "val_train_ratio": val_train_ratio,

        # Validation convergence
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,

        # Existing overfit metric
        "overfit_percent": overfit_percent,

        # Additional useful information
        "epochs_after_best": epochs[-1] - best_epoch,
        "validation_degradation": (
            final_val_loss - best_val_loss
        ),

        # Train/validation gap at best epoch
        "train_loss_at_best_epoch": train_loss_at_best,
        "best_epoch_gap": best_epoch_gap,
        "best_epoch_gap_percent": best_epoch_gap_percent,

        # Train/validation gap at final epoch
        "final_train_loss": final_train_loss,
        "final_gap": final_gap,
        "final_gap_percent": final_gap_percent,

        "loss_history_available": True
    }


# ============================================================
# Analyze one search run
# ============================================================

def analyze_run(run_dir, architecture):
    """
    Analyze one hyperparameter configuration.
    """

    model_dir = find_model_dir(run_dir)

    if model_dir is None:
        print(
            f"WARNING: no model directory found in {run_dir}"
        )
        return None

    model_name = model_dir.name

    # --------------------------------------------------------
    # Comparison metrics
    # --------------------------------------------------------

    comparison_metrics_file = (
        run_dir
        / "_comparison"
        / "metrics.json"
    )

    if not comparison_metrics_file.exists():
        print(
            f"WARNING: compare_experiment metrics not found:\n"
            f"  {comparison_metrics_file}"
        )

        return None

    comparison = load_json(
        comparison_metrics_file
    )

    if model_name not in comparison["models"]:
        print(
            f"WARNING: model '{model_name}' not found in "
            f"{comparison_metrics_file}"
        )
        return None

    model_comparison = comparison["models"][model_name]

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    config_file = (
        model_dir
        / "training_config.json"
    )

    if config_file.exists():
        config = load_json(config_file)
    else:
        config = {}

    # --------------------------------------------------------
    # Overfitting
    # --------------------------------------------------------

    overfitting = analyze_overfitting(
        model_dir
    )

    # --------------------------------------------------------
    # Combine everything
    # --------------------------------------------------------

    return {
        "architecture": architecture,
        "model": model_name,

        "run_directory": str(run_dir),

        "hyperparameters": {
            "learning_rate": config.get(
                "learning_rate"
            ),
            "batch_size": config.get(
                "batch_size"
            ),
            "num_epochs": config.get(
                "num_epochs"
            ),
            "learning_rate_decay": config.get(
                "learning_rate_decay"
            ),
            "learning_rate_period": config.get(
                "learning_rate_period"
            ),
            "gaussian_noise": config.get(
                "gaussian_noise"
            ),
            "gaussian_noise_std": config.get(
                "gaussian_noise_std"
            ),
            "validation_size": config.get(
                "validation_size"
            ),
            "seed": config.get(
                "seed"
            ),
        },

        # EXACT metrics produced by compare_experiment.py
        "comparison_metrics": {
            "avg_mse": model_comparison.get(
                "avg_mse"
            ),
            "r2": model_comparison.get(
                "r2"
            ),
            "avg_sam_deg": model_comparison.get(
                "avg_sam_deg"
            ),
            "avg_pwe_um": model_comparison.get(
                "avg_pwe_um"
            ),
            "best_epoch": model_comparison.get(
                "best_epoch"
            ),
        },

        "overfitting": overfitting
    }


# ============================================================
# Analyze all runs in a directory
# ============================================================

def analyze_search(search_dir, architecture):

    search_dir = Path(search_dir)

    if not search_dir.exists():
        print(
            f"WARNING: directory does not exist: "
            f"{search_dir}"
        )
        return []

    results = []

    for run_dir in sorted(search_dir.iterdir()):

        if not run_dir.is_dir():
            continue

        if run_dir.name.startswith("."):
            continue

        result = analyze_run(
            run_dir,
            architecture
        )

        if result is not None:
            results.append(result)

    return results


# ============================================================
# Ranking helpers
# ============================================================

def rank_by_metric(results, metric, higher_is_better=False):

    valid = [
        r for r in results
        if r["comparison_metrics"].get(metric) is not None
    ]

    return sorted(
        valid,
        key=lambda r: r["comparison_metrics"][metric],
        reverse=higher_is_better
    )


# ============================================================
# Formatting
# ============================================================

def fmt(value, digits=5):

    if value is None:
        return "N/A"

    if isinstance(value, float):
        if not np.isfinite(value):
            return "N/A"

        return f"{value:.{digits}g}"

    return str(value)


# ============================================================
# Text report
# ============================================================

def write_text_report(
    output_path,
    mlp_results,
    unet_results
):

    with open(output_path, "w") as f:

        f.write(
            "HYPERPARAMETER SEARCH COMPARISON\n"
        )
        f.write("=" * 100 + "\n\n")

        # ====================================================
        # Function for architecture table
        # ====================================================

        def write_architecture_table(
            title,
            results
        ):

            f.write(title + "\n")
            f.write("-" * 100 + "\n")

            if not results:
                f.write(
                    "No completed comparison results found.\n\n"
                )
                return

            ranked = rank_by_metric(
                results,
                "avg_mse",
                higher_is_better=False
            )

            header = (
                f"{'Rank':<6}"
                f"{'LR':<11}"
                f"{'Batch':<8}"
                f"{'Period':<9}"
                f"{'MSE':<13}"
                f"{'R2':<10}"
                f"{'SAM':<10}"
                f"{'PWE':<10}"
                f"{'BestEp':<9}"
                f"{'Overfit%':<10}"
            )

            f.write(header + "\n")
            f.write("-" * 100 + "\n")

            for i, result in enumerate(
                ranked,
                start=1
            ):

                hp = result["hyperparameters"]
                cm = result["comparison_metrics"]
                ov = result["overfitting"]

                f.write(
                    f"{i:<6}"
                    f"{str(hp['learning_rate']):<11}"
                    f"{str(hp['batch_size']):<8}"
                    f"{str(hp['learning_rate_period']):<9}"
                    f"{fmt(cm['avg_mse']):<13}"
                    f"{fmt(cm['r2']):<10}"
                    f"{fmt(cm['avg_sam_deg']):<10}"
                    f"{fmt(cm['avg_pwe_um']):<10}"
                    f"{str(cm['best_epoch']):<9}"
                    f"{fmt(ov.get('overfit_percent')):<10}"
                    "\n"
                )

            f.write("\n")

        # ====================================================
        # MLP
        # ====================================================

        write_architecture_table(
            "MLP SEARCH",
            mlp_results
        )

        # ====================================================
        # U-Net
        # ====================================================

        write_architecture_table(
            "U-NET SEARCH",
            unet_results
        )

        # ====================================================
        # Best configurations
        # ====================================================

        f.write(
            "BEST CONFIGURATIONS\n"
        )
        f.write("-" * 100 + "\n\n")

        def write_best(
            architecture,
            results
        ):

            if not results:
                return

            ranked = rank_by_metric(
                results,
                "avg_mse",
                higher_is_better=False
            )

            best = ranked[0]

            hp = best["hyperparameters"]
            cm = best["comparison_metrics"]
            ov = best["overfitting"]

            f.write(
                f"BEST {architecture.upper()}\n"
            )

            f.write(
                f"Run: {best['run_directory']}\n"
            )

            f.write(
                f"Learning rate: "
                f"{hp['learning_rate']}\n"
            )

            f.write(
                f"Batch size: "
                f"{hp['batch_size']}\n"
            )

            f.write(
                f"LR decay: "
                f"{hp['learning_rate_decay']}\n"
            )

            f.write(
                f"LR period: "
                f"{hp['learning_rate_period']}\n"
            )

            f.write(
                f"Epochs: "
                f"{hp['num_epochs']}\n"
            )

            f.write("\n")

            f.write(
                f"Average MSE: "
                f"{fmt(cm['avg_mse'])}\n"
            )

            f.write(
                f"R2: "
                f"{fmt(cm['r2'])}\n"
            )

            f.write(
                f"Average SAM: "
                f"{fmt(cm['avg_sam_deg'])} deg\n"
            )

            f.write(
                f"Average PWE: "
                f"{fmt(cm['avg_pwe_um'])} um\n"
            )

            f.write(
                f"Best epoch: "
                f"{cm['best_epoch']}\n"
            )

            f.write("\n")

            if ov["available"]:

                f.write(
                    f"Train MSE: "
                    f"{fmt(ov['train_mse'])}\n"
                )

                f.write(
                    f"Validation MSE: "
                    f"{fmt(ov['val_mse'])}\n"
                )

                f.write(
                    f"MSE gap: "
                    f"{fmt(ov['mse_gap'])}\n"
                )

                f.write(
                    f"Validation/train ratio: "
                    f"{fmt(ov['val_train_ratio'])}\n"
                )

                f.write(
                    f"Overfit %: "
                    f"{fmt(ov['overfit_percent'])}%\n"
                )

                f.write(
                    f"Validation degradation: "
                    f"{fmt(ov['validation_degradation'])}\n"
                )

                f.write(
                    f"Epochs after best: "
                    f"{ov['epochs_after_best']}\n"
                )

            else:

                f.write(
                    f"Overfitting data: "
                    f"{ov.get('reason', 'unavailable')}\n"
                )

            f.write("\n")

        write_best(
            "MLP",
            mlp_results
        )

        write_best(
            "U-Net",
            unet_results
        )

        # ====================================================
        # Direct MLP vs U-Net comparison
        # ====================================================

        if mlp_results and unet_results:

            f.write(
                "BEST MLP VS BEST U-NET\n"
            )
            f.write("-" * 100 + "\n")

            best_mlp = rank_by_metric(
                mlp_results,
                "avg_mse"
            )[0]

            best_unet = rank_by_metric(
                unet_results,
                "avg_mse"
            )[0]

            mlp_cm = best_mlp["comparison_metrics"]
            unet_cm = best_unet["comparison_metrics"]

            f.write(
                f"{'Metric':<25}"
                f"{'MLP':>18}"
                f"{'U-Net':>18}\n"
            )

            f.write("-" * 65 + "\n")

            f.write(
                f"{'MSE':<25}"
                f"{fmt(mlp_cm['avg_mse']):>18}"
                f"{fmt(unet_cm['avg_mse']):>18}\n"
            )

            f.write(
                f"{'R2':<25}"
                f"{fmt(mlp_cm['r2']):>18}"
                f"{fmt(unet_cm['r2']):>18}\n"
            )

            f.write(
                f"{'SAM (deg)':<25}"
                f"{fmt(mlp_cm['avg_sam_deg']):>18}"
                f"{fmt(unet_cm['avg_sam_deg']):>18}\n"
            )

            f.write(
                f"{'PWE (um)':<25}"
                f"{fmt(mlp_cm['avg_pwe_um']):>18}"
                f"{fmt(unet_cm['avg_pwe_um']):>18}\n"
            )

            f.write(
                f"{'Best epoch':<25}"
                f"{str(mlp_cm['best_epoch']):>18}"
                f"{str(unet_cm['best_epoch']):>18}\n"
            )

            f.write("\n")

            # ------------------------------------------------
            # Determine winner for each metric
            # ------------------------------------------------

            comparisons = [
                (
                    "MSE",
                    mlp_cm["avg_mse"],
                    unet_cm["avg_mse"],
                    False
                ),
                (
                    "R2",
                    mlp_cm["r2"],
                    unet_cm["r2"],
                    True
                ),
                (
                    "SAM",
                    mlp_cm["avg_sam_deg"],
                    unet_cm["avg_sam_deg"],
                    False
                ),
                (
                    "PWE",
                    mlp_cm["avg_pwe_um"],
                    unet_cm["avg_pwe_um"],
                    False
                ),
            ]

            f.write(
                "METRIC WINNERS\n"
            )
            f.write("-" * 65 + "\n")

            for name, mlp_value, unet_value, higher in comparisons:

                if higher:
                    if mlp_value > unet_value:
                        winner = "MLP"
                    elif unet_value > mlp_value:
                        winner = "U-Net"
                    else:
                        winner = "Tie"
                else:
                    if mlp_value < unet_value:
                        winner = "MLP"
                    elif unet_value < mlp_value:
                        winner = "U-Net"
                    else:
                        winner = "Tie"

                f.write(
                    f"{name:<25}: {winner}\n"
                )

            f.write("\n")

            # ------------------------------------------------
            # Relative MSE improvement
            # ------------------------------------------------

            if mlp_cm["avg_mse"] != 0:

                mse_improvement = (
                    (
                        mlp_cm["avg_mse"]
                        - unet_cm["avg_mse"]
                    )
                    / mlp_cm["avg_mse"]
                    * 100
                )

                f.write(
                    f"U-Net MSE improvement "
                    f"relative to MLP: "
                    f"{mse_improvement:.3f}%\n"
                )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare MLP and U-Net hyperparameter searches "
            "using compare_experiment metrics and training "
            "overfitting metrics."
        )
    )

    parser.add_argument(
        "--mlp-dir",
        default="experiments/mlp_search"
    )

    parser.add_argument(
        "--unet-dir",
        default="experiments/unet_search"
    )

    parser.add_argument(
        "--output-dir",
        default="experiments/search_comparison"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("HYPERPARAMETER SEARCH COMPARISON")
    print("=" * 70)

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    print("\nSearching MLP runs...")

    mlp_results = analyze_search(
        args.mlp_dir,
        "MLP"
    )

    print(
        f"Found {len(mlp_results)} completed MLP "
        f"comparison runs."
    )

    # --------------------------------------------------------
    # U-Net
    # --------------------------------------------------------

    print("\nSearching U-Net runs...")

    unet_results = analyze_search(
        args.unet_dir,
        "U-Net"
    )

    print(
        f"Found {len(unet_results)} completed U-Net "
        f"comparison runs."
    )

    # --------------------------------------------------------
    # Master JSON
    # --------------------------------------------------------

    output = {
        "mlp": mlp_results,
        "unet": unet_results
    }

    json_path = (
        output_dir
        / "search_comparison.json"
    )

    with open(json_path, "w") as f:
        json.dump(
            output,
            f,
            indent=4
        )

    print(
        f"\nSaved JSON report:"
        f"\n  {json_path}"
    )

    # --------------------------------------------------------
    # Human-readable TXT
    # --------------------------------------------------------

    txt_path = (
        output_dir
        / "search_comparison.txt"
    )

    write_text_report(
        txt_path,
        mlp_results,
        unet_results
    )

    print(
        f"Saved text report:"
        f"\n  {txt_path}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()