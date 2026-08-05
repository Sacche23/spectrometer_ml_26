import argparse
import csv
import json
import shutil
from pathlib import Path


def main():

    parser = argparse.ArgumentParser(
        description="Summarize overfitting across all models in an experiment."
    )

    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Experiment directory (e.g. experiments/unet_grid_analysis)"
    )

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)

    comparison_dir = experiment_dir / "_comparison"
    comparison_dir.mkdir(exist_ok=True)

    out_dir = comparison_dir / "overfitting_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    model_dirs = sorted(
        d for d in experiment_dir.iterdir()
        if d.is_dir() and d.name != "_comparison"
    )

    print(f"\nFound {len(model_dirs)} models.\n")

    for model_dir in model_dirs:

        model = model_dir.name
        eval_dir = model_dir / "evaluation"

        if not eval_dir.exists():
            print(f"Skipping {model}: no evaluation folder.")
            continue

        print(f"Processing {model}")

        # ------------------------------------------------------
        # Copy evaluation files
        # ------------------------------------------------------

        for filename in [
            "metrics.json",
            "loss_history.json",
            "loss_curve.png",
        ]:

            src = eval_dir / filename

            if src.exists():
                dst = out_dir / f"{model}_{filename}"
                shutil.copy2(src, dst)

        # ------------------------------------------------------
        # Read metrics.json
        # ------------------------------------------------------

        metrics_file = eval_dir / "metrics.json"

        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        train_mse = metrics["train"]["mse"]
        val_mse   = metrics["validation"]["mse"]

        mse_gap = val_mse - train_mse

        ratio = (
            val_mse / train_mse
            if train_mse > 0 else float("inf")
        )

        # ------------------------------------------------------
        # Read loss history
        # ------------------------------------------------------

        history_file = eval_dir / "loss_history.json"

        best_epoch = None
        best_val = None
        final_val = None
        overfit_increase = None

        if history_file.exists():

            with open(history_file) as f:
                history = json.load(f)

            val_loss = history["val_loss"]
            epochs = history["epoch"]

            best_idx = min(
                range(len(val_loss)),
                key=lambda i: val_loss[i]
            )

            best_epoch = epochs[best_idx]
            best_val = val_loss[best_idx]
            final_val = val_loss[-1]

            if best_val > 0:
                overfit_increase = (
                    (final_val - best_val)
                    / best_val
                    * 100
                )

        rows.append({

            "model": model,

            "train_mse": train_mse,
            "val_mse": val_mse,

            "mse_gap": mse_gap,
            "val/train": ratio,

            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "final_val_loss": final_val,
            "overfit_%": overfit_increase,

        })

    # ----------------------------------------------------------
    # Sort by least overfitting
    # ----------------------------------------------------------

    rows.sort(key=lambda r: r["mse_gap"])

    csv_file = out_dir / "overfitting_summary.csv"

    with open(csv_file, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n")

    print("=" * 90)
    print("OVERFITTING SUMMARY")
    print("=" * 90)

    print(
        f"{'Model':<18}"
        f"{'Gap':>12}"
        f"{'Ratio':>10}"
        f"{'BestEp':>10}"
        f"{'Overfit %':>14}"
    )

    print("-" * 90)

    for r in rows:

        print(
            f"{r['model']:<18}"
            f"{r['mse_gap']:>12.4e}"
            f"{r['val/train']:>10.3f}"
            f"{str(r['best_epoch']):>10}"
            f"{r['overfit_%']:>14.2f}"
        )

    print("=" * 90)

    print(f"\nSaved summary to:\n{csv_file}")


if __name__ == "__main__":
    main()