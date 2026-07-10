import argparse
import torch
from torch.utils.data import DataLoader
from src.datasets.registry import get_dataset, DATASET_REGISTRY
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from src.models.model import *
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import json
import shutil


def train(
        experiment_dir: str|None,

        dataset_name: str,
        seed: int,

        batch_size: int,
        num_epochs: int,

        learning_rate: float,
        learning_rate_decay: float,
        learning_rate_period: int,

        gaussian_noise: bool,
        gaussian_noise_std: float,

        validation_size: int,

        model_type: str,
        
        num_workers: int=0,
        device: torch.device=None,
        resume: str=None):
    
    # TRANSFORMATIONS

    # add transformations here in future if necessary!!!
    
    # PRINT MODEL 

    print(f"Training model {model_type}")

    # DEVICE SETUP

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DATASET AND DATALOADER

    DSClass = get_dataset(dataset_name)
    ds = DSClass(
        root="./data/spectra_data/",
        seed=seed
    )

    # PRINT DATASET INFO
    print("\nDataset information:")

    if hasattr(ds, "metadata"):
        for key, value in ds.metadata.items():
            print(f"  {key}: {value}")

    print(f"  Number of samples: {len(ds)}")

    total = len(ds)
    val_size = validation_size
    train_size = total-val_size
    train_ds, val_ds = torch.utils.data.random_split(
        ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed) # for reproducibility
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type=="cuda"))

    # MODEL

    x0, y0 = ds[0]
    input_size, output_size = len(x0), len(y0)
    print(f"Input dimension : {input_size}")
    print(f"Output dimension: {output_size}")
    ModelClass = get_model(model_type)
    model = ModelClass(input_dim=input_size, output_dim=output_size).to(device)
    print(f"Model is on device: {next(model.parameters()).device}")

    # LOSS AND OPTIMIZER

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=learning_rate_period, gamma=learning_rate_decay) # multiply lr * gamma every N steps

    # TRACK BEST EPOCH AND LOSS
    best_val_loss = float("inf")
    best_epoch = 1

    # LOAD PREVIOUS STATE

    start_epoch = 1

    if resume is not None:
        checkpoint = torch.load(resume, map_location=device)
        # if saved a dict of all states:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint: 
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', optimizer.state_dict()))
            scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict()))
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            best_epoch = checkpoint.get("best_epoch", 1)
        else:
            # only saved model.state_dict()- expected case
            model.load_state_dict(checkpoint)
        print(f"Resumed from {resume!r}, starting at epoch {start_epoch}")

    # LOG AND CHECKPOINT SETUP

    if experiment_dir is None or experiment_dir.lower() == "auto":
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_dir = (
            Path("experiments")
            / f"{ts}_{dataset_name}_seed{seed}"
        )
    else:
        experiment_dir = Path(experiment_dir) 

    model_dir = experiment_dir / model_type
    comparison_dir = experiment_dir / "_comparison"

    logs_dir = model_dir / "logs"
    ckpt_dir = model_dir / "checkpoints"
    eval_dir = model_dir / "evaluation"

    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    print(f"Logging to:      {logs_dir}")
    print(f"Checkpoints to:  {ckpt_dir}")

    writer = SummaryWriter(log_dir=str(logs_dir))
    
    # SAVE DATASET METADATA

    dataset_meta = Path("./data/spectra_data/processed") / dataset_name / "dataset.json"

    dataset_copy = experiment_dir / "dataset.json"

    if dataset_meta.exists() and not dataset_copy.exists():
        shutil.copy2(dataset_meta, dataset_copy)

    training_config = {
        "model": model_type,
        "dataset": dataset_name,
        "seed": seed,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "learning_rate_decay": learning_rate_decay,
        "learning_rate_period": learning_rate_period,
        "gaussian_noise": gaussian_noise,
        "gaussian_noise_std": gaussian_noise_std,
        "validation_size": validation_size,
        "device": str(device),
        "input_dim": input_size,
        "output_dim": output_size,
        "experiment_dir": str(experiment_dir),
    }

    with open(model_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=4)

    # TRAINING LOOP

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (currents, spectra) in enumerate(train_loader, start=1):
            currents = currents.to(device, non_blocking=True)
            spectra = spectra.to(device, non_blocking=True)

            if gaussian_noise:
                currents = currents + torch.randn_like(currents) * gaussian_noise_std

            optimizer.zero_grad()
            outputs = model(currents)
            loss = criterion(outputs, spectra)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * currents.size(0)
        
        scheduler.step()
        train_loss = running_loss/train_size

        # Validation

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for currents, spectra in val_loader:
                currents = currents.to(device).float()
                spectra  = spectra.to(device).float()
                preds    = model(currents)
                val_loss += criterion(preds, spectra).item() * currents.size(0)
        val_loss = val_loss / val_size

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch

        # Save checkpoints

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch
        }
        
        # Saves latest epoch to resume training if necessary
        torch.save(
                checkpoint,
                ckpt_dir / "latest.pth"
            )
        
        # Update and save best loss
        if is_best:
            torch.save(
                checkpoint,
                ckpt_dir / "best.pth"
            )
        
        # Plot and save epochs

        if epoch % 10 == 0:
            # grab one batch from val
            xb, yb = next(iter(val_loader))
            xb, yb = xb.to(device).float(), yb.to(device).float()
            with torch.no_grad():
                y_pred = model(xb)

            n_plots = 16
            fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
            axes = axes.flatten()

            for i in range(n_plots):
                ax = axes[i]
                ax.plot(yb[i].cpu().numpy(),    label="True")
                ax.plot(y_pred[i].cpu().numpy(), label="Pred")
                ax.set_title(f"#{i}")
                ax.axis("off")

            fig.tight_layout()
            writer.add_figure("Predictions/val_grid", fig, epoch)
            plt.close(fig)

            print(f"Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}")
            torch.save(
                checkpoint,
                ckpt_dir / f"epoch{epoch:03d}.pth"
            )
    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")        

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment-dir", type=str, default=None, help="Existing experiment directory. If omitted, create a new experiment.")
    parser.add_argument('--dataset', type=str, required=True, help='Name of dataset to train on', choices=sorted(DATASET_REGISTRY))
    parser.add_argument('--model', type=str, required=True, help='Name of model', choices=sorted(MODEL_REGISTRY))
    parser.add_argument('--seed', type=int, required=False, default=42, help='RNG seed')
    
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size')
    parser.add_argument('--num-epochs', type=int, required=True, help='Number of epochs')
    
    parser.add_argument('--learning-rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--learning-rate-decay', type=float, required=True, help='Every period learning rate is mutliplied by this factor')
    parser.add_argument('--learning-rate-period', type=int, required=True, help='Number of epochs before learning rate decay occurs')

    parser.add_argument('--gaussian-noise', type=bool, required=True, help='True if gaussian noise is to be added onto current vectors')
    parser.add_argument('--gaussian-noise-std', type=float, required=False, default=1e-4, help='Standard deviation of gaussian noise')

    parser.add_argument('--validation-size', type=int, required=True, help='Size of validation dataset')

    parser.add_argument('--num-workers', type=int, required=False, default=0, help='Number of workers')
    parser.add_argument('--device', type=str, required=False, default=None, help='device (cpu, cuda)')

    parser.add_argument('--resume', type=str, default=None, help='Path to a .pth checkpoint to resume training from')

    args = parser.parse_args()


    train(experiment_dir=args.experiment_dir,
          dataset_name=args.dataset,
          seed=args.seed,
          batch_size=args.batch_size,
          num_epochs=args.num_epochs,
          learning_rate=args.learning_rate,
          learning_rate_decay=args.learning_rate_decay,
          learning_rate_period=args.learning_rate_period,
          gaussian_noise=args.gaussian_noise,
          gaussian_noise_std=args.gaussian_noise_std,
          validation_size=args.validation_size,
          model_type=args.model,
          num_workers=args.num_workers,
          device=torch.device(args.device) if args.device else None,
          resume=args.resume)