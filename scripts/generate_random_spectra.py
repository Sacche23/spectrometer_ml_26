import numpy as np
import os
from pathlib import Path
import argparse


def main(seed: int,
         n_samples: int,
         y_dim: int,
         x_dim: int,
         out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 1) generate N "true" spectra, each of length y_dim
    y = rng.normal(size=(n_samples, y_dim)).astype(np.float32)

    # 2) load (or randomly generate) your responsivity A
    # If you have a real responsivity matrix on disk, replace this:
    # A = rng.normal(size=(y_dim, x_dim)).astype(np.float32)
    A = np.load("data/responsivity_data/processed/responsivity.npy")
    # if instead you do: A = np.load(raw_dir/"responsivity.npy")

    # 3) compute N detector‐read vectors, each of length x_dim
    X = y @ A

    # 4) save out
    np.save(out_dir / f"y_s{seed}.npy", y)
    np.save(out_dir / f"X_s{seed}.npy", X)

    print(f"Wrote y.shape={y.shape}, X.shape={X.shape} to {out_dir}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--n-samples",  type=int,   default=50000)
    p.add_argument("--y-dim",      type=int,   default=1000,
                   help="length of each 'true' spectrum")
    p.add_argument("--x-dim",      type=int,   default=41,
                   help="number of measurement channels")
    p.add_argument("--out-dir",    type=Path,  default=Path("data/spectra_data/processed/rand_sop"))
    args = p.parse_args()
    main(args.seed, args.n_samples, args.y_dim, args.x_dim, args.out_dir)
