import numpy as np
import pandas as pd
import os
from pathlib import Path
import argparse

# Imports
from src.datasets.utils.nist_utils import (
    list_parquet_files,
    build_lambda_grid_um,
    _total_rows,
    _sample_global_indices,
    _partition_by_file,
    stream_selected_rows_from_parquet,
)

VALID_METHODS = {"rand_sop", "uniform", "NIST", "custom1"}

def generate_S(rng: np.random.Generator, n_samples: int, s_dim : int, method : str):
    if method == "uniform":
        return rng.uniform(size=(n_samples, s_dim)).astype(float)
    elif method == "rand_sop":
        return sum_of_peaks(rng, n_samples, s_dim)
    elif method == "NIST":
        return nist_dataset(n_samples, s_dim)
    elif method == "custom1":
        return custom1(n_samples, s_dim)
    else:
        # should never hit this
        raise ValueError(f"Unknown method {method}")

# ====================================================================================
# Custom Generation Functions
# ====================================================================================

def sum_of_peaks(
    rng: np.random.Generator,
    num_spectra: int,
    num_points: int,
    low: float = 1.0,
    high: float = 9.5,
    lam: float = 4.0,
    width_frac: tuple[float, float] = (0.005, 0.02),
    noise_std: float = 0.0,
    baseline_std: float = 0.001):
    """
    Generate spectra as a sum of N Gaussian peaks (N ~ ZTPoisson(lam)),
    then add a linear baseline + noise and normalize to unit RMS.
    """
    # build wavelength axis
    wavelengths = np.linspace(low, high, num_points)
    span = high - low

    data = np.zeros((num_spectra, num_points), dtype=float)
    for i in range(num_spectra):
        # number of peaks
        Np = 0
        while Np == 0:
            Np = rng.poisson(lam) #zero-truncated poission
        # add peaks
        for _ in range(Np):
            center = rng.uniform(low, high)
            width  = rng.uniform(width_frac[0]*span, width_frac[1]*span)
            amp    = rng.lognormal(mean=0.0, sigma=1.0)
            data[i] += amp * np.exp(-(wavelengths - center)**2 / (2 * width**2))
        # linear baseline
        b0 = rng.normal(0.0, baseline_std)
        b1 = rng.normal(0.0, baseline_std)
        data[i] += b0 + b1 * (wavelengths - (low + span/2))
        # white noise
        data[i] += rng.normal(0.0, noise_std, size=num_points)
        # normalize to unit RMS
        rms = np.sqrt(np.mean(data[i]**2))
        data[i] /= (rms + 1e-6)

    return data

# ====================================================================================
# Custom Data Sampling Functions
# ====================================================================================

def nist_dataset(
    num_spectra: int,
    num_points: int
):
    """
    Memory-safe loader for Parquet IR data.

    - Scans Parquet metadata to count total rows (no heavy load).
    - Randomly selects `num_spectra` rows across all files (or all rows if num_spectra >= total).
    - Streams selected rows with PyArrow iter_batches(), converts ν→λ, and linearly resamples
      onto an inclusive λ-grid from 1.0..9.5 µm of length `num_points`.
    - Returns a float32 array of shape (num_spectra, num_points).

    """
    files = list_parquet_files("data/spectra_data/raw/nist/IR_data_chunk*_of_009.parquet")

    # Build target λ-grid (inclusive endpoints). Change low_um=2.5 if you want to drop sub-2.5 entirely.
    lam_grid_um = build_lambda_grid_um(num_points, low_um=1.0, high_um=9.5)

    # 1) Count total rows across files (lightweight; metadata only)
    total, counts = _total_rows(files)

    # 2) Choose exactly num_spectra rows globally (or all if requesting more)
    global_idxs = _sample_global_indices(total, num_spectra, seed=0)

    # 3) Map to per-file local indices
    per_file = _partition_by_file(files, counts, global_idxs)

    # 4) Stream only those rows and resample each
    X = stream_selected_rows_from_parquet(
        files=files,
        select_per_file=per_file,
        lam_grid_um=lam_grid_um,
        low_nu_cutoff=50.0,
        apply_jacobian=True,   # set True if you want per-µm intensity
        rms_normalize=True,
        batch_size=512,
    )
    return X


def custom1(
    num_spectra: int,
    num_points: int
):
    # TODO: Import dataset to work with, take a sample of num_spectra spectra, and then
    # sample each of these at num_points points (may require linear interpolation), and
    # finally return a numpy array of shape (num_spectra, num_points) of floats.
    return np.zeros((num_spectra, num_points), dtype=float)

# ====================================================================================
# Main Script
# ====================================================================================    

def main(seed: int,
         n_samples: int,
         s_dim: int,
         method: str):
    
    if method not in VALID_METHODS:
        raise ValueError(f"Invalid method '{method}'. Valid choices are: {sorted(VALID_METHODS)}")
    
    out_dir = Path(f"data/spectra_data/processed/{method}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    S = generate_S(rng=rng, n_samples=n_samples, s_dim=s_dim, method=method)

    R = np.load(args.responsivity)
    I = S @ R

    if method in {"NIST", "custom1", }: # Add every real dataset here
        np.save(out_dir / f"S.npy", S)
        np.save(out_dir / f"I.npy", I)
    else:
        np.save(out_dir / f"S_s{seed}_{n_samples}x{s_dim}.npy", S)
        np.save(out_dir / f"I_s{seed}_{n_samples}x{s_dim}.npy", I)
    print(f"Wrote S.shape={S.shape}, I.shape={I.shape} to {out_dir!r}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--n-samples",  type=int,   default=50000)
    p.add_argument("--s-dim",      type=int,   default=1000,
                   help="length of each 'true' spectrum")
    p.add_argument("--method",     type=str,   required=True,
                   choices=sorted(VALID_METHODS),
                   help="generation method")
    p.add_argument("--responsivity",type=str,  required=True)
    args = p.parse_args()
    main(args.seed, args.n_samples, args.s_dim, args.method)
