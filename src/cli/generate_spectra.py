# generate_spectra.py

import numpy as np
import pandas as pd
import os
from pathlib import Path
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Imports
from src.datasets.utils.nist_utils import (
    list_parquet_files,
    build_lambda_grid_um,
    _total_rows,
    _sample_global_indices,
    _partition_by_file,
    stream_selected_rows_from_parquet,
)

# ====================================================================================
# Responsivity resolution (base matrix + on-demand cropping, with caching)
# ====================================================================================

# The "master negative": one real calibration measurement of the BP sensor.
# This never changes across experiments — it's the same physical device.
BASE_DIR = Path("data/responsivity_data/processed")
BASE_WAVELENGTHS_PATH = BASE_DIR / "wavelengths.npy"     # meters, full sensor range
BASE_RESPONSIVITY_PATH = BASE_DIR / "responsivity.npy"   # shape (1000, 41)

# Per-method default wavelength window, used only when the user doesn't
# pass --lam-min/--lam-max explicitly on the command line.
METHOD_DEFAULT_RANGE = {
    "rand_sop":      (1.0, 9.5),
    "uniform":       (1.0, 9.5),
    "rand_sop_nist": (2.5, 9.5),
    "NIST":          (2.5, 9.5),
    "nist":          (2.5, 9.5),
}

def _range_tag(lam_min: float, lam_max: float) -> str:
    """(2.5, 9.5) -> 'cropped_2p5_9p5' -- a filesystem-safe directory name."""
    fmt = lambda x: f"{x:g}".replace(".", "p")
    return f"cropped_{fmt(lam_min)}_{fmt(lam_max)}"

def get_wavelength_grid_and_responsivity(lam_min: float, lam_max: float):
    """
    Resolve (wavelengths_um, R) for the requested [lam_min, lam_max] window.

    - If the requested window covers the sensor's full measured range,
      the base 1000-point files are used directly.
    - Otherwise, crops the base matrix down to that window (no interpolation --
      points outside the window are simply dropped), caches the result to
      disk, and reuses the cache on future calls with the same range.
    """
    lam_full_um = np.load(BASE_WAVELENGTHS_PATH) * 1e6   # meters -> microns
    full_min, full_max = float(lam_full_um.min()), float(lam_full_um.max())

    if lam_min < full_min or lam_max > full_max:
        raise ValueError(
            f"Requested range [{lam_min}, {lam_max}] um is outside the sensor's "
            f"measured range [{full_min:.3f}, {full_max:.3f}] um. The responsivity "
            f"matrix can only be cropped, not extrapolated."
        )

    # Full range requested -> just use the base files, no cropping needed.
    if np.isclose(lam_min, full_min) and np.isclose(lam_max, full_max):
        return lam_full_um, np.load(BASE_RESPONSIVITY_PATH)

    out_dir = BASE_DIR / _range_tag(lam_min, lam_max)
    wl_path = out_dir / "wavelengths_um.npy"
    r_path  = out_dir / "responsivity.npy"

    # Cache hit -- reuse the previously-cropped version.
    if wl_path.exists() and r_path.exists():
        return np.load(wl_path), np.load(r_path)

    # Cache miss -- crop now, from the base matrix, and save for next time.
    R_full = np.load(BASE_RESPONSIVITY_PATH)
    mask = (lam_full_um >= lam_min) & (lam_full_um <= lam_max)
    n_keep = int(mask.sum())
    if n_keep == 0:
        raise ValueError(f"No wavelength points fall inside [{lam_min}, {lam_max}] um.")

    axis0_matches = (R_full.shape[0] == lam_full_um.size)
    axis1_matches = (R_full.shape[1] == lam_full_um.size)
    if axis0_matches and not axis1_matches:
        R_cropped = R_full[mask, :].astype(np.float32)
    elif axis1_matches and not axis0_matches:
        R_cropped = R_full[:, mask].astype(np.float32)
    else:
        raise ValueError(
            f"Cannot determine wavelength axis: wavelengths size={lam_full_um.size}, "
            f"R shape={R_full.shape}."
        )

    lam_cropped_um = lam_full_um[mask].astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(wl_path, lam_cropped_um)
    np.save(r_path, R_cropped)
    print(f"Cropped responsivity to [{lam_min}, {lam_max}] um -> {n_keep} points. "
          f"Cached at {out_dir}")

    return lam_cropped_um, R_cropped

VALID_METHODS = {"rand_sop", 
                 "rand_sop_nist", "rand_sop_NIST", 
                 "uniform", 
                 "NIST", "nist", 
                 "custom1"}

# ====================================================================================
# Dispatcher
# ====================================================================================

def generate_S(rng: np.random.Generator, n_samples: int, wavelengths: np.ndarray, method: str):
    num_points = len(wavelengths)
    if method == "uniform":
        return rng.uniform(size=(n_samples, num_points)).astype(float)
    elif method in ("rand_sop", "rand_sop_nist"):
        return sum_of_peaks(rng, n_samples, wavelengths)
    elif method in ("nist", "NIST"):
        return nist_dataset(n_samples, wavelengths)
    elif method == "custom1":
        return custom1(n_samples, num_points)
    else:
        raise ValueError(f"Unknown method {method}")

# ====================================================================================
# Custom Generation Functions
# ====================================================================================

def sum_of_peaks(
    rng: np.random.Generator,
    num_spectra: int,
    wavelengths: np.ndarray,
    lam: float = 4.0,
    width_frac: tuple[float, float] = (0.005, 0.02),
    noise_std: float = 0.0,
    baseline_std: float = 0.001):
    """
    Generate spectra as a sum of N Gaussian peaks (N ~ ZTPoisson(lam)),
    then add a linear baseline + noise and normalize to unit RMS.
    Operates on whatever wavelength grid is passed in -- the full 1000-point
    sensor grid for 'rand_sop', or a cropped grid for 'rand_sop_nist'.
    """
    # build wavelength axis
    num_points = len(wavelengths)
    low, high = float(wavelengths.min()), float(wavelengths.max())
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
    wavelengths: np.ndarray,
):
    """
    Memory-safe loader for Parquet IR data.

    - Scans Parquet metadata to count total rows (no heavy load).
    - Randomly selects `num_spectra` rows across all files (or all rows if num_spectra >= total).
    - Streams selected rows with PyArrow iter_batches(), converts ν→λ, and linearly resamples
      onto an inclusive λ-grid from 2.5-9.5 µm of length `num_points`.
    - Returns a float32 array of shape (num_spectra, num_points).

    """
    files = list_parquet_files("data/spectra_data/raw/nist/IR_data_chunk*_of_009.parquet")

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
        lam_grid_um=wavelengths,
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
         method: str,
         lam_min: float,
         lam_max: float,
         ):
    
    if method not in VALID_METHODS:
        raise ValueError(f"Invalid method '{method}'. Valid choices are: {sorted(VALID_METHODS)}")

    default_min, default_max = METHOD_DEFAULT_RANGE[method]
    lam_min = default_min if lam_min is None else lam_min
    lam_max = default_max if lam_max is None else lam_max

    wavelengths, R = get_wavelength_grid_and_responsivity(lam_min, lam_max)
    print(f"Using {len(wavelengths)}-point grid over [{lam_min}, {lam_max}] um "
          f"(method={method})")
    
    out_dir = Path(f"data/spectra_data/processed/{method}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    S = generate_S(rng=rng, n_samples=n_samples, wavelengths=wavelengths, method=method)

    I = S @ R


    np.save(out_dir / f"S.npy", S)
    np.save(out_dir / f"I.npy", I)
    print(f"Wrote S.shape={S.shape}, I.shape={I.shape} to {out_dir!r}")

    metadata = {
        "dataset_version": 2,
        "method": method,
        "seed": seed,
        "n_samples": n_samples,
        "s_dim": S.shape[1],
        "lam_min_um": lam_min,
        "lam_max_um": lam_max,
        "S_shape": list(S.shape),
        "I_shape": list(I.shape),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=50000)
    p.add_argument("--method", type=str, required=True, choices=sorted(VALID_METHODS))
    p.add_argument("--lam-min", type=float, default=None,
                   help="Lower wavelength bound (um). Defaults to the method's "
                        "standard range: 1.0um for rand_sop/uniform, 2.5um for NIST/rand_sop_nist.")
    p.add_argument("--lam-max", type=float, default=None,
                   help="Upper wavelength bound (um). Defaults to 9.5um.")
    args = p.parse_args()
    main(
        args.seed, 
        args.n_samples, 
        args.method, 
        args.lam_min, 
        args.lam_max)
