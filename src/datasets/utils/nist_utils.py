# src/datasets/utils/nist_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple
import numpy as np
import pandas as pd
import glob

# Default location for your Parquet chunks
DEFAULT_NIST_GLOB = "data/spectra_data/raw/nist/IR_data_chunk*_of_009.parquet"

def list_parquet_files(pattern: str = DEFAULT_NIST_GLOB) -> List[str]:
    """Return sorted list of Parquet files to use."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No Parquet files matched: {pattern}\n"
            "Expected files like data/spectra_data/raw/nist/IR_data_chunk001_of_009.parquet"
        )
    return files

def build_lambda_grid_um(num_points: int, low_um: float = 1.0, high_um: float = 9.5) -> np.ndarray:
    """Inclusive wavelength grid (µm) for supervision."""
    return np.linspace(low_um, high_um, num_points)

def _resample_one_spectrum(
    nu_cm: np.ndarray,
    I_nu: np.ndarray,
    lam_grid_um: np.ndarray,
    low_nu_cutoff: float = 50.0,
    apply_jacobian: bool = True,
    rms_normalize: bool = True,
) -> np.ndarray:
    """
    Convert ν→λ and resample intensity onto lam_grid_um.
    - Drops ultra-low ν to avoid DC artifact
    - Optionally applies Jacobian: Iλ = Iν * (1e4 / λ^2)
    - Returns float32 array len = len(lam_grid_um)
    """
    # 1) drop ultra-low wavenumbers (zero-frequency artifact)
    m = (nu_cm >= low_nu_cutoff)
    if m.sum() < 2:
        return np.zeros_like(lam_grid_um, dtype=np.float32)
    nu_cm = nu_cm[m]
    I_nu  = I_nu[m]

    # 2) ν(cm^-1) → λ(µm)
    lam_um = 1e4 / nu_cm

    # 3) optionally convert intensity density to per-µm
    if apply_jacobian:
        I_val = I_nu * (1e4 / (lam_um ** 2))
    else:
        I_val = I_nu

    # 4) sort by λ and band-limit to grid span
    order = np.argsort(lam_um)
    lam_um = lam_um[order]
    I_val  = I_val[order]

    lo, hi = lam_grid_um.min(), lam_grid_um.max()
    band = (lam_um >= lo) & (lam_um <= hi)
    if band.sum() < 2:
        return np.zeros_like(lam_grid_um, dtype=np.float32)

    lam_b = lam_um[band]
    I_b   = I_val[band]

    # 5) linear interpolation to the target grid
    I_res = np.interp(lam_grid_um, lam_b, I_b, left=0.0, right=0.0).astype(np.float32)

    # 6) optional RMS normalization (matches your simulated pipeline)
    if rms_normalize:
        rms = float(np.sqrt(np.mean(I_res ** 2)))
        if rms > 0:
            I_res /= rms

    return I_res

def resample_from_parquet_file(
    parquet_path: str,
    row_indices: Iterable[int],
    lam_grid_um: np.ndarray,
    low_nu_cutoff: float = 50.0,
    apply_jacobian: bool = False,
    rms_normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load specific rows from a Parquet file and resample onto lam_grid_um.
    Returns:
      X  -> np.ndarray shape (len(row_indices), len(lam_grid_um))
      ids -> np.ndarray of row ids (optional; for tracing)
    Assumes columns: 'Frequency(cm^-1)' and 'ir_spectra' hold equal-length arrays.
    """
    df = pd.read_parquet(parquet_path)  # load once (you can optimize later if needed)
    X_list = []
    ids = []

    for i in row_indices:
        row = df.iloc[i]
        nu = np.asarray(row["Frequency(cm^-1)"], dtype=float)
        I  = np.asarray(row["ir_spectra"], dtype=float)

        x = _resample_one_spectrum(
            nu_cm=nu,
            I_nu=I,
            lam_grid_um=lam_grid_um,
            low_nu_cutoff=low_nu_cutoff,
            apply_jacobian=apply_jacobian,
            rms_normalize=rms_normalize,
        )
        X_list.append(x)
        ids.append(row.get("id", i))

    X = np.vstack(X_list).astype(np.float32)
    return X, np.asarray(ids)
