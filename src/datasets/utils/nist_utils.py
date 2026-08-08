# src/datasets/utils/nist_utils.py
from __future__ import annotations
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_NIST_GLOB = "data/spectra_data/raw/nist/IR_data_chunk*_of_009.parquet"

# -------------------------
# Grid helpers
# -------------------------
def build_lambda_grid_um(num_points: int, low_um: float = 1.0, high_um: float = 9.5) -> np.ndarray:
    """Inclusive wavelength grid (µm) for supervision."""
    return np.linspace(low_um, high_um, num_points)

# -------------------------
# Resampling core
# -------------------------
def _resample_one_spectrum(
    nu_cm: np.ndarray,
    I_nu: np.ndarray,
    lam_grid_um: np.ndarray,
    low_nu_cutoff: float = 50.0,
    apply_jacobian: bool = False,
    rms_normalize: bool = True,
) -> np.ndarray:
    """Convert ν→λ and linearly resample onto lam_grid_um. Returns float32."""
    # 1) drop ultra-low ν
    m = (nu_cm >= low_nu_cutoff)
    if m.sum() < 2:
        return np.zeros_like(lam_grid_um, dtype=np.float32)
    nu_cm = nu_cm[m]
    I_nu  = I_nu[m]

    # 2) ν→λ (µm)
    lam_um = 1e4 / nu_cm

    # 3) optional density conversion: Iλ = Iν * (1e4 / λ^2)
    if apply_jacobian:
        I_val = I_nu * (1e4 / (lam_um ** 2))
    else:
        I_val = I_nu

    # 4) sort + band-limit
    order = np.argsort(lam_um)
    lam_um = lam_um[order]
    I_val  = I_val[order]
    lo, hi = lam_grid_um.min(), lam_grid_um.max()
    band = (lam_um >= lo) & (lam_um <= hi)
    if band.sum() < 2:
        return np.zeros_like(lam_grid_um, dtype=np.float32)

    lam_b = lam_um[band]; I_b = I_val[band]

    # 5) interpolate
    out = np.interp(lam_grid_um, lam_b, I_b, left=0.0, right=0.0).astype(np.float32)

    # 6) RMS norm
    if rms_normalize:
        rms = float(np.sqrt(np.mean(out**2)))
        if rms > 0:
            out /= rms
    return out

# -------------------------
# Parquet streaming helpers
# -------------------------
def list_parquet_files(pattern: str = DEFAULT_NIST_GLOB) -> List[Path]:
    p = Path().glob(pattern) if "*" in pattern else [Path(pattern)]
    files = sorted(Path(f) for f in p if Path(f).exists())
    if not files:
        raise FileNotFoundError(f"No Parquet files matched: {pattern}")
    return files

def _total_rows(files: List[Path]) -> Tuple[int, List[int]]:
    totals = []
    for f in files:
        pf = pq.ParquetFile(f)
        totals.append(pf.metadata.num_rows)
    return sum(totals), totals

def _sample_global_indices(total: int, k: int, seed: int = 0) -> np.ndarray:
    """Choose k unique row indices from [0, total)."""
    rng = np.random.default_rng(seed)
    if k >= total:
        return np.arange(total, dtype=np.int64)
    return rng.choice(total, size=k, replace=False)

def _partition_by_file(files: List[Path], file_row_counts: List[int], global_idxs: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Map global row indices to per-file local indices.
    Returns dict: file_idx -> local_idx_array
    """
    out: Dict[int, np.ndarray] = {}
    offsets = np.cumsum([0] + file_row_counts[:-1])
    for fi, (off, n) in enumerate(zip(offsets, file_row_counts)):
        lo, hi = off, off + n
        mask = (global_idxs >= lo) & (global_idxs < hi)
        if mask.any():
            local = (global_idxs[mask] - lo).astype(np.int64)
            out[fi] = np.sort(local)
    return out

def stream_selected_rows_from_parquet(
    files: List[Path],
    select_per_file: Dict[int, np.ndarray],
    lam_grid_um: np.ndarray,
    *,
    low_nu_cutoff: float = 50.0,
    apply_jacobian: bool = False,
    rms_normalize: bool = True,
    batch_size: int = 512,
) -> np.ndarray:
    K = int(sum(len(v) for v in select_per_file.values()))
    X = np.empty((K, lam_grid_um.size), dtype=np.float32)

    write_pos = 0
    for fi, f in enumerate(files):
        if fi not in select_per_file:
            continue
        wanted = select_per_file[fi]
        if wanted.size == 0:
            continue

        pf = pq.ParquetFile(f)
        cols = ["Frequency(cm^-1)", "ir_spectra"]

        next_local = 0
        row_cursor = 0

        for batch in pf.iter_batches(columns=cols, batch_size=batch_size):
            n_in_batch = len(batch)

            # Figure out which wanted rows (if any) fall in THIS batch's
            # range, WITHOUT converting the batch to Python yet.
            hits = []
            probe = next_local
            while probe < wanted.size and wanted[probe] < row_cursor + n_in_batch:
                hits.append(wanted[probe] - row_cursor)  # local offset within this batch
                probe += 1

            if hits:
                # Only now, and only for these specific rows, pay the
                # conversion cost -- batch.take() pulls just the rows we
                # need, still as an Arrow batch, THEN we convert.
                sub = batch.take(hits).to_pydict()
                freqs_sub = sub["Frequency(cm^-1)"]
                specs_sub = sub["ir_spectra"]
                for j in range(len(hits)):
                    nu = np.asarray(freqs_sub[j], dtype=float)
                    I  = np.asarray(specs_sub[j], dtype=float)
                    X[write_pos] = _resample_one_spectrum(
                        nu_cm=nu, I_nu=I, lam_grid_um=lam_grid_um,
                        low_nu_cutoff=low_nu_cutoff,
                        apply_jacobian=apply_jacobian,
                        rms_normalize=rms_normalize,
                    )
                    write_pos += 1
                next_local = probe

            row_cursor += n_in_batch
            if next_local == wanted.size:
                break

    assert write_pos == K, f"Filled {write_pos} rows, expected {K}"
    return X