import numpy as np
import os
from pathlib import Path
import argparse

VALID_METHODS = {"rand_sop", "uniform"}

def generate_S(rng: np.random.Generator, n_samples: int, s_dim : int, method : str):
    if method == "uniform":
        return rng.uniform(size=(n_samples, s_dim)).astype(float)
    elif method == "rand_sop":
        return sum_of_peaks(rng, n_samples, s_dim)
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

    if method == "uniform":
        S = rng.uniform(size=(n_samples, s_dim)).astype(float)
    elif method == "rand_sop":
        S = sum_of_peaks(rng=rng, num_spectra=n_samples, num_points=s_dim)

    R = np.load("data/responsivity_data/processed/responsivity.npy")
    I = S @ R

    np.save(out_dir / f"S_s{seed}.npy", S)
    np.save(out_dir / f"I_s{seed}.npy", I)
    print(f"Wrote S.shape={S.shape}, I.shape={I.shape} to {out_dir!r}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--n-samples",  type=int,   default=50000)
    p.add_argument("--s-dim",      type=int,   default=1000,
                   help="length of each 'true' spectrum")
    p.add_argument("--method",     type=str,   required=True,
                   choices=sorted(VALID_METHODS),
                   help="random generation method")
    args = p.parse_args()
    main(args.seed, args.n_samples, args.s_dim, args.method)
