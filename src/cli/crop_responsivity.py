#!/usr/bin/env python3
import numpy as np
from pathlib import Path

# Paths (adjust if yours differ)
PROC_DIR = Path("data/responsivity_data/processed")
LAM_PATH = PROC_DIR / "wavelengths.npy"        # length 1000, in METERS
R_PATH   = PROC_DIR / "responsivity.npy"       # 2D, either (n_lambda, n_D) or (n_D, n_lambda)
D_PATH   = PROC_DIR / "displacements.npy"      # unchanged
OUT_DIR  = PROC_DIR / "cropped_2p5_9p5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load
lam_m = np.load(LAM_PATH)          # meters
R     = np.load(R_PATH)            # 2D
D     = np.load(D_PATH)            # leave as-is

# --- Convert wavelength to microns for thresholding/plotting
lam_um = lam_m * 1e6

# --- Basic sanity prints
print(f"wavelengths.npy shape: {lam_m.shape}, meters range: [{lam_m.min():.3e}, {lam_m.max():.3e}]")
print(f"Converted to microns range: [{lam_um.min():.3f}, {lam_um.max():.3f}]")
print(f"responsivity.npy shape: {R.shape} (rows, cols)")

# --- Create crop mask in microns
mask = lam_um >= 2.5
n_keep = int(mask.sum())
print(f"Keeping λ >= 2.5 µm → {n_keep} wavelengths (expected 823)")

if n_keep == 0:
    raise SystemExit("Mask empty! Check units/ranges.")

# --- Decide which axis of R is wavelength by matching lengths
axis0_matches = (R.shape[0] == lam_um.size)
axis1_matches = (R.shape[1] == lam_um.size)

if axis0_matches and not axis1_matches:
    # Wavelength is axis-0 (rows): shape (n_lambda, n_D)
    R_823 = R[mask, :].astype(np.float32)
    lam_823_um = lam_um[mask]
    # displacements unchanged
elif axis1_matches and not axis0_matches:
    # Wavelength is axis-1 (cols): shape (n_D, n_lambda)
    R_823 = R[:, mask].astype(np.float32)
    lam_823_um = lam_um[mask]
else:
    raise SystemExit(
        f"Cannot determine wavelength axis: lam size={lam_um.size}, "
        f"R shape={R.shape}. One dimension of R must equal {lam_um.size}."
    )

print(f"Cropped responsivity shape: {R_823.shape}")
print(f"Cropped wavelength range (µm): [{lam_823_um.min():.3f}, {lam_823_um.max():.3f}]")

# --- Save outputs
np.save(OUT_DIR / "responsivity_823.npy", R_823)
# Save wavelengths in BOTH microns and meters for convenience
np.save(OUT_DIR / "wavelengths_823_um.npy", lam_823_um.astype(np.float32))
np.save(OUT_DIR / "wavelengths_823_m.npy", (lam_823_um / 1e6).astype(np.float64))  # back to meters
# Displacements unchanged
np.save(OUT_DIR / "displacements.npy", D)

print("Saved:")
print(" ", OUT_DIR / "responsivity_823.npy")
print(" ", OUT_DIR / "wavelengths_823_um.npy")
print(" ", OUT_DIR / "wavelengths_823_m.npy")
print(" ", OUT_DIR / "displacements.npy")

# --- Quick integrity check
assert (R_823.shape[0] == n_keep) or (R_823.shape[1] == n_keep), "Crop did not affect the wavelength axis!"
print("Integrity check OK.")
