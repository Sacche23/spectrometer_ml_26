#!/usr/bin/env python3

import numpy as np
import torch
from datasets.registry import get_dataset
import matplotlib.pyplot as plt

def main():
    # -- Load full responsivity matrix
    #    adjust path if your processed file lives elsewhere
    R = np.load("data/responsivity_data/processed/responsivity.npy")
    print(f"Responsivity matrix shape: {R.shape}")

    # -- Instantiate the rand_sop dataset
    DS = get_dataset("rand_sop")
    ds = DS(root="data/spectra_data/")  # change root if needed

    # -- Get first sample: (measurement, spectrum)
    meas_tensor, spec_tensor = ds[0]
    plt.plot(spec_tensor)
    plt.show()

    # Convert spectrum to NumPy
    spec_np = spec_tensor.detach().cpu().numpy() if torch.is_tensor(spec_tensor) else spec_tensor
    print(f"First spectrum shape: {spec_np.shape}")

    # -- Compute predicted measurement = R^T @ spectrum
    #    If R is (n_wavelengths, n_meas), then R^T is (n_meas, n_wavelengths)
    pred = R.T @ spec_np
    print(f"Predicted measurement shape: {pred.shape}")
    print("Predicted measurement:\n", pred)
    print(meas_tensor)

if __name__ == "__main__":
    main()
