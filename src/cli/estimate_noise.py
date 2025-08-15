import numpy as np
import argparse

def main():
    p = argparse.ArgumentParser(description="Estimate realistic noise for BP spectrometer data.")
    p.add_argument("--resp", default="data/responsivity_data/processed/responsivity.npy")
    p.add_argument("--currents", default="data/spectra_data/processed/rand_sop/I_s42.npy")
    p.add_argument("--spectra", default="data/spectra_data/processed/rand_sop/S_s42.npy")
    p.add_argument("--subset", type=int, default=1000, help="Subset size for estimation to avoid memory issues.")
    args = p.parse_args()

    # Load data
    print("loaded")
    R = np.load(args.resp)  # (m, n)
    print("loaded")
    I = np.load(args.currents)  # (Nspec, m)
    print("loaded")
    S = np.load(args.spectra)   # (Nspec, n)

    # Ensure orientation: R rows == m
    if R.shape[0] != I.shape[1]:
        R = R.T

    # Subset
    Nspec = I.shape[0]
    if args.subset > 0 and args.subset < Nspec:
        idx = np.random.choice(Nspec, args.subset, replace=False)
        I = I[idx]
        S = S[idx]
        print(f"Using subset of {args.subset} spectra")

    print(f"R shape: {R.shape}, I shape: {I.shape}, S shape: {S.shape}")

    # === Physics-based noise estimate ===
    # Assumptions
    e = 1.602e-19  # C
    bandwidth = 1e3  # Hz, assume 1 kHz readout bandwidth
    integration_time = 1 / bandwidth  # s

    # Step 1: RMS current from *simulated* dataset
    rms_current_sim = np.sqrt(np.mean(I**2, axis=1))  # In simulation units

    # Step 2: Estimate scaling factor for realistic currents
    # Assume BP detector responsivity ~ 1 A/W in mid-IR as baseline
    # and that incident power is ~ 1 µW (typical for lab mid-IR setups)
    target_rms_current_phys = 1e-6  # 1 µA RMS
    median_rms_current_sim = np.median(rms_current_sim)
    scale_factor = target_rms_current_phys / median_rms_current_sim

    # Step 3: Compute physical currents after scaling
    I_phys = I * scale_factor
    rms_current_phys = np.sqrt(np.mean(I_phys**2, axis=1))

    # Step 4: Shot noise std (A) = sqrt(2 * q * I * bandwidth)
    shot_noise_std_phys = np.sqrt(2 * e * np.abs(rms_current_phys) * bandwidth)

    # Step 5: Map shot noise back to simulation units
    shot_noise_std_sim = shot_noise_std_phys / scale_factor

    # Step 6: Print results
    print("\n=== Physics-Based Noise Estimation ===")
    print(f"Median RMS current (sim units): {median_rms_current_sim:.3e}")
    print(f"Median RMS current (phys): {np.median(rms_current_phys):.3e} A")
    print(f"Scale factor applied to match physical currents: {scale_factor:.3e}")

    print(f"\nShot noise std (phys units): {np.median(shot_noise_std_phys):.3e} A")
    print(f"Shot noise std (sim units): {np.median(shot_noise_std_sim):.3e}")

    frac_noise_phys = np.median(shot_noise_std_phys / rms_current_phys) * 100
    frac_noise_sim = np.median(shot_noise_std_sim / rms_current_sim) * 100
    print(f"Estimated fractional noise (phys): {frac_noise_phys:.4f} %")
    print(f"Estimated fractional noise (sim): {frac_noise_sim:.4f} %")

    snr_phys = np.median(rms_current_phys / shot_noise_std_phys)
    snr_sim = np.median(rms_current_sim / shot_noise_std_sim)
    print(f"Estimated SNR (phys): {snr_phys:.2f}")
    print(f"Estimated SNR (sim): {snr_sim:.2f}")

if __name__ == "__main__":
    main()
