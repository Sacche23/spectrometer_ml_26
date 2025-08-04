import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def load_spectrum_values(csv_file, index):
    df = pd.read_csv(csv_file)
    if df.shape[0] < 2:
        raise ValueError("CSV must have at least one metadata row and one data row.")
    data = df.iloc[1:].reset_index(drop=True)
    if index < 0 or index >= data.shape[0]:
        raise IndexError(f"Spectrum index {index} out of range (0 to {data.shape[0]-1}).")
    return data.iloc[index].values.astype(float)


def load_responsivity(mat_file):
    df = pd.read_csv(mat_file, header=None)
    return df.iloc[1:, 1:].values.astype(float)


def build_design_matrix(spectra_csv, resp_csv, index, low_cutoff, high_cutoff, skip):
    # load spectrum and responsivity
    spectrum = load_spectrum_values(spectra_csv, index)
    R_raw = load_responsivity(resp_csv)
    # build wavelength grid
    N = spectrum.size
    wavelengths = np.linspace(1.0, 9.5, N)
    # drop first bin
    wavelengths = wavelengths[1:]
    spectrum = spectrum[1:]
    R = R_raw[1:, :]
    # apply low-cutoff
    if low_cutoff is not None:
        mask = wavelengths >= low_cutoff
        wavelengths = wavelengths[mask]
        spectrum = spectrum[mask]
        R = R[mask, :]
    # apply high-cutoff
    if high_cutoff is not None:
        mask = wavelengths <= high_cutoff
        wavelengths = wavelengths[mask]
        spectrum = spectrum[mask]
        R = R[mask, :]
    # drop near-zero responsivity
    A_full = R.T
    norms = np.linalg.norm(A_full, axis=0)
    keep = norms > 1e-6
    A = A_full[:, keep]
    spectrum = spectrum[keep]
    # downsample if skip
    if skip > 0:
        idx = np.arange(0, A.shape[1], skip+1)
        A = A[:, idx]
        spectrum = spectrum[idx]
    # simulate currents
    I_obs = A.dot(spectrum)
    return A, I_obs


def main():
    parser = argparse.ArgumentParser(description="Cross-validate Lasso alpha for spectrum reconstruction.")
    parser.add_argument('--spectra_csv', required=True, help='CSV of ground-truth spectra')
    parser.add_argument('--resp_csv',    required=True, help='CSV of responsivity matrix')
    parser.add_argument('--index', type=int, required=True, help='0-based index of spectrum row')
    parser.add_argument('--low-cutoff', type=float, default=None, help='Low-end cutoff (µm)')
    parser.add_argument('--high-cutoff',type=float, default=None, help='High-end cutoff (µm)')
    parser.add_argument('--skip', type=int, default=0, help='Downsample: skip every n points')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--alphas', type=str,
                        default='1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1',
                        help='Comma-separated alpha grid (e.g. "1e-4,1e-3,1e-2")')
    args = parser.parse_args()

    # parse alphas
    alpha_list = [float(a) for a in args.alphas.split(',')]

    # build design matrix and observations
    A, y = build_design_matrix(
        args.spectra_csv, args.resp_csv, args.index,
        args.low_cutoff, args.high_cutoff, args.skip)

    # K-fold CV
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=0)
    mse_scores = {alpha: [] for alpha in alpha_list}

    for train_idx, test_idx in kf.split(A):
        A_train, A_test = A[train_idx], A[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        for alpha in alpha_list:
            model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
            model.fit(A_train, y_train)
            y_pred = model.predict(A_test)
            mse = mean_squared_error(y_test, y_pred)
            mse_scores[alpha].append(mse)

    # compute average MSE
    avg_mse = {alpha: np.mean(mse_scores[alpha]) for alpha in alpha_list}
    best_alpha = min(avg_mse, key=avg_mse.get)

    # print results
    print("Alpha\tAverage MSE")
    for alpha in alpha_list:
        print(f"{alpha}\t{avg_mse[alpha]:.6e}")
    print(f"\nBest alpha: {best_alpha}")

if __name__ == '__main__':
    main()
