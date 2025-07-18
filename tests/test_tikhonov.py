# tests/test_tikhonov_responsivity.py

import os
import numpy as np
import pytest

from src.models.tikhonov import TikhonovInverter

@pytest.fixture(autouse=True)
def rng_seed():
    np.random.seed(0)
    yield

@pytest.fixture(scope="module")
def A():
    # load and transpose your responsivity matrix
    R = np.load(os.path.join("data", "responsivity_data" "processed", "responsivity.npy"))  # (1000,41)
    return R.T                                                          # (41,1000)

def test_zero_alpha_minimum_norm_solution(A):
    """
    With alpha=0, solver.solve(b) should match the minimum‐norm LS solution:
      x_pinv = pinv(A) @ b
    """
    m, n = A.shape  # m=41, n=1000
    x_true = np.random.randn(n)
    b = A @ x_true

    solver = TikhonovInverter(alpha=0.0)
    solver.set_matrix(A)
    x_est = solver.solve(b)

    x_pinv = np.linalg.pinv(A) @ b
    assert np.allclose(x_est, x_pinv, atol=1e-6), \
        "Zero‑α solve must match pseudoinverse solution"

def test_nonzero_alpha_within_tolerance(A):
    """
    With alpha>0 and noise, we check relative error and MSE are below thresholds.
    """
    alpha = 0.1
    m, n = A.shape
    x_true = np.random.randn(n)
    b = A @ x_true + 1e-2 * np.random.randn(m)

    solver = TikhonovInverter(alpha=alpha)
    solver.set_matrix(A)
    x_est = solver.solve(b)

    # relative error and MSE
    rel_err = np.linalg.norm(x_est - x_true) / np.linalg.norm(x_true)
    mse     = np.mean((x_est - x_true)**2)

    assert rel_err < 0.2, f"Relative error too high: {rel_err:.3f}"
    assert mse     < 0.05, f"MSE too high: {mse:.4f}"

