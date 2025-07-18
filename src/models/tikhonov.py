import numpy as np
from scipy.linalg import cholesky, solve_triangular

class TikhonovInverter:
    def __init__(self, alpha: float = 1.0):
        '''
        '''
        self.alpha = alpha
        self.R = None # Forward matrix, shape (m, n)
        self.B = None # Upper‑triangular Cholesky factor of (A^T A + α I)

    def set_matrix(self, R: np.ndarray):
        '''
        '''
        self.R = R
        m, n = R.shape
        # Form G = A^T A + alpha * I
        G = R.T @ R + self.alpha * np.eye(n)
        # Compute upper‑triangular R such that G = R^T @ R
        self.B = cholesky(G, lower=False)

    def solve(self, y: np.ndarray) -> np.ndarray:
        '''
        '''
        if self.R is None or self.B is None:
            raise RuntimeError("Call set_matrix(R) before solve(y).")

        # 1) compute A^T y (O(m·n))
        Atb = self.R.T @ y

        # 2) forward substitution: solve R^T z = A^T y  (O(n^2))
        z = solve_triangular(self.B.T, Atb, lower=True)

        # 3) backward substitution: solve R x = z      (O(n^2))
        x = solve_triangular(self.B, z, lower=False)

        return x

    def update_alpha(self, alpha: float):
        '''
        '''
        self.alpha = alpha
        if self.R is not None:
            self.set_matrix(self.R)