import numpy as np
from scipy.linalg import cholesky, solve_triangular

class TikhonovInverter:
    def __init__(self, alpha: float = 1.0):
        '''
        alpha: L2 penalty weight
        '''
        self.alpha = alpha
        self.R = None # Forward matrix, shape (m, n)
        self.B = None # Upper‑triangular Cholesky factor of (A^T * A + alpha * I)

    def set_matrix(self, R: np.ndarray):
        '''
        Stores R and Choelsky factorization G = B^T * B for G = (A^T * A + alpha * Id)
        Allows for O(n^2) inference after computation
        R: responsivity matrix
        '''
        self.R = R
        m, n = R.shape
        # Form G = A^T A + alpha * I
        G = R.T @ R + self.alpha * np.eye(n)
        # Compute upper‑triangular R such that G = R^T @ R
        self.B = cholesky(G, lower=False)

    def solve(self, I: np.ndarray) -> np.ndarray:
        '''
        Minimize [ ||R S - I||^2 + alpha * ||S||^2 ] over vectors S in three steps:
            1.) c = A^T * I
            2.) Solve B^T * z = c --> z = (R^T)^-1 * c
            3.) Solve B^T * S = z --> S = R^-1 (R^T)^-1 c = [(A^T * A + alpha * Id)^-1 * A^T * I]
        A: (m, n), I: (m,)
        Returns S: (n,)
        '''
        if self.R is None or self.B is None:
            raise RuntimeError("Call set_matrix(R) before solve(y).")

        Atb = self.R.T @ I # compute A^T y (O(m·n))
        z = solve_triangular(self.B.T, Atb, lower=True) # solve R^T * z = A^T * y (O(n^2))
        S = solve_triangular(self.B, z, lower=False) # solve R * x = z (O(n^2))
        return S

    def update_alpha(self, alpha: float):
        '''
        If alpha is changed, Choelsky factorization must be updated!
        '''
        self.alpha = alpha
        if self.R is not None:
            self.set_matrix(self.R)