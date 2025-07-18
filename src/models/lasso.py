import numpy as np
from sklearn.linear_model import Lasso

class LassoInverter:
    def __init__(self, alpha: float=1.0, warm_start: bool=True, precompute: bool=True, max_iter: int=1000):
        '''
        alpha: L1 penalty weight
        warm_start: reuse previous solution as init-point
        precompute: cache R^T R for faster solves
        '''
        self.alpha = alpha
        self.model = Lasso(alpha=self.alpha, warm_start=warm_start, precompute=precompute, max_iter=max_iter)

    def solve(self, R: np.ndarray, I: np.ndarray):
        '''
        Minimize [ ||R S - I||^2 + alpha * ||S||_1 ] over vectors S
        R: (m, n), I: (m,)
        Returns S: (n,)
        '''
        self.model.fit(R, I)
        return self.model.coef_.copy()
