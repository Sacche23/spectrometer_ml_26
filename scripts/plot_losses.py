import numpy as np
import matplotlib.pyplot as plt

# Data from logs/42736319.out
res = np.array([1, 2, 4, 10, 20, 25, 40], dtype=float)
m_model = np.array([1.9912, 1.9912, 1.9912, 1.9912, 1.9912, 1.9912, 1.9912])
m_tikh  = np.array([2.7652, 2.6518, 2.5733, 2.9706, 3.8158, 4.3922, 6.1382])
m_lasso = np.array([12.9030, 11.5330, 9.9733, 5.1602, 5.1999, 5.6316, 6.7615])

plt.figure()
plt.plot(res, m_model, marker='o', label='Model')
plt.plot(res, m_tikh,  marker='o', label='Tikhonov')
plt.plot(res, m_lasso, marker='o', label='Lasso')
plt.xscale('log')
plt.xlabel('Resolution (×)')
plt.ylabel('Average MSE')
plt.title('MSE vs Resolution')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig()
