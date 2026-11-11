# Check: under joint normality (known zero mean), for the single-index target
# f_ij = s_0i s_0j / s_00 (i != j), is Cov(f_ij, s_ij) = Var(f_ij)?
import numpy as np
rng = np.random.default_rng(1)
T, R = 20, 400000
b = np.array([1.2, 0.8]); s00 = 1.5
Psi = np.array([[0.6, 0.15], [0.15, 0.9]])          # residual covariance (correlated residuals)
x0 = rng.standard_normal((R, T)) * np.sqrt(s00)
e = rng.standard_normal((R, T, 2)) @ np.linalg.cholesky(Psi).T
x = x0[:, :, None] * b + e
S00 = (x0**2).mean(1); S0 = np.einsum('rt,rti->ri', x0, x) / T
s12 = np.einsum('rt,rt->r', x[:, :, 0], x[:, :, 1]) / T
f12 = S0[:, 0] * S0[:, 1] / S00
sig12 = b[0]*b[1]*s00 + Psi[0, 1]
print(f"T={T}: Cov(f,s)={np.mean(f12*(s12-sig12)):.5f}  Var(f)={f12.var():.5f}  Cov(f, s-f)={np.cov(f12, s12-f12)[0,1]:.5f}")
