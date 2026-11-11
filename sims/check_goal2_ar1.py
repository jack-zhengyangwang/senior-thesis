# Goal 2 feasibility: returns correlated across days with a KNOWN correlation matrix A (AR(1), phi).
# Model: X = A^{1/2} [x0 | stocks]: market column x0 ~ N(0, s00*A); x_i = b_i x0 + e_i, e ~ N(0, A (x) Psi).
# Checks: (1) E S = Sigma; (2) Var(s_ij) = (s_ij^2 + s_ii s_jj) tr(A^2)/T^2 and the variance estimator
# with effective sample size T_eff = T^2/tr(A^2); (3) does Cov(f_ij, s_ij) = Var(f_ij) still hold?
import numpy as np
rng = np.random.default_rng(3)
import sys; T, R, phi = int(sys.argv[1]), 400000, float(sys.argv[2])
b = np.array([1.2, 0.8]); s00 = 1.5
Psi = np.array([[0.6, 0.15], [0.15, 0.9]])
Sig = s00*np.outer(b, b) + Psi
A = phi**np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
LA = np.linalg.cholesky(A); LP = np.linalg.cholesky(Psi)
Teff = T**2/np.trace(A@A)
x0 = (rng.standard_normal((R, T)) @ LA.T) * np.sqrt(s00)
E = np.einsum('ts,rsk->rtk', LA, rng.standard_normal((R, T, 2)) @ LP.T)
X = x0[:, :, None]*b + E
S = np.einsum('rti,rtj->rij', X, X)/T
c = (x0**2).mean(1); s0 = np.einsum('rt,rti->ri', x0, X)/T
f = s0[:, 0]*s0[:, 1]/c; s12 = S[:, 0, 1]
print(f"T={T}, phi={phi}, T_eff={Teff:.1f}")
print("(1) E S - Sigma (max abs):", np.abs(S.mean(0)-Sig).max().round(4))
v_formula = (Sig[0,1]**2 + Sig[0,0]*Sig[1,1])*np.trace(A@A)/T**2
m = Teff
v_hat = ((m-2)*s12**2 + m*S[:,0,0]*S[:,1,1])/((m+2)*(m-1))
v_iid = ((T-2)*s12**2 + T*S[:,0,0]*S[:,1,1])/((T+2)*(T-1))
print(f"(2) Var(s12): MC {s12.var():.5f} | formula {v_formula:.5f} | E[est, T_eff] {v_hat.mean():.5f} | E[est, iid T] {v_iid.mean():.5f}")
print(f"(3) Cov(f,s) {np.mean(f*(s12-Sig[0,1])):.5f} | Var(f) {f.var():.5f} | Cov(f, s-f) {np.cov(f, s12-f)[0,1]:.5f}")
