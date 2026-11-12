"""Push Stein's lemma directly on the single-index target (Section 4, Jack's route).
Day vector z_t = (x_t0, x_t1, ..., x_tN) iid N(0, Sig_full); s_ab = (1/T) sum_t x_ta x_tb; f_ij = s_0i s_0j / s_00.
Stein's lemma gives (derived by hand):
  Var(s_ij)      = (1/T) E[ sig_ii s_jj + sig_ij s_ij ]
  Cov(f_ij,s_ij) = (1/T) E[ sig_i0 (s_ij s_0j/s_00 + s_0i s_jj/s_00 - 2 s_0i s_0j^2/s_00^2)
                            + sig_ii s_0j^2/s_00 + sig_ij s_0i s_0j/s_00 ]
Checks both against brute-force Monte Carlo."""
import numpy as np
rng = np.random.default_rng(3)
T, R = 12, 400000
b = np.array([1.0, 1.2, 0.8]); s00 = 1.5
Psi = np.diag([0.0, 0.6, 0.9]); Psi[1, 2] = Psi[2, 1] = 0.15   # index 0 = market (no residual)
Sig = s00 * np.outer(b, b) + Psi                                # (N+1)x(N+1), market first
L = np.linalg.cholesky(Sig + 1e-12*np.eye(3))
Z = rng.standard_normal((R, T, 3)) @ L.T
S = np.einsum('rta,rtb->rab', Z, Z) / T
i, j = 1, 2
f = S[:, 0, i] * S[:, 0, j] / S[:, 0, 0]
cov_mc = np.mean(f * (S[:, i, j] - Sig[i, j])); se = np.std(f * (S[:, i, j] - Sig[i, j])) / np.sqrt(R)
g = (Sig[i, 0] * (S[:, i, j]*S[:, 0, j]/S[:, 0, 0] + S[:, 0, i]*S[:, j, j]/S[:, 0, 0] - 2*S[:, 0, i]*S[:, 0, j]**2/S[:, 0, 0]**2)
     + Sig[i, i]*S[:, 0, j]**2/S[:, 0, 0] + Sig[i, j]*S[:, 0, i]*S[:, 0, j]/S[:, 0, 0]) / T
var_mc = np.var(S[:, i, j]); var_st = np.mean((Sig[i, i]*S[:, j, j] + Sig[i, j]*S[:, i, j]) / T)
print(f"Var(s_ij): MC {var_mc:.5f}  Stein {var_st:.5f}  closed form {(Sig[i,i]*Sig[j,j]+Sig[i,j]**2)/T:.5f}")
print(f"Cov(f_ij,s_ij): MC {cov_mc:.5f} (+/- {2*se:.5f})  Stein {g.mean():.5f}")
