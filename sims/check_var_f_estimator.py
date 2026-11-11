# Unbiased estimator of Var(f_ij), f_ij = s_0i s_0j / s_00 (i != j), single-index target.
# Model: x_0t ~ N(0, s00) iid; x_it = b_i x_0t + e_it, e_t ~ N(0, Psi) iid, indep of x_0. Known zero mean.
import numpy as np
rng = np.random.default_rng(2)
b = np.array([1.2, 0.8]); s00 = 1.5
Psi = np.array([[0.6, 0.15], [0.15, 0.9]])
L = np.linalg.cholesky(Psi)
for T in (10, 20, 60):
    R = 400000; m = T - 1
    x0 = rng.standard_normal((R, T)) * np.sqrt(s00)
    x = x0[:, :, None] * b + rng.standard_normal((R, T, 2)) @ L.T
    c = (x0**2).mean(1)                                   # s_00
    s0 = np.einsum('rt,rti->ri', x0, x) / T               # s_0i
    S = np.einsum('rti,rtj->rij', x, x) / T
    bh = s0 / c[:, None]                                  # beta-hat
    Pt = (S - s0[:, :, None]*s0[:, None, :]/c[:, None, None]) * T/m   # unbiased residual cov
    f = s0[:, 0]*s0[:, 1]/c
    p11, p22, p12 = Pt[:, 0, 0], Pt[:, 1, 1], Pt[:, 0, 1]
    Bh  = (m*(m-2)*p12**2 + m**2*p11*p22) / ((m+2)*(m-1))   # unbiased for Psi_ii Psi_jj + Psi_ij^2
    Bh2 = (2*m*p12**2 + m*p11*p22) / (m+2)                  # unbiased for Psi_ii Psi_jj + 2 Psi_ij^2
    Ah  = bh[:,0]**2*p22 + bh[:,1]**2*p11 + 2*bh[:,0]*bh[:,1]*p12
    Ah2 = bh[:,0]**2*p22 + bh[:,1]**2*p11 + 4*bh[:,0]*bh[:,1]*p12
    est_sA  = c*Ah - 2*Bh/T                                          # -> s00 * A
    est_s2b = (c**2*bh[:,0]**2*bh[:,1]**2 - c*Ah2/T + Bh2/T**2) / (1 + 2/T)  # -> s00^2 b_i^2 b_j^2
    Vhat = est_sA/T + 2*est_s2b/T + Bh/T**2
    A = b[0]**2*Psi[1,1] + b[1]**2*Psi[0,0] + 2*b[0]*b[1]*Psi[0,1]
    B = Psi[0,0]*Psi[1,1] + Psi[0,1]**2
    Vtrue = (s00*A + 2*b[0]**2*b[1]**2*s00**2)/T + B/T**2
    se = Vhat.std()/np.sqrt(R)
    print(f"T={T:3d}: formula Var(f)={Vtrue:.5f}  MC Var(f)={f.var():.5f}  E[Vhat]={Vhat.mean():.5f} (+/-{2*se:.5f})")
