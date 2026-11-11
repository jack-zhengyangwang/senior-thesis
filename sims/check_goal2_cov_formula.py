# Goal 2: closed form for Cov(f_ij, s_ij) when days are correlated with known A (shared by all series).
# Derived (conditioning on the market series x0, NO independence across days):
#   Cov(f,s) = [s00*Ac + 2 b_i^2 b_j^2 s00^2] tr(A^2)/T^2 + B*E[q2]/T^2 + b_i b_j Psi_ij Cov(c,q)/T
#   Ac = b_i^2 Psi_jj + b_j^2 Psi_ii + 2 b_i b_j Psi_ij,  B = Psi_ii Psi_jj + Psi_ij^2,
#   c = x0'x0/T,  q = x0'A x0 / x0'x0,  q2 = x0'A^2 x0 / x0'x0  (q, q2 depend only on A and the market).
import numpy as np, sys
rng = np.random.default_rng(4)
T, phi = int(sys.argv[1]), float(sys.argv[2]); R = 400000
b = np.array([1.2, 0.8]); s00 = 1.5
Psi = np.array([[0.6, 0.15], [0.15, 0.9]]); Sig = s00*np.outer(b, b) + Psi
A = phi**np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
LA = np.linalg.cholesky(A); LP = np.linalg.cholesky(Psi); A2 = A @ A
x0 = (rng.standard_normal((R, T)) @ LA.T) * np.sqrt(s00)
E = np.einsum('ts,rsk->rtk', LA, rng.standard_normal((R, T, 2)) @ LP.T)
X = x0[:, :, None]*b + E
s12 = np.einsum('rt,rt->r', X[:, :, 0], X[:, :, 1])/T
c = (x0**2).mean(1); s0 = np.einsum('rt,rti->ri', x0, X)/T
f = s0[:, 0]*s0[:, 1]/c
xx = (x0**2).sum(1)
q  = np.einsum('rt,ts,rs->r', x0, A, x0)/xx
q2 = np.einsum('rt,ts,rs->r', x0, A2, x0)/xx
Ac = b[0]**2*Psi[1,1] + b[1]**2*Psi[0,0] + 2*b[0]*b[1]*Psi[0,1]
B = Psi[0,0]*Psi[1,1] + Psi[0,1]**2
trA2 = np.trace(A2)
formula = (s00*Ac + 2*b[0]**2*b[1]**2*s00**2)*trA2/T**2 + B*q2.mean()/T**2 + b[0]*b[1]*Psi[0,1]*np.cov(c, q)[0,1]/T
mc = np.mean(f*(s12 - Sig[0,1])); se = (f*(s12-Sig[0,1])).std()/np.sqrt(R)
iid = (s00*Ac + 2*b[0]**2*b[1]**2*s00**2)/T + B/T**2
print(f"T={T} phi={phi}: MC Cov(f,s)={mc:.5f} (+/-{2*se:.5f})  formula={formula:.5f}  [iid formula would give {iid:.5f}]")
