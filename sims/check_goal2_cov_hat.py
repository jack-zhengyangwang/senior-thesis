"""Sharp test of the Goal 2 covariance estimator: E[f*sigma - U] should be 0 for every pair (i != j).
Also reports how badly the i.i.d. (Goal 1) estimator does under serial correlation.
Usage: python3 sims/check_goal2_cov_hat.py [ar1 phi | overlap k] T reps"""
import sys
import numpy as np
from check_goal2_sure import make_A, k1_exact, cov_fs_hat

kind, par, T, reps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(5)
A = make_A(kind, par, T); LA = np.linalg.cholesky(A + 1e-12*np.eye(T)); k1 = k1_exact(A)
b = np.array([1.2, 0.8, 1.0]); s00 = 1.5
Psi = np.array([[0.6, 0.15, 0.1], [0.15, 0.9, -0.2], [0.1, -0.2, 0.7]])
Sig = s00*np.outer(b, b) + Psi; LP = np.linalg.cholesky(Psi)
dev = []; tgt = []
for _ in range(reps):
    x0 = LA @ rng.standard_normal(T) * np.sqrt(s00)
    X = np.outer(x0, b) + LA @ rng.standard_normal((T, 3)) @ LP.T
    Chat, f, S = cov_fs_hat(x0, X, A, k1)
    dev.append([(f*Sig - (f*S - Chat))[i, j] for i, j in [(0, 1), (0, 2), (1, 2)]])
    tgt.append([(f*(S - Sig))[i, j] for i, j in [(0, 1), (0, 2), (1, 2)]])
dev, tgt = np.array(dev), np.array(tgt)
print(f"A={kind}({par}) T={T} reps={reps}")
for k, p in enumerate(["(1,2)", "(1,3)", "(2,3)"]):
    print(f"  pair {p}: true Cov(f,s) ~ {tgt[:,k].mean():.4f};  bias of Cov-hat = {dev[:,k].mean():+.5f} (2 s.e. {2*dev[:,k].std()/np.sqrt(reps):.5f})")
