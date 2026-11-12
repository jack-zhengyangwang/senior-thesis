"""Goal 2, step 5: exact SURE for the single-index target when days are serially correlated.

Model: x0 ~ N(0, s00*A) (T-vector); x_i = b_i x0 + e_i; Cov(e_si, e_tj) = Psi_ij A_st; e indep. of x0;
A known, unit diagonal. Known zero mean. A = I is the i.i.d. case.

Notation given x0: h = x0/|x0|, H = hh', M = I-H, n0 = x0'x0,
  y_i = h'x_i,   R_ij = x_i' M x_j,   so  f_ij = y_i y_j / T,  s_ij = (y_i y_j + R_ij)/T.
  q = h'Ah, q2 = h'A^2 h, w = q2 - q^2, m1 = tr(MA) = T - q, m2 = tr(MAMA) = tr(A^2) - 2 q2 + q^2.
Estimator:
  Cov-hat(f_ij, s_ij) = f_ij s_ij - [ betaSq + (1 + k1/T) betaPsi + (k1/T) u ],  k1 = E[q] (depends on A only)
  Var-hat(s_ij) = a s_ij^2 + b s_ii s_jj  with  tau = tr(A^2)/T^2.
Usage: python3 sims/check_goal2_sure.py [ar1 phi | overlap k] T reps
"""
import sys
import numpy as np
from scipy.integrate import quad


def make_A(kind, par, T):
    t = np.arange(T)
    if kind == "ar1":
        return float(par) ** np.abs(np.subtract.outer(t, t))
    k = int(par)  # overlapping k-period sums of iid daily returns
    return np.maximum(k - np.abs(np.subtract.outer(t, t)), 0) / k


def k1_exact(A):
    """E[x0'A x0 / x0'x0] for x0 ~ N(0, A): one-dimensional integral (Magnus 1986)."""
    lam = np.linalg.eigvalsh(A)
    g = lambda t: np.exp(-0.5 * np.log1p(2 * t * lam).sum()) * (lam**2 / (1 + 2 * t * lam)).sum()
    return quad(g, 0, np.inf, limit=500)[0]


def var_s_hat(S, tau):
    d = np.diag(S)
    a = tau * (1 - 2 * tau) / ((1 - tau) * (1 + 2 * tau))
    b = tau / ((1 - tau) * (1 + 2 * tau))
    return a * S**2 + b * np.outer(d, d)


def cov_fs_hat(x0, X, A, k1):
    """Unbiased estimate of Cov(f_ij, s_ij), all i,j (use off-diagonal)."""
    T = len(x0)
    n0 = x0 @ x0
    h = x0 / np.sqrt(n0)
    y = X.T @ h                                   # y_i
    Xm = X - np.outer(h, y)                       # M X
    R = Xm.T @ Xm                                 # R_ij
    Ah = A @ h
    q, q2 = h @ Ah, Ah @ Ah
    w = q2 - q**2
    trA2 = np.sum(A * A)
    m1, m2 = T - q, trA2 - 2 * q2 + q**2
    D = (m1**2 - m2) * (m1**2 + 2 * m2)
    r = np.diag(R)
    RR = np.outer(r, r)
    u = (m1**2 * R**2 - m2 * RR) / D              # -> Psi_ij^2
    v = ((m1**2 + m2) * RR - 2 * m2 * R**2) / D   # -> Psi_ii Psi_jj
    yy = np.outer(y, y)
    G1 = (yy * R - q * m1 * u - w * (u + v)) / m1                  # -> mu_i mu_j Psi_ij
    G2 = (np.outer(y**2, r) - q * m1 * v - 2 * w * u) / m1         # -> mu_i^2 Psi_jj
    mu4 = yy**2 - q * (G2 + G2.T + 4 * G1) - q**2 * (v + 2 * u)    # -> mu_i^2 mu_j^2
    betaSq = mu4 / (T**2 + 2 * trA2)              # -> s00^2 b_i^2 b_j^2
    betaPsi = G1 / T                              # -> s00 b_i b_j Psi_ij
    f = yy / T
    S = (yy + R) / T
    U = betaSq + (1 + k1 / T) * betaPsi + (k1 / T) * u   # -> E[f_ij] * sigma_ij
    return f * S - U, f, S


if __name__ == "__main__":
    kind, par, T, reps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    rng = np.random.default_rng(11)
    N = 8
    A = make_A(kind, par, T)
    LA = np.linalg.cholesky(A + 1e-12 * np.eye(T))
    tau = np.sum(A * A) / T**2
    k1 = k1_exact(A)
    s00 = 1.0
    b = rng.uniform(0.5, 1.5, N)
    d = rng.uniform(0.5, 1.5, N)
    C = np.where(np.arange(N)[:, None] // 4 == np.arange(N)[None, :] // 4, 0.3, 0.0); np.fill_diagonal(C, 1)
    Psi = np.sqrt(d)[:, None] * C * np.sqrt(d)[None, :]
    Sig = s00 * np.outer(b, b) + Psi
    LP = np.linalg.cholesky(Psi)
    off = ~np.eye(N, dtype=bool)
    alphas = np.linspace(0, 1, 11)
    sure_sum = np.zeros_like(alphas); loss_sum = np.zeros_like(alphas); dsq = np.zeros_like(alphas)
    cov_hat_sum = np.zeros((N, N)); fs_sum = np.zeros((N, N)); f_sum = np.zeros((N, N))
    qs = []
    iid_sum = np.zeros_like(alphas)
    for _ in range(reps):
        x0 = LA @ rng.standard_normal(T) * np.sqrt(s00)
        X = np.outer(x0, b) + LA @ rng.standard_normal((T, N)) @ LP.T
        Chat, f, S = cov_fs_hat(x0, X, A, k1)
        F = f.copy(); np.fill_diagonal(F, np.diag(S))
        Vs = var_s_hat(S, tau)
        V = Vs.sum()
        Cv = np.trace(Vs) + Chat[off].sum()
        D2 = ((F - S) ** 2).sum()
        sure = alphas**2 * D2 + (1 - 2 * alphas) * V + 2 * alphas * Cv
        loss = np.array([((a * F + (1 - a) * S - Sig) ** 2).sum() for a in alphas])
        sure_sum += sure; loss_sum += loss; dsq += (sure - loss) ** 2
        cov_hat_sum += Chat; fs_sum += f * (S - Sig); f_sum += f
        qs.append(x0 @ A @ x0 / (x0 @ x0))
    ES, Rk = sure_sum / reps, loss_sum / reps
    se = np.sqrt((dsq / reps - (ES - Rk) ** 2) / reps)
    print(f"A={kind}({par}), T={T}, N={N}, reps={reps}; tr(A^2)/T = {tau*T:.3f}; k1 exact = {k1:.4f}, MC mean q = {np.mean(qs):.4f}")
    print(f"sum_(i!=j) Cov-hat = {cov_hat_sum[off].sum()/reps:.4f}   vs MC sum Cov(f,s) = {fs_sum[off].sum()/reps:.4f}")
    print(" alpha  E[SURE]   R(alpha)   diff  (2 s.e.)")
    for i, a in enumerate(alphas):
        print(f" {a:.1f}  {ES[i]:8.3f}  {Rk[i]:8.3f}  {ES[i]-Rk[i]:+.3f} ({2*se[i]:.3f})")
