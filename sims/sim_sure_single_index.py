"""End-to-end check of the exact SURE for Ledoit-Wolf shrinkage toward the single-index target.

Model (assumptions of Theorem 1):
  market x_0t ~ N(0, sigma00) iid; stocks x_it = b_i x_0t + e_it, e_t ~ N(0, Psi) iid, indep. of market;
  known zero mean; market is an outside series (not built from the N stocks).

SURE(a) = a^2 ||F - S||^2 + (1 - 2a) sum_ij Varhat(s_ij)
          + 2a [ sum_i Varhat(s_ii) + sum_{i != j} Varhat(f_ij) ]
Checks E[SURE(a)] = R(a) = E||a F + (1-a) S - Sigma||^2 on a grid of a, and plots performance curves.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

N, T, R = 20, 40, 60000
sigma00 = 1.0
b = rng.uniform(0.5, 1.5, N)
# residual covariance with some cross-correlation (so the single-index target is misspecified)
d = rng.uniform(0.5, 1.5, N)
blocks = np.repeat(np.arange(4), N // 4)
C = np.where(blocks[:, None] == blocks[None, :], 0.3, 0.0); np.fill_diagonal(C, 1.0)
Psi = np.sqrt(d)[:, None] * C * np.sqrt(d)[None, :]
Sigma = sigma00 * np.outer(b, b) + Psi
Lp = np.linalg.cholesky(Psi)
alphas = np.linspace(0, 1, 41)
m = T - 1
off = ~np.eye(N, dtype=bool)

sure_sum = np.zeros_like(alphas); loss_sum = np.zeros_like(alphas); diff_sq = np.zeros_like(alphas)
keep_sure, a_sure, a_oracle = [], [], []
for r in range(R):
    x0 = rng.standard_normal(T) * np.sqrt(sigma00)
    X = np.outer(x0, b) + rng.standard_normal((T, N)) @ Lp.T
    c = x0 @ x0 / T                       # market variance s00
    s0 = X.T @ x0 / T                     # s_0i
    S = X.T @ X / T
    F = np.outer(s0, s0) / c; np.fill_diagonal(F, np.diag(S))
    # --- variance term (all i,j); its diagonal is also the diagonal covariance term
    sd = np.diag(S)
    var_s = ((T - 2) * S**2 + T * np.outer(sd, sd)) / ((T + 2) * (T - 1))
    # --- Var(f_ij), i != j
    bh = s0 / c
    P = (S - np.outer(s0, s0) / c) * T / m          # unbiased residual covariance
    p = np.diag(P); bb2 = bh**2
    Bh  = (m * (m - 2) * P**2 + m**2 * np.outer(p, p)) / ((m + 2) * (m - 1))
    Bh2 = (2 * m * P**2 + m * np.outer(p, p)) / (m + 2)
    Ah  = np.outer(bb2, p) + np.outer(p, bb2) + 2 * np.outer(bh, bh) * P
    Ah2 = np.outer(bb2, p) + np.outer(p, bb2) + 4 * np.outer(bh, bh) * P
    est_sA  = c * Ah - 2 * Bh / T
    est_s2b = (c**2 * np.outer(bb2, bb2) - c * Ah2 / T + Bh2 / T**2) / (1 + 2 / T)
    var_f = est_sA / T + 2 * est_s2b / T + Bh / T**2
    V = var_s.sum()
    Cv = np.trace(var_s) + var_f[off].sum()          # estimate of sum_ij Cov(f_ij, s_ij)
    D2 = ((F - S)**2).sum()
    sure = alphas**2 * D2 + (1 - 2 * alphas) * V + 2 * alphas * Cv
    loss = np.array([((a * F + (1 - a) * S - Sigma)**2).sum() for a in alphas])
    sure_sum += sure; loss_sum += loss; diff_sq += (sure - loss)**2
    if r < 12: keep_sure.append(sure)
    a_sure.append(np.clip((V - Cv) / D2, 0, 1))
    a_oracle.append(alphas[np.argmin(loss)])

ES, Rk = sure_sum / R, loss_sum / R
print(f"N={N}, T={T}, reps={R}")
se = np.sqrt((diff_sq / R - (ES - Rk)**2) / R)
print(" alpha   E[SURE]    R(alpha)   diff   (2 s.e.)")
for i in range(0, len(alphas), 8):
    print(f" {alphas[i]:.2f}  {ES[i]:9.3f}  {Rk[i]:9.3f}  {ES[i]-Rk[i]:+.3f} ({2*se[i]:.3f})")
a_star = alphas[np.argmin(Rk)]
print(f"risk-minimizing alpha (true curve) = {a_star:.3f}; mean SURE-chosen alpha = {np.mean(a_sure):.3f} "
      f"(sd {np.std(a_sure):.3f}); mean per-sample oracle alpha = {np.mean(a_oracle):.3f}")

# --- figure: performance curves
ink, muted, blue = "#0b0b0b", "#52514e", "#2a78d6"
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
ax = axes[0]
ax.plot(alphas, Rk, color=ink, lw=2, label="True risk R(α)")
ax.plot(alphas, ES, color=blue, lw=2, ls="--", label="Average SURE(α)")
ax.set_title("Average SURE matches the true risk", loc="left", fontsize=11)
ax.legend(frameon=False)
ax = axes[1]
for s in keep_sure:
    ax.plot(alphas, s, color=blue, lw=1, alpha=0.35)
ax.plot(alphas, Rk, color=ink, lw=2, label="True risk R(α)")
ax.plot([], [], color=blue, lw=1, alpha=0.6, label="SURE(α), single samples")
ax.set_title("Individual SURE curves", loc="left", fontsize=11)
ax.legend(frameon=False)
for ax in axes:
    ax.set_xlabel("Shrinkage intensity α")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(colors=muted)
axes[0].set_ylabel("Frobenius risk")
fig.suptitle(f"Exact SURE, single-index target (N={N}, T={T}, {R:,} samples)", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "sure_single_index_curves.png", dpi=160)
fig.savefig(OUT / "sure_single_index_curves.pdf")
print("saved", OUT / "sure_single_index_curves.png")
