"""Monte Carlo check of the formulas in thesis Section 5 (distribution-free SURE, target nu*I).

Checks, for N=5, T=12, a non-identity Sigma, Gaussian and multivariate t_10 data:
  E[M_hat] = E||x||^4,  E[U2_hat] = tr(Sigma^2),  E[U3_hat] = (tr Sigma)^2
  E[V_hat] = V = sum_ij Var(s_ij),  E[C_hat] = C = sum_ij Cov(f_ij, s_ij)
  E[SURE(alpha)] = R(alpha)  (closed form and direct Monte Carlo of the loss)
  LW intensity (before capping) = (T-1)/T * V_hat / ||F - S||^2   (exact identity)
It also reports the Section 4 Gaussian estimates under both distributions, compares
the Gaussian and distribution-free alpha_hat, and checks the Rao-Blackwell identity
  E[M_hat | S] = T((tr S)^2 + 2 tr(S^2)) / (T + 2)   (Gaussian data)
through E[(M_hat - that) * h(S)] = 0 for several test functions h.

Run: python sims/check_sec5.py   (REPS=4_000_000 by default, about 30 s; env REPS, SEED override)
"""
import os

import numpy as np

N, T = 5, 12
REPS = int(os.environ.get("REPS", 4_000_000))
CHUNK = 50_000
DF = 10
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

rng = np.random.default_rng(int(os.environ.get("SEED", 20261109)))
A = rng.normal(size=(N, N))
Sig = A @ A.T / N + np.diag([0.5, 1.0, 1.5, 2.0, 3.0])
L = np.linalg.cholesky(Sig)
trSig = np.trace(Sig)
trSig2 = np.trace(Sig @ Sig)


def draw(dist, n):
    z = rng.normal(size=(n, T, N)) @ L.T
    if dist == "gauss":
        return z
    w = rng.chisquare(DF, size=(n, T, 1))
    return z * np.sqrt((DF - 2) / w)  # multivariate t_DF rescaled to covariance Sigma


def truth(dist):
    kappa = 1.0 if dist == "gauss" else (DF - 2) / (DF - 4)
    m4 = kappa * (trSig**2 + 2 * trSig2)            # E||x||^4
    V = (m4 - trSig2) / T
    C = (m4 - trSig**2) / (N * T)
    D = trSig2 + V - trSig**2 / N - C               # E||F - S||^2
    R = {a: a**2 * D + (1 - 2 * a) * V + 2 * a * C for a in ALPHAS}
    return dict(M=m4, U2=trSig2, U3=trSig**2, V=V, C=C, D=D, R=R)


def stats(X):
    S = np.einsum("rti,rtj->rij", X, X) / T
    sq = (X**2).sum(-1)                              # ||x_t||^2, shape (r, T)
    M = (sq**2).mean(1)
    G = np.einsum("rti,rui->rtu", X, X)              # Gram matrix x_t' x_u
    off = ~np.eye(T, dtype=bool)
    U2 = (G[:, off] ** 2).sum(1) / (T * (T - 1))
    U3 = (sq[:, :, None] * sq[:, None, :])[:, off].sum(1) / (T * (T - 1))
    trS = np.trace(S, axis1=1, axis2=2)
    trS2 = np.einsum("rij,rji->r", S, S)
    V = (M - U2) / T
    C = (M - U3) / (N * T)
    FS2 = trS2 - trS**2 / N                          # ||F - S||^2
    F = (trS / N)[:, None, None] * np.eye(N)
    loss = {a: ((a * F + (1 - a) * S - Sig) ** 2).sum((1, 2)) for a in ALPHAS}
    sure = {a: a**2 * FS2 + (1 - 2 * a) * V + 2 * a * C for a in ALPHAS}
    # LW (2004) intensity before the cap at 1: b_bar^2 / d^2
    dev = np.einsum("rti,rtj->rtij", X, X) - S[:, None]
    bbar2 = (dev**2).sum((1, 2, 3)) / T**2
    lw_gap = np.abs(bbar2 / FS2 - (T - 1) / T * V / FS2).max()
    # shortcut forms
    short_gap = max(np.abs(V - (M - trS2) / (T - 1)).max(),
                    np.abs(C - (M - trS**2) / (N * (T - 1))).max())
    # Section 4 Gaussian estimate of V
    Vg = (T * trS**2 + (T - 2) * trS2) / ((T + 2) * (T - 1))
    bg = T * (T * trS2 - trS**2) / ((T + 2) * (T - 1))
    out = dict(M=M, U2=U2, U3=U3, V=V, C=C, D=FS2, Vgauss=Vg, Cgauss=2 * bg / (N * T))
    out["aDF"] = (V - C) / FS2
    out["aG"] = (Vg - 2 * bg / (N * T)) / FS2
    out["aG_lt_aDF"] = (out["aG"] < out["aDF"]).astype(float)
    # Rao-Blackwell check: E[(M - Mrb) h(S)] = 0 under Gaussian data
    res = M - T * (trS**2 + 2 * trS2) / (T + 2)
    for name, h in [("1", np.ones_like(trS)), ("trS", trS), ("trS2", trS2),
                    ("(trS)^2", trS**2), ("s11*s22", S[:, 0, 0] * S[:, 1, 1]),
                    ("s12^2", S[:, 0, 1] ** 2)]:
        out[f"RB_{name}"] = res * h / h.mean()
    out.update({f"SURE{a}": sure[a] for a in ALPHAS})
    out.update({f"loss{a}": loss[a] for a in ALPHAS})
    return out, lw_gap, short_gap


for dist in ["gauss", "t10"]:
    tr = truth(dist)
    acc, gaps = {}, []
    for _ in range(REPS // CHUNK):
        o, g1, g2 = stats(draw(dist, CHUNK))
        gaps += [g1, g2]
        for k, v in o.items():
            acc.setdefault(k, []).append(v)
    acc = {k: np.concatenate(v) for k, v in acc.items()}
    mean = {k: v.mean() for k, v in acc.items()}
    se = {k: v.std(ddof=1) / np.sqrt(REPS) for k, v in acc.items()}
    print(f"\n=== {dist}: N={N}, T={T}, reps={REPS} ===")
    print(f"{'quantity':>12} {'truth':>10} {'MC mean':>10} {'s.e.':>8} {'z':>6}")
    for k in ["M", "U2", "U3", "V", "C", "D"]:
        z = (mean[k] - tr[k]) / se[k]
        print(f"{k:>12} {tr[k]:10.4f} {mean[k]:10.4f} {se[k]:8.4f} {z:6.2f}")
    print(f"{'Vgauss(S4)':>12} {tr['V']:10.4f} {mean['Vgauss']:10.4f} {se['Vgauss']:8.4f} "
          f"{(mean['Vgauss'] - tr['V']) / se['Vgauss']:6.2f}")
    print(f"{'Cgauss(S4)':>12} {tr['C']:10.4f} {mean['Cgauss']:10.4f} {se['Cgauss']:8.4f} "
          f"{(mean['Cgauss'] - tr['C']) / se['Cgauss']:6.2f}")
    print(f"  sd(V_hat) = {acc['V'].std():.4f}   sd(Vgauss) = {acc['Vgauss'].std():.4f}")
    print(f"  mean alpha_hat: DF {mean['aDF']:.4f}  Gaussian {mean['aG']:.4f};"
          f"  P(alpha_G < alpha_DF) = {mean['aG_lt_aDF']:.4f}")
    print("  Rao-Blackwell check, z-scores of E[(M - T((trS)^2+2trS^2)/(T+2)) h(S)]:")
    print("   " + "  ".join(f"{k[3:]}: {mean[k] / se[k]:6.2f}" for k in mean if k.startswith("RB_")))
    print(f"{'alpha':>6} {'R closed':>10} {'MC loss':>10} {'s.e.':>7} {'E SURE':>10} {'s.e.':>7}")
    for a in ALPHAS:
        print(f"{a:6.2f} {tr['R'][a]:10.4f} {mean[f'loss{a}']:10.4f} {se[f'loss{a}']:7.4f} "
              f"{mean[f'SURE{a}']:10.4f} {se[f'SURE{a}']:7.4f}")
    print(f"  max |LW - (T-1)/T*V/||F-S||^2| = {max(gaps[0::2]):.2e};"
          f"  max shortcut-form gap = {max(gaps[1::2]):.2e}")
    print(f"  alpha* = (V-C)/D = {(tr['V'] - tr['C']) / tr['D']:.4f}")
