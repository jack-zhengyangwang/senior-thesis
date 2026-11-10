"""Single-index target: four ways to estimate the cross term C, and their performance curves.

Model (one day t): z_t = (m_t, x_t') in R^{1+N}, mean zero, Cov(z_t) = J, i.i.d. over days.
    x_t = b m_t + e_t,  Var(m_t) = SIGMA_M2,  e_t has covariance Psi = diag(d) + g g'  (a weak second
    "industry" factor, so the single-index model is misspecified).
    Gaussian: z_t ~ N(0, J).  t10: z_t = sqrt(w_t) * N(0, J), w_t = 8 / chi2_10 (E w = 1, so Cov = J);
    the mixing variable multiplies the WHOLE vector (market and assets together).

Statistics (known zero mean, divisor T):
    S = (1/T) sum x_t x_t',  a_i = s_{0i} = (1/T) sum m_t x_ti,  B = s_{00}^2 = (1/T) sum m_t^2,
    in-sample single-index target F: f_ij = a_i a_j / B (i != j), f_ii = s_ii.
    stale target Ft (direction #5): ft_ij = B bt_i bt_j (i != j), ft_ii = s_ii, bt from an earlier,
        independent window of T0 = T days.
    reference target nu I, nu = tr(S)/N.
    Sigma_hat(alpha) = alpha F + (1 - alpha) S.

Risk (Yi's lemma): R(alpha) = alpha^2 D + (1 - 2 alpha) V + 2 alpha C, with D = E||F - S||^2,
V = sum_ij Var(s_ij), C = sum_ij Cov(f_ij, s_ij) = V_diag + C_off (diagonal of F equals diagonal of S).
All directions share ||F - S||^2 and the distribution-free V-hat of Section 5; they differ in C_off-hat:
    #2 plug-in Stein  : (1/T) sum_{i!=j} [(s_ii a_j^2 + s_jj a_i^2 + 2 s_ij a_i a_j)/B - 2 f_ij^2]
    #3 delta method   : sum_{i!=j} (a_j/B) K1_ij + (a_i/B) K1_ji - (a_i a_j/B^2) K2_ij,
                        K1_ij = cov-hat(m x_i, x_i x_j)/T, K2_ij = cov-hat(m^2, x_i x_j)/T  (divisor T-1)
    #4 bootstrap      : bootstrap covariance of (f*_ij, s*_ij), B_BOOT resamples of days
    #5 stale beta     : sum_{i!=j} bt_i bt_j K2_ij   (exactly unbiased, for the stale-target risk)

True risk: Monte Carlo of the actual loss. The loss is quadratic in alpha,
    L(alpha) = ||S - Sigma||^2 + 2 alpha <S - Sigma, F - S> + alpha^2 ||F - S||^2,
so the true curve needs only three Monte Carlo means. C_off is estimated by the Monte Carlo mean of
sum_{i!=j} f_ij (s_ij - sigma_ij), which is unbiased for sum_{i!=j} Cov(f_ij, s_ij).

Every experiment has its own generator default_rng([SEED, k, ...]).
Run:  python3 sims/sim_single_index.py
Writes sims/out/si_*.pdf/png and sims/out/si_results.txt.
"""
import os
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
SEED = 20261110
REPS = 20000          # replications per cell (true risk and the cheap directions #2, #3, #5)
BOOT_REPS = 2000      # replications per cell that also run the bootstrap (#4); the first BOOT_REPS of REPS
B_BOOT = 200          # bootstrap resamples
CHUNK = 500
BOOT_CHUNK = 25
NS = (10, 30)
TS = (30, 120)
DISTS = ("gauss", "t10")
DF_T = 10
KEY_DIST = {"gauss": 1, "t10": 2}
SIGMA_M2 = 1.0
REP_CELL = (30, 30)   # representative (N, T) for the performance-curve figure
GRID = np.linspace(0, 1, 201)

DIRS = ("stein", "delta", "boot", "stale")
COL = {"stein": "#0072B2", "delta": "#E69F00", "boot": "#009E73", "stale": "#CC79A7", "nuI": "#D55E00",
       "truth": "black", "insample": "0.45"}
NAME = {"stein": "#2 plug-in Stein", "delta": "#3 delta method (LW 2003)", "boot": "#4 bootstrap",
        "stale": r"#5 stale $\beta$ (exact)"}
SHORT = {"stein": "#2 Stein", "delta": "#3 delta", "boot": "#4 boot", "stale": "#5 stale"}
MARK = {"stein": "o", "delta": "s", "boot": "^", "stale": "D", "oracle": "_"}
LABEL_DIST = {"gauss": "Gaussian", "t10": r"$t_{10}$"}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "0.9", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "savefig.bbox": "tight"})


# ------------------------------------------------------------------ model
def model(N):
    """Joint covariance J of z = (m, x'), and the asset covariance Sigma = J[1:, 1:]."""
    b = np.linspace(0.5, 1.5, N)
    d = np.linspace(0.5, 1.5, N)[::-1]
    g = 0.4 * np.where(np.arange(N) < N // 2, 1.0, -1.0)     # second factor: +0.4 / -0.4 loadings
    Psi = np.diag(d) + np.outer(g, g)
    J = np.empty((N + 1, N + 1))
    J[0, 0] = SIGMA_M2
    J[0, 1:] = J[1:, 0] = SIGMA_M2 * b
    J[1:, 1:] = SIGMA_M2 * np.outer(b, b) + Psi
    return J


def draw(rng, R, T, LJ, dist):
    Z = rng.standard_normal((R, T, LJ.shape[0])) @ LJ.T
    if dist == "gauss":
        return Z
    w = (DF_T - 2) / rng.chisquare(DF_T, size=(R, T))
    return Z * np.sqrt(w)[:, :, None]


def kappa(dist):
    return 1.0 if dist == "gauss" else (DF_T - 2) / (DF_T - 4)


def V_exact(Sigma, T, dist):
    """sum_ij Var(s_ij) for an elliptical scale mixture: Var(x_i x_j) = kappa (s_ii s_jj + 2 s_ij^2) - s_ij^2."""
    dg = np.diag(Sigma)
    v = kappa(dist) * (np.outer(dg, dg) + 2 * Sigma ** 2) - Sigma ** 2
    return v.sum() / T, np.trace(v) / T


# ------------------------------------------------------------------ per-replication statistics
def offsum(M):
    """Sum of off-diagonal entries, over the last two axes."""
    return M.sum(axis=(-2, -1)) - np.trace(M, axis1=-2, axis2=-1)


def stats_chunk(Z, Zp, Sigma, J):
    R, T, _ = Z.shape
    N = Sigma.shape[0]
    m, X = Z[:, :, 0], Z[:, :, 1:]
    S = np.matmul(X.transpose(0, 2, 1), X) / T
    a = np.einsum("rt,rti->ri", m, X) / T
    B = np.mean(m ** 2, axis=1)
    sdiag = np.diagonal(S, axis1=1, axis2=2)
    eye = np.eye(N, dtype=bool)

    F = a[:, :, None] * a[:, None, :] / B[:, None, None]
    F[:, eye] = sdiag
    # stale target from an earlier, independent window
    mp, Xp = Zp[:, :, 0], Zp[:, :, 1:]
    bt = np.einsum("rt,rti->ri", mp, Xp) / np.sum(mp ** 2, axis=1)[:, None]
    Ft = B[:, None, None] * bt[:, :, None] * bt[:, None, :]
    Ft[:, eye] = sdiag
    # nu I
    trS = np.trace(S, axis1=1, axis2=2)
    Fn = (trS / N)[:, None, None] * np.eye(N)

    SmS = S - Sigma
    out = {"A": np.sum(SmS ** 2, axis=(1, 2))}                      # ||S - Sigma||^2 = L(S)
    for k, FF in (("in", F), ("st", Ft), ("nu", Fn)):
        FmS = FF - S
        out["D_" + k] = np.sum(FmS ** 2, axis=(1, 2))                # ||F - S||^2
        out["Cr_" + k] = np.sum(SmS * FmS, axis=(1, 2))              # <S - Sigma, F - S>
    out["H_in"] = offsum(F * SmS)                                    # MC-unbiased for C_off
    out["H_st"] = offsum(Ft * SmS)

    # distribution-free V-hat (Section 5), split into its diagonal and off-diagonal parts
    M = np.mean(np.sum(X ** 2, axis=2) ** 2, axis=1)
    trS2 = np.sum(S ** 2, axis=(1, 2))
    Vhat = (M - trS2) / (T - 1)
    Vdiag = np.sum(np.mean(X ** 4, axis=1) - sdiag ** 2, axis=1) / (T - 1)
    out["Vhat"], out["Vdiag"] = Vhat, Vdiag

    # direction #2: plug-in Stein
    Bb = B[:, None, None]
    st = (sdiag[:, :, None] * a[:, None, :] ** 2 + sdiag[:, None, :] * a[:, :, None] ** 2
          + 2 * S * a[:, :, None] * a[:, None, :]) / Bb - 2 * F ** 2
    out["Coff_stein"] = offsum(st) / T
    # the same expression with the TRUE J (verification (b)): exactly unbiased under normality
    s0 = J[0, 1:]
    dS = np.diag(Sigma)
    G = ((dS[None, :, None] * a[:, None, :] ** 2 + Sigma[None] * a[:, :, None] * a[:, None, :]
          + s0[None, :, None] * (S * a[:, None, :] + sdiag[:, None, :] * a[:, :, None])) / Bb
         - 2 * s0[None, :, None] * a[:, :, None] * a[:, None, :] ** 2 / Bb ** 2)
    out["G_trueJ"] = offsum(G) / T

    # direction #3: delta method with sample covariances of daily products (divisor T - 1)
    Q1 = np.matmul((m[:, :, None] * X ** 2).transpose(0, 2, 1), X)  # Q1_ij = sum_t m x_i^2 x_j
    K1 = (Q1 - T * a[:, :, None] * S) / ((T - 1) * T)                # cov-hat(m x_i, x_i x_j) / T
    Q2 = np.matmul((m[:, :, None] ** 2 * X).transpose(0, 2, 1), X)   # sum_t m^2 x_i x_j
    K2 = (Q2 - T * Bb * S) / ((T - 1) * T)                           # cov-hat(m^2, x_i x_j) / T
    dl = (a[:, None, :] * K1 + a[:, :, None] * K1.transpose(0, 2, 1)) / Bb \
        - a[:, :, None] * a[:, None, :] * K2 / Bb ** 2
    out["Coff_delta"] = offsum(dl)
    # direction #5: stale beta
    out["Coff_stale"] = offsum(bt[:, :, None] * bt[:, None, :] * K2)
    return out


def boot_chunk(rng, Z):
    """Bootstrap estimate of C_off = sum_{i != j} Cov(f_ij, s_ij), resampling days with replacement."""
    R, T, _ = Z.shape
    m, X = Z[:, :, 0], Z[:, :, 1:]
    N = X.shape[2]
    P = (X[:, :, :, None] * X[:, :, None, :]).reshape(R, T, N * N)
    W = rng.multinomial(T, np.full(T, 1.0 / T), size=(R, B_BOOT)).astype(float)   # R x B x T counts
    Ss = (W @ P / T).reshape(R, B_BOOT, N, N)
    As = W @ (m[:, :, None] * X) / T
    Bs = W @ (m ** 2)[:, :, None] / T
    Fs = As[:, :, :, None] * As[:, :, None, :] / Bs[:, :, :, None]
    cov = (np.mean(Fs * Ss, axis=1) - Fs.mean(axis=1) * Ss.mean(axis=1)) * B_BOOT / (B_BOOT - 1)
    return offsum(cov)


def run_cell(N, T, dist, reps=REPS, boot_reps=BOOT_REPS):
    J = model(N)
    Sigma = J[1:, 1:]
    LJ = np.linalg.cholesky(J)
    rng = np.random.default_rng([SEED, 1, KEY_DIST[dist], N, T])
    rngb = np.random.default_rng([SEED, 2, KEY_DIST[dist], N, T])
    parts, boots = [], []
    for r0 in range(0, reps, CHUNK):
        r = min(CHUNK, reps - r0)
        Z = draw(rng, r, T, LJ, dist)
        Zp = draw(rng, r, T, LJ, dist)        # earlier window, T0 = T
        parts.append(stats_chunk(Z, Zp, Sigma, J))
        nb = max(0, min(r, boot_reps - r0))
        for b0 in range(0, nb, BOOT_CHUNK):
            boots.append(boot_chunk(rngb, Z[b0:min(nb, b0 + BOOT_CHUNK)]))
    c = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    c["Coff_boot"] = np.concatenate(boots)
    c["N"], c["T"], c["dist"], c["Sigma"] = N, T, dist, Sigma
    c["V_exact"], c["Vdiag_exact"] = V_exact(Sigma, T, dist)
    return c


# ------------------------------------------------------------------ derived quantities
def target_of(d):
    return "st" if d == "stale" else "in"


def true_curve(c, k, idx=slice(None)):
    """R(alpha) on GRID from the Monte Carlo means; alpha* (clipped) and the unclipped alpha*."""
    A, Cr, D = c["A"][idx].mean(), c["Cr_" + k][idx].mean(), c["D_" + k][idx].mean()
    R = A + 2 * GRID * Cr + GRID ** 2 * D
    a_raw = -Cr / D
    return R, float(np.clip(a_raw, 0, 1)), a_raw


def C_off_true(c, k):
    h = c["H_" + k]
    return h.mean(), h.std(ddof=1) / np.sqrt(h.size)


def chat(c, d):
    """C-hat = V-hat_diag + C_off-hat (full C including the diagonal Var(s_ii) terms)."""
    co = c["Coff_" + d]
    return c["Vdiag"][: co.size] + co


def alpha_hat(c, d):
    n = c["Coff_" + d].size
    raw = (c["Vhat"][:n] - chat(c, d)) / c["D_" + target_of(d)][:n]
    return np.clip(raw, 0, 1), raw


def sure_curves(c, d):
    n = c["Coff_" + d].size
    k = target_of(d)
    return (GRID[None, :] ** 2 * c["D_" + k][:n, None] + (1 - 2 * GRID[None, :]) * c["Vhat"][:n, None]
            + 2 * GRID[None, :] * chat(c, d)[:, None])


def loss_at(c, k, a):
    n = a.size
    return c["A"][:n] + 2 * a * c["Cr_" + k][:n] + a ** 2 * c["D_" + k][:n]


def prial(c, d):
    """PRIAL of alpha-hat_d and of the oracle alpha*, on the same replications as direction d.
    Oracle uses the alpha* from all REPS replications; returns (PRIAL_d, SE of PRIAL_d - PRIAL_oracle, PRIAL_oracle)."""
    k = target_of(d)
    a, _ = alpha_hat(c, d)
    n = a.size
    _, a_star, _ = true_curve(c, k)
    LS = c["A"][:n]
    Ld = loss_at(c, k, a)
    Lo = loss_at(c, k, np.full(n, a_star))
    base = LS.mean()
    p_d = 100 * (base - Ld.mean()) / base
    p_o = 100 * (base - Lo.mean()) / base
    se = 100 * (Ld - Lo).std(ddof=1) / np.sqrt(n) / base
    return p_d, se, p_o


# ------------------------------------------------------------------ verification
def check_fii(rng, N=6, T=25):
    """(a) The LW 2003 single-index target (zero mean, divisor T) has f_ii = s_ii."""
    J = model(N)
    Z = draw(rng, 1, T, np.linalg.cholesky(J), "gauss")[0]
    m, X = Z[:, 0], Z[:, 1:]
    beta = X.T @ m / (m @ m)                        # OLS slope without intercept
    resid = X - np.outer(m, beta)
    delta = np.mean(resid ** 2, axis=0)             # residual variance, divisor T
    s00 = np.mean(m ** 2)
    S = X.T @ X / T
    fii = s00 * beta ** 2 + delta
    return "(a) f_ii = s00^2 beta_i^2 + delta_ii vs s_ii: max |diff| = %.2e" % np.max(np.abs(fii - np.diag(S)))


def check_stein_trueJ(N=5, T=12, reps=400000):
    """(b) E[sum_{i!=j} f_ij (s_ij - sigma_ij)] = E[G_trueJ] under normality; fails under t10."""
    lines = []
    for dist in DISTS:
        J = model(N)
        Sigma = J[1:, 1:]
        rng = np.random.default_rng([SEED, 3, KEY_DIST[dist]])
        LJ = np.linalg.cholesky(J)
        H, G, Ps = [], [], []
        for r0 in range(0, reps, 20000):
            Z = draw(rng, 20000, T, LJ, dist)
            st = stats_chunk(Z, Z, Sigma, J)
            H.append(st["H_in"]); G.append(st["G_trueJ"]); Ps.append(st["Coff_stein"])
        H, G, Ps = map(np.concatenate, (H, G, Ps))
        d = H - G
        z = d.mean() / (d.std(ddof=1) / np.sqrt(reps))
        dp = Ps - H
        zp = dp.mean() / (dp.std(ddof=1) / np.sqrt(reps))
        lines.append("(b) %-5s N=%d T=%d reps=%d: C_off (MC of sum f(s-sigma)) = %.5f; E[Stein with true J] = %.5f;"
                     " diff z = %+.2f | plug-in Stein mean = %.5f, ratio %.3f, z = %+.1f"
                     % (dist, N, T, reps, H.mean(), G.mean(), z, Ps.mean(), Ps.mean() / H.mean(), zp))
    return lines


def check_stale(c):
    """(c) E[C_off-hat_5] = C_off,5 and (d) E[SURE_5(alpha)] = R_5(alpha): paired z-scores."""
    h, co = c["H_st"], c["Coff_stale"]
    d = co - h
    z = d.mean() / (d.std(ddof=1) / np.sqrt(d.size))
    out = ["(c) %-5s N=%d T=%d: C_off,5 true (MC) = %.5f, mean C_off-hat_5 = %.5f, ratio %.4f, paired z = %+.2f"
           % (c["dist"], c["N"], c["T"], h.mean(), co.mean(), co.mean() / h.mean(), z)]
    zs = []
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        sure = a ** 2 * c["D_st"] + (1 - 2 * a) * c["Vhat"] + 2 * a * chat(c, "stale")
        L = loss_at(c, "st", np.full(c["A"].size, a))
        dd = sure - L
        zs.append("a=%.2f: R=%.3f meanSURE=%.3f z=%+.2f" % (a, L.mean(), sure.mean(),
                                                             dd.mean() / (dd.std(ddof=1) / np.sqrt(dd.size))))
    out.append("(d) %-5s N=%d T=%d: " % (c["dist"], c["N"], c["T"]) + "; ".join(zs))
    vd = c["Vhat"] - c["A"]
    out.append("    V-hat check: mean V-hat = %.4f, exact V = %.4f, MC E L(S) = %.4f, z(V-hat - L(S)) = %+.2f"
               % (c["Vhat"].mean(), c["V_exact"], c["A"].mean(), vd.mean() / (vd.std(ddof=1) / np.sqrt(vd.size))))
    return out


def check_stale_big():
    """(c), (d) again with many more replications (no bootstrap), for tight z-scores."""
    lines = []
    for dist in DISTS:
        for N, T, reps in ((10, 30, 400000), (10, 120, 200000)):
            J = model(N)
            Sigma = J[1:, 1:]
            LJ = np.linalg.cholesky(J)
            rng = np.random.default_rng([SEED, 4, KEY_DIST[dist], N, T])
            parts = []
            for r0 in range(0, reps, 10000):
                Z = draw(rng, 10000, T, LJ, dist)
                Zp = draw(rng, 10000, T, LJ, dist)
                st = stats_chunk(Z, Zp, Sigma, J)
                parts.append({k: st[k] for k in ("A", "D_st", "Cr_st", "H_st", "Coff_stale", "Vhat", "Vdiag")})
            c = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            c["Coff_stein"] = c["Coff_stale"]   # placeholder so helper functions see the right size
            c.update(N=N, T=T, dist=dist)
            c["V_exact"], _ = V_exact(Sigma, T, dist)
            lines += [l.replace("(c)", "(c+)").replace("(d)", "(d+)") + ("  [reps=%d]" % reps if l.startswith("(c)") else "")
                      for l in check_stale(c)]
    return lines


# ------------------------------------------------------------------ figures
def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, name + "." + ext), dpi=200)
    plt.close(fig)


def fig_curves(res):
    N, T = REP_CELL
    fig, axes = plt.subplots(4, 2, figsize=(6.0, 8.2), sharex=True)
    for j, dist in enumerate(DISTS):
        c = res[(dist, N, T)]
        for i, d in enumerate(DIRS):
            ax = axes[i, j]
            k = target_of(d)
            R, a_star, _ = true_curve(c, k)
            sc = sure_curves(c, d)
            lo, hi = np.percentile(sc, [10, 90], axis=0)
            ah, _ = alpha_hat(c, d)
            ax.fill_between(GRID, lo, hi, color=COL[d], alpha=0.18, lw=0, label="SURE 10–90% band")
            ax.plot(GRID, sc.mean(axis=0), color=COL[d], lw=1.8, label=r"mean SURE$_d(\alpha)$")
            ax.plot(GRID, R, color="black", ls=(0, (3, 2)), lw=1.1, label=r"true $R(\alpha)$")
            if d == "stale":
                Rin, _, _ = true_curve(c, "in")
                ax.plot(GRID, Rin, color=COL["insample"], ls=":", lw=1.1, label=r"true $R(\alpha)$, in-sample $\beta$")
            ax.axvline(a_star, color="black", lw=0.8)
            ax.axvline(ah.mean(), color=COL[d], lw=0.9, ls="--")
            top = max(R[0], R[-1]) * 1.4
            ax.set_ylim(0, top)
            ax.text(0.97, 0.05, r"$\alpha^*$=%.2f, mean $\hat\alpha_d$=%.2f" % (a_star, ah.mean()),
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
            if i == 0:
                ax.set_title(LABEL_DIST[dist] + " data")
            if j == 0:
                ax.set_ylabel(NAME[d] + "\nFrobenius risk", fontsize=8)
            if i == 3:
                ax.set_xlabel(r"shrinkage intensity $\alpha$")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], color="black", ls=(0, (3, 2)), lw=1.1),
         Line2D([], [], color="0.3", lw=1.8), Patch(color="0.3", alpha=0.25),
         Line2D([], [], color="black", lw=0.8), Line2D([], [], color="0.3", lw=0.9, ls="--"),
         Line2D([], [], color=COL["insample"], ls=":", lw=1.1)]
    l = [r"true risk $R(\alpha)$", r"mean SURE$_d(\alpha)$ (color = direction)", "SURE 10–90% band",
         r"oracle $\alpha^*$", r"mean $\hat\alpha_d$", r"in-sample-$\beta$ true risk (row #5)"]
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.035))
    fig.suptitle(r"Performance curves, single-index target, $N=%d$, $T=%d$" % (N, T), fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "si_curves")


def fig_targets(res, dist="gauss"):
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.6))
    for i, N in enumerate(NS):
        for j, T in enumerate(TS):
            ax = axes[i, j]
            c = res[(dist, N, T)]
            base = c["A"].mean()
            for k, lab, col, ls in (("nu", r"$\nu I$ target", COL["nuI"], "-"),
                                    ("in", r"single index, in-sample $\beta$", "black", "-"),
                                    ("st", r"single index, stale $\beta$ (#5)", COL["stale"], "--")):
                R, a_star, _ = true_curve(c, k)
                ax.plot(GRID, R / base, color=col, ls=ls, lw=1.6, label=lab)
                ax.plot([a_star], [np.interp(a_star, GRID, R) / base], marker="o", ms=4.5, color=col,
                        mec="white", mew=0.6, ls="none")
            ax.axhline(1, color="0.6", lw=0.7)
            ax.set_ylim(0, 2.0)
            ax.set_title(r"$N=%d$, $T=%d$" % (N, T))
            if i == 1:
                ax.set_xlabel(r"shrinkage intensity $\alpha$")
            if j == 0:
                ax.set_ylabel(r"$R(\alpha)\,/\,R(0)$")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("True risk curves by target (%s data); dots mark each target's $\\alpha^*$" % LABEL_DIST[dist],
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "si_targets")


def fig_prial(res):
    cells = [(N, T) for N in NS for T in TS]
    xl = ["N=%d\nT=%d" % nt for nt in cells]
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.0), sharex=True)
    offs = dict(zip(DIRS, (-0.24, -0.08, 0.08, 0.24)))
    for j, dist in enumerate(DISTS):
        for x, (N, T) in enumerate(cells):
            c = res[(dist, N, T)]
            # one oracle bar per target: in-sample (under #2-#4) and stale (under #5), both from all REPS
            for k, (x0, x1), col, lab in (("in", (-0.33, 0.15), "black", r"oracle (in-sample $\beta$)"),
                                          ("st", (0.16, 0.33), COL["stale"], r"oracle (stale $\beta$)")):
                R, a_star, _ = true_curve(c, k)
                base = c["A"].mean()
                po = 100 * (base - np.interp(a_star, GRID, R)) / base
                axes[0, j].plot([x + x0, x + x1], [po, po], color=col, lw=1.6, label=lab if x == 0 else None)
                axes[1, j].plot([x + x0, x + x1], [a_star, a_star], color=col, lw=1.6)
            for d in DIRS:
                p, se, _ = prial(c, d)
                axes[0, j].plot(x + offs[d], p, marker=MARK[d], color=COL[d], ms=5.5, ls="none",
                                label=SHORT[d] if x == 0 else None)
                ah, _ = alpha_hat(c, d)
                axes[1, j].errorbar(x + offs[d], ah.mean(), yerr=ah.std(), marker=MARK[d], color=COL[d], ms=5,
                                    elinewidth=0.9, capsize=0, ls="none")
        axes[0, j].set_title(LABEL_DIST[dist] + " data")
        axes[1, j].set_xticks(range(len(cells)))
        axes[1, j].set_xticklabels(xl)
    axes[0, 0].set_ylabel("PRIAL over $S$ (%)")
    axes[1, 0].set_ylabel(r"$\hat\alpha_d$ (mean $\pm$ sd); bar = $\alpha^*$")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "si_prial")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    t0 = time.time()
    out = ["Single-index target: four estimates of the cross term. seed %d; REPS=%d per cell "
           "(BOOT_REPS=%d for #4, B=%d); T0=T; alpha clipped to [0,1]." % (SEED, REPS, BOOT_REPS, B_BOOT),
           "C_off = sum_{i!=j} Cov(f_ij, s_ij) (MC mean of sum f_ij (s_ij - sigma_ij)); bias ratio = E[C_off-hat]/C_off.",
           "PRIAL = 100 (E L(S) - E L(est)) / E L(S) on the same replications; oracle uses alpha* of that target.",
           "Directions #2-#4 estimate the in-sample-beta target's curve; #5 estimates the stale-beta target's curve.", "",
           "VERIFICATION", check_fii(np.random.default_rng([SEED, 0]))]
    out += check_stein_trueJ()
    res = {}
    for dist in DISTS:
        for N in NS:
            for T in TS:
                tc = time.time()
                res[(dist, N, T)] = run_cell(N, T, dist)
                print("cell", dist, N, T, "%.1fs" % (time.time() - tc), flush=True)
    for c in res.values():
        out += check_stale(c)
    out += check_stale_big()

    out += ["", "RESULTS",
            "%-5s %3s %4s | %-8s %8s %8s | %-8s %7s %6s | %6s %6s %6s %6s | %6s %13s %6s"
            % ("dist", "N", "T", "dir", "C_off", "(se)", "E[Chat]", "ratio", "a*", "PRIAL", "(seΔ)", "orac",
               "Δ", "a*raw", "mean(sd) ahat", "clip%")]
    for (dist, N, T), c in res.items():
        for d in DIRS:
            k = target_of(d)
            co, se = C_off_true(c, k)
            # the truth uses all REPS replications (the BOOT_REPS subset is too noisy for a denominator)
            _, a_star, a_raw = true_curve(c, k)
            ah, raw = alpha_hat(c, d)
            p, pse, po = prial(c, d)
            n = c["Coff_" + d].size
            dd = c["Coff_" + d] - c["H_" + k][:n]        # paired: estimate minus unbiased MC draw of the truth
            zb = dd.mean() / (dd.std(ddof=1) / np.sqrt(n))
            out.append("%-5s %3d %4d | %-8s %8.4f %8.4f | %8.4f %7.3f %6.3f | %6.1f %6.2f %6.1f %+6.1f | %6.3f %6.3f(%5.3f) %5.1f | z_bias %+6.2f"
                       % (dist, N, T, d, co, se, c["Coff_" + d].mean(), c["Coff_" + d].mean() / co, a_star,
                          p, pse, po, p - po, a_raw, ah.mean(), ah.std(), 100 * np.mean((raw < 0) | (raw > 1)), zb))
        Rn, an, _ = true_curve(c, "nu")
        Ri, ai, _ = true_curve(c, "in")
        Rs, as_, _ = true_curve(c, "st")
        base = c["A"].mean()
        mn = lambda R: 100 * (base - R.min()) / base
        out.append("      oracle PRIAL by target: nuI %.1f (a*=%.3f) | in-sample SI %.1f (a*=%.3f) | stale SI %.1f (a*=%.3f)"
                   % (mn(Rn), an, mn(Ri), ai, mn(Rs), as_))
        sc = {d: sure_curves(c, d).mean(axis=0) for d in DIRS}
        gaps = []
        for d in DIRS:
            R, _, _ = true_curve(c, target_of(d), slice(0, c["Coff_" + d].size))
            gaps.append("%s %.3f" % (d, np.max(np.abs(sc[d] - R)) / R[0]))
        out.append("      max_alpha |mean SURE_d - R|/R(0): " + ", ".join(gaps))
    out.append("")
    out.append("runtime %.1f s" % (time.time() - t0))
    fig_curves(res)
    fig_targets(res)
    fig_prial(res)
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(OUT, "si_results.txt"), "w") as f:
        f.write(txt + "\n")
