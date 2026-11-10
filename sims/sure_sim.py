"""SURE-tuned shrinkage toward nu*I: unbiasedness checks and comparison with LW, RBLW, OAS.

Setup matches Chen et al. (2010): zero-mean data, S = (1/T) sum_t X_t X_t^T, F = tr(S)/p * I.
Run:  python3 sims/sure_sim.py   (writes figures + results.txt into sims/out/)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261026)
NU_T = 10  # t degrees of freedom; >8 so Vhat, Chat have finite variance


# ---------------------------------------------------------------- data
def ar1(p, r=0.5):
    idx = np.arange(p)
    return r ** np.abs(idx[:, None] - idx[None, :])


def draw(T, L, dist):
    """T x p sample with covariance L L^T. 't' is multivariate t scaled to the same covariance."""
    Z = rng.standard_normal((T, L.shape[0])) @ L.T
    if dist == "gauss":
        return Z
    w = (NU_T - 2) / rng.chisquare(NU_T, size=T)  # E w = 1
    return Z * np.sqrt(w)[:, None]


# ---------------------------------------------------------------- true quantities (closed form)
def truth(Sigma, T, dist):
    """Return V = sum Var(s_ij), C = sum Cov(f_ij, s_ij), D = E||F - S||^2."""
    p = Sigma.shape[0]
    tr, tr2 = np.trace(Sigma), np.sum(Sigma ** 2)
    kurt = 1.0 if dist == "gauss" else (NU_T - 2) / (NU_T - 4)  # E w^2
    m4 = kurt * (tr ** 2 + 2 * tr2)  # E||X||^4
    V = (m4 - tr2) / T
    C = (m4 - tr ** 2) / (p * T)
    D = tr2 + V - tr ** 2 / p - C
    return V, C, D


def risk(alpha, V, C, D):
    return alpha ** 2 * D + (1 - 2 * alpha) * V + 2 * alpha * C


# ---------------------------------------------------------------- estimators
def stats(X):
    T, p = X.shape
    S = X.T @ X / T
    sq = np.sum(X ** 2, axis=1)  # ||X_t||^2
    M = np.mean(sq ** 2)
    trS, trS2 = np.trace(S), np.sum(S ** 2)
    U2 = (T ** 2 * trS2 - T * M) / (T * (T - 1))
    U3 = (T ** 2 * trS ** 2 - T * M) / (T * (T - 1))
    Vh = (M - U2) / T
    Ch = (M - U3) / (p * T)
    d2 = trS2 - trS ** 2 / p  # ||F - S||^2
    return S, trS, trS2, Vh, Ch, d2


def alphas(X):
    T, p = X.shape
    S, trS, trS2, Vh, Ch, d2 = stats(X)
    a = {
        "SURE": (Vh - Ch) / d2,
        "LW": (T - 1) / T * Vh / d2,  # = eq. (13) of Chen et al., see note Prop. 2
        "RBLW": ((T - 2) / T * trS2 + trS ** 2) / ((T + 2) * d2),
        "OAS": ((1 - 2 / p) * trS2 + trS ** 2) / ((T + 1 - 2 / p) * d2),
    }
    return {k: float(np.clip(v, 0, 1)) for k, v in a.items()}, S, trS, Vh, Ch, d2


def loss(alpha, S, trS, Sigma):
    p = S.shape[0]
    E = (1 - alpha) * S + alpha * trS / p * np.eye(p) - Sigma
    return np.sum(E ** 2)


# ---------------------------------------------------------------- experiment 1: SURE curve vs true risk
def exp_curves(p=100, T=20, reps=4000, n_show=25):
    Sigma = ar1(p)
    L = np.linalg.cholesky(Sigma)
    grid = np.linspace(0, 1, 101)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    lines = []
    for ax, dist in zip(axes, ["gauss", "t"]):
        V, C, D = truth(Sigma, T, dist)
        sure_sum, loss_sum = np.zeros_like(grid), np.zeros_like(grid)
        for r in range(reps):
            X = draw(T, L, dist)
            S, trS, _, Vh, Ch, d2 = stats(X)
            sure = grid ** 2 * d2 + (1 - 2 * grid) * Vh + 2 * grid * Ch
            sure_sum += sure
            loss_sum += np.array([loss(a, S, trS, Sigma) for a in grid]) if r < 500 else 0
            if r < n_show:
                ax.plot(grid, sure, color="0.75", lw=0.7, zorder=1)
        R = risk(grid, V, C, D)
        ax.plot(grid, sure_sum / reps, color="C0", lw=2, label="mean SURE(α) over %d draws" % reps)
        ax.plot(grid, R, "k--", lw=1.6, label="true risk R(α), closed form")
        ax.plot(grid, loss_sum / 500, "C3:", lw=1.6, label="Monte Carlo loss (500 draws)")
        astar = (V - C) / D
        ax.axvline(astar, color="C2", lw=1, label="oracle α* = %.3f" % astar)
        ax.axvline(min(1, 2 * astar), color="C2", lw=1, ls=":", label="2α* (risk-reduction bound)")
        ax.set_title("%s data, p=%d, T=%d" % ("Gaussian" if dist == "gauss" else "Multivariate t(%d)" % NU_T, p, T))
        ax.set_xlabel("shrinkage intensity α")
        ax.set_ylabel("Frobenius risk")
        ax.set_ylim(0, R[0] * 1.6)
        ax.legend(fontsize=7.5, loc="upper right")
        gap = np.max(np.abs(sure_sum / reps - R)) / R.max()
        lines.append("curves %-5s p=%d T=%d: max |mean SURE - R| / max R = %.4f, alpha*=%.4f, R(0)=%.2f, R(alpha*)=%.2f"
                     % (dist, p, T, gap, astar, R[0], risk(astar, V, C, D)))
    axes[0].text(0.02, 0.02, "grey: individual SURE(α) curves", transform=axes[0].transAxes, fontsize=7.5, color="0.4")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_sure_vs_risk.pdf"))
    fig.savefig(os.path.join(OUT, "fig_sure_vs_risk.png"), dpi=150)
    return lines


# ---------------------------------------------------------------- experiment 2: unbiasedness of V-hat, C-hat
def exp_unbiased(reps=20000):
    lines = ["%-6s %4s %4s %12s %12s %8s %12s %12s %8s" % ("dist", "p", "T", "V true", "mean Vhat", "MCse", "C true", "mean Chat", "MCse")]
    rows = []
    for dist in ["gauss", "t"]:
        for p, T in [(10, 5), (50, 10), (100, 40)]:
            Sigma = ar1(p)
            L = np.linalg.cholesky(Sigma)
            V, C, _ = truth(Sigma, T, dist)
            vs, cs = np.empty(reps), np.empty(reps)
            for r in range(reps):
                _, _, _, vs[r], cs[r], _ = stats(draw(T, L, dist))
            lines.append("%-6s %4d %4d %12.4f %12.4f %8.4f %12.5f %12.5f %8.5f" % (
                dist, p, T, V, vs.mean(), vs.std() / np.sqrt(reps), C, cs.mean(), cs.std() / np.sqrt(reps)))
            rows.append((dist, p, T, V, vs.mean(), vs.std() / np.sqrt(reps), C, cs.mean(), cs.std() / np.sqrt(reps)))
    return lines, rows


# ---------------------------------------------------------------- experiment 3: compare estimators
def exp_compare(p=100, Ts=(6, 10, 15, 20, 30, 50, 80), reps=2000):
    Sigma = ar1(p)
    L = np.linalg.cholesky(Sigma)
    names = ["Oracle", "SURE", "LW", "RBLW", "OAS"]
    res, lines = {}, []
    for dist in ["gauss", "t"]:
        mse = {k: [] for k in names}
        rho = {k: [] for k in names}
        for T in Ts:
            V, C, D = truth(Sigma, T, dist)
            astar = (V - C) / D
            acc = {k: [] for k in names}
            acr = {k: [] for k in names}
            for _ in range(reps):
                a, S, trS, *_ = alphas(draw(T, L, dist))
                a["Oracle"] = astar
                for k in names:
                    acc[k].append(loss(a[k], S, trS, Sigma))
                    acr[k].append(a[k])
            for k in names:
                mse[k].append(np.mean(acc[k]))
                rho[k].append(np.mean(acr[k]))
            lines.append("%-5s T=%3d  " % (dist, T) + "  ".join("%s %.2f (a=%.3f)" % (k, mse[k][-1], rho[k][-1]) for k in names))
        res[dist] = (mse, rho)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    style = {"Oracle": "k--", "SURE": "C0-o", "LW": "C1-s", "RBLW": "C2-^", "OAS": "C3-v"}
    for j, dist in enumerate(["gauss", "t"]):
        mse, rho = res[dist]
        base = np.array(mse["Oracle"])
        for k in names:
            axes[0, j].plot(Ts, np.array(mse[k]) / base, style[k], ms=4, label=k)
            axes[1, j].plot(Ts, rho[k], style[k], ms=4, label=k)
        axes[0, j].set_title("%s data, AR(1) r=0.5, p=%d" % ("Gaussian" if dist == "gauss" else "Multivariate t(%d)" % NU_T, p))
        axes[0, j].set_ylabel("MSE / oracle MSE")
        axes[1, j].set_ylabel("mean shrinkage α̂")
        axes[1, j].set_xlabel("sample size T")
        axes[0, j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_compare.pdf"))
    fig.savefig(os.path.join(OUT, "fig_compare.png"), dpi=150)
    return lines


# ---------------------------------------------------------------- identity check: LW formula (13) vs (T-1)/T * Vhat / d2
def check_lw_identity():
    X = draw(15, np.linalg.cholesky(ar1(30)), "t")
    T = X.shape[0]
    S, trS, trS2, Vh, Ch, d2 = stats(X)
    num = sum(np.sum((np.outer(x, x) - S) ** 2) for x in X)
    lw13 = num / (T ** 2 * d2)
    return "LW eq.(13) = %.10f ; (T-1)/T * Vhat/d2 = %.10f" % (lw13, (T - 1) / T * Vh / d2)


if __name__ == "__main__":
    out = [check_lw_identity(), ""]
    out += exp_curves()
    out.append("")
    ul, _ = exp_unbiased()
    out += ul
    out.append("")
    out += exp_compare()
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write(txt + "\n")
