"""Section 6 simulations: Gaussian SURE (Section 4) vs distribution-free SURE (Section 5) vs Ledoit-Wolf.

Setting: x_1..x_T iid, mean zero, Cov = Sigma; S = (1/T) sum_t x_t x_t'; target F = nu I, nu = tr(S)/N;
estimator Sigma_hat = alpha F + (1 - alpha) S; loss = squared Frobenius norm.

Run:  python3 sims/sim_section6.py
Writes sims/out/sec6_*.pdf/png and sims/out/sec6_results.txt.
"""
import os
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
SEED = 20261109
REPS = 5000
CHUNK = 250
NS = (10, 50, 100)
TS = (20, 60, 250)
DISTS = ("gauss", "t10", "t5")
DF = {"t10": 10, "t5": 5}
LABEL_DIST = {"gauss": "Gaussian", "t10": r"$t_{10}$", "t5": r"$t_5$"}

# estimator order, colors (Okabe-Ito), markers -- fixed across all figures
EST = ("S", "LW", "SURE-G", "SURE-DF", "Oracle")
COL = {"S": "0.55", "LW": "#E69F00", "SURE-G": "#0072B2", "SURE-DF": "#009E73", "Oracle": "black"}
MRK = {"S": "x", "LW": "s", "SURE-G": "o", "SURE-DF": "^", "Oracle": None}
LS = {"S": ":", "LW": "-", "SURE-G": "-", "SURE-DF": "-", "Oracle": "--"}
NAME = {"S": r"sample $S$", "LW": "LW", "SURE-G": "Gaussian SURE", "SURE-DF": "distribution-free SURE",
        "Oracle": r"oracle $\alpha^*$"}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "0.9", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "savefig.bbox": "tight"})


# ------------------------------------------------------------------ population covariances
def sigma_factor(N):
    """Single-factor (market) model: Sigma = sigma_m^2 beta beta' + D."""
    beta = np.linspace(0.5, 1.5, N)
    d = np.linspace(0.5, 1.5, N)[::-1]
    return 1.0 * np.outer(beta, beta) + np.diag(d)


def sigma_spread(N):
    """Eigenvalues spread linearly on [1, 10]. Both data models are elliptical and the risk is
    rotation invariant, so a diagonal Sigma loses no generality."""
    return np.diag(np.linspace(1.0, 10.0, N))


SIGMAS = {"factor": sigma_factor, "spread": sigma_spread}


def kurt_factor(dist):
    """E w^2 where x = sqrt(w) z, z ~ N(0, Sigma), E w = 1. E||x||^4 = E w^2 [(tr S)^2 + 2 tr S^2]."""
    if dist == "gauss":
        return 1.0
    nu = DF[dist]
    return (nu - 2) / (nu - 4)


def truth(Sigma, T, dist):
    """V = sum Var(s_ij), C = sum Cov(f_ij, s_ij), D = E||F - S||^2, alpha* = (V - C)/D."""
    N = Sigma.shape[0]
    tr, tr2 = np.trace(Sigma), np.sum(Sigma ** 2)
    m4 = kurt_factor(dist) * (tr ** 2 + 2 * tr2)
    V = (m4 - tr2) / T
    C = (m4 - tr ** 2) / (N * T)
    D = tr2 + V - tr ** 2 / N - C
    return V, C, D, (V - C) / D


def risk(a, V, C, D):
    return a ** 2 * D + (1 - 2 * a) * V + 2 * a * C


# ------------------------------------------------------------------ data
def draw(rng, R, T, L, dist):
    Z = rng.standard_normal((R, T, L.shape[0])) @ L.T
    if dist == "gauss":
        return Z
    nu = DF[dist]
    w = (nu - 2) / rng.chisquare(nu, size=(R, T))  # E w = 1, so Cov(x) = Sigma
    return Z * np.sqrt(w)[:, :, None]


def statistics(X, Sigma):
    """Per-replication statistics, vectorised over the first axis of X (R x T x N)."""
    R, T, N = X.shape
    S = np.matmul(X.transpose(0, 2, 1), X) / T
    trS = np.trace(S, axis1=1, axis2=2)
    trS2 = np.sum(S ** 2, axis=(1, 2))
    sq = np.sum(X ** 2, axis=2)                        # ||x_t||^2
    M = np.mean(sq ** 2, axis=1)                       # M-hat
    U2 = (T * trS2 - M) / (T - 1)                      # = 1/(T(T-1)) sum_{t!=u} (x_t'x_u)^2
    U3 = (T * trS ** 2 - M) / (T - 1)                  # = 1/(T(T-1)) sum_{t!=u} ||x_t||^2 ||x_u||^2
    d2 = trS2 - trS ** 2 / N                           # ||F - S||^2
    # Section 5 (distribution-free)
    V_df = (M - U2) / T
    C_df = (M - U3) / (N * T)
    # Section 4 (Gaussian)
    V_g = (T * trS ** 2 + (T - 2) * trS2) / ((T + 2) * (T - 1))
    b_g = T * (T * trS2 - trS ** 2) / ((T + 2) * (T - 1))
    C_g = 2 * b_g / (N * T)
    # loss pieces: Sigma_hat - Sigma = (1-a)(S - Sigma) + a(F - Sigma)
    tr, tr2 = np.trace(Sigma), np.sum(Sigma ** 2)
    SSig = np.einsum("rij,ij->r", S, Sigma)
    nu = trS / N
    A = trS2 - 2 * SSig + tr2                          # ||S - Sigma||^2
    B = N * nu ** 2 - 2 * nu * tr + tr2                # ||F - Sigma||^2
    Cr = nu * trS - SSig - nu * tr + tr2               # <S - Sigma, F - Sigma>
    return dict(d2=d2, V_df=V_df, C_df=C_df, V_g=V_g, C_g=C_g, A=A, B=B, Cr=Cr, T=T)


def alphas(st, a_star):
    T = st["T"]
    raw = {
        "S": np.zeros_like(st["d2"]),
        "LW": (T - 1) / T * st["V_df"] / st["d2"],
        "SURE-G": (st["V_g"] - st["C_g"]) / st["d2"],
        "SURE-DF": (st["V_df"] - st["C_df"]) / st["d2"],
        "Oracle": np.full_like(st["d2"], a_star),
    }
    return {k: np.clip(v, 0.0, 1.0) for k, v in raw.items()}, raw


def loss(a, st):
    return (1 - a) ** 2 * st["A"] + a ** 2 * st["B"] + 2 * a * (1 - a) * st["Cr"]


def run_cell(rng, Sigma, T, dist, reps=REPS):
    L = np.linalg.cholesky(Sigma)
    V, C, D, a_star = truth(Sigma, T, dist)
    a_star_c = min(max(a_star, 0.0), 1.0)
    keep = {k: [] for k in ("d2", "V_df", "C_df", "V_g", "C_g")}
    losses = {k: [] for k in EST}
    al = {k: [] for k in EST}
    n_clip = {k: 0 for k in EST}
    for r0 in range(0, reps, CHUNK):
        st = statistics(draw(rng, min(CHUNK, reps - r0), T, L, dist), Sigma)
        a, raw = alphas(st, a_star_c)
        for k in EST:
            losses[k].append(loss(a[k], st))
            al[k].append(a[k])
            n_clip[k] += int(np.sum((raw[k] < 0) | (raw[k] > 1)))
        for k in keep:
            keep[k].append(st[k])
    cat = lambda d: {k: np.concatenate(v) for k, v in d.items()}
    return dict(V=V, C=C, D=D, a_star=a_star, losses=cat(losses), alphas=cat(al), stats=cat(keep),
                n_clip=n_clip, reps=reps)


# ------------------------------------------------------------------ experiment A: full grid
def exp_grid(rng):
    res = {}
    for sname, sfun in SIGMAS.items():
        for dist in DISTS:
            for N in NS:
                Sigma = sfun(N)
                for T in TS:
                    res[(sname, dist, N, T)] = run_cell(rng, Sigma, T, dist)
    return res


def prial(cell, k):
    """Percentage relative improvement in average loss over S: 100 (E L(S) - E L(k)) / E L(S)."""
    l0 = cell["losses"]["S"].mean()
    return 100 * (l0 - cell["losses"][k].mean()) / l0


def table_lines(res):
    lines = []
    hdr = ("%-6s %-5s %4s %4s %6s | %7s %7s %7s %7s | %7s %13s %13s %13s | %6s %6s %6s %6s"
           % ("Sigma", "dist", "N", "T", "N/T", "PRIAL:LW", "SURE-G", "SURE-DF", "Oracle",
              "alpha*", "LW mean(sd)", "G mean(sd)", "DF mean(sd)",
              "Vg/V", "Vdf/V", "Cg/C", "Cdf/C"))
    lines.append(hdr)
    for (sname, dist, N, T), c in res.items():
        a = c["alphas"]
        ms = lambda k: "%.3f(%.3f)" % (a[k].mean(), a[k].std())
        st = c["stats"]
        lines.append("%-6s %-5s %4d %4d %6.2f | %7.1f %7.1f %7.1f %7.1f | %7.3f %13s %13s %13s | %6.3f %6.3f %6.3f %6.3f" % (
            sname, dist, N, T, N / T, prial(c, "LW"), prial(c, "SURE-G"), prial(c, "SURE-DF"), prial(c, "Oracle"),
            c["a_star"], ms("LW"), ms("SURE-G"), ms("SURE-DF"),
            st["V_g"].mean() / c["V"], st["V_df"].mean() / c["V"],
            st["C_g"].mean() / c["C"], st["C_df"].mean() / c["C"]))
    return lines


def check_lw_identity(rng, N=30, T=15):
    """LW (2004) intensity toward nu I, sum_t ||x_t x_t' - S||^2 / (T^2 ||F - S||^2), equals (T-1)/T * Vhat / ||F-S||^2."""
    X = draw(rng, 1, T, np.linalg.cholesky(sigma_spread(N)), "t10")
    st = statistics(X, sigma_spread(N))
    x = X[0]
    S = x.T @ x / T
    lw = sum(np.sum((np.outer(v, v) - S) ** 2) for v in x) / (T ** 2 * st["d2"][0])
    return "LW identity check: original formula %.12f ; (T-1)/T * Vhat/d2 = %.12f" % (
        lw, (T - 1) / T * st["V_df"][0] / st["d2"][0])


def latex_rows(res, N=100):
    """Rows for the LaTeX tables in thesis/sec6_simulations.tex (pasted from sec6_results.txt)."""
    lines = ["%% Table 1 rows (N = %d): Sigma & data & T & alpha* & mean alpha-hat LW, G, DF & PRIAL LW, G, DF, Oracle" % N]
    for sname in SIGMAS:
        for dist in DISTS:
            for T in TS:
                c = res[(sname, dist, N, T)]
                a = c["alphas"]
                lines.append("%s & %s & %d & %.3f & %.3f & %.3f & %.3f & %.1f & %.1f & %.1f & %.1f \\\\" % (
                    sname, dist, T, c["a_star"], a["LW"].mean(), a["SURE-G"].mean(), a["SURE-DF"].mean(),
                    prial(c, "LW"), prial(c, "SURE-G"), prial(c, "SURE-DF"), prial(c, "Oracle")))
    lines.append("%% Table 2 rows (N = %d, T = 20): Sigma & data & EVg/V & EVdf/V & ECg/C & ECdf/C" % N)
    for sname in SIGMAS:
        for dist in DISTS:
            c = res[(sname, dist, N, 20)]
            st = c["stats"]
            lines.append("%s & %s & %.2f & %.2f & %.2f & %.2f \\\\" % (
                sname, dist, st["V_g"].mean() / c["V"], st["V_df"].mean() / c["V"],
                st["C_g"].mean() / c["C"], st["C_df"].mean() / c["C"]))
    return lines


def fig_prial(res):
    """PRIAL against N/T. Points with the same N are joined (T = 250, 60, 20 from left to right)."""
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.4), sharex=True, sharey="row")
    for i, sname in enumerate(SIGMAS):
        for j, dist in enumerate(DISTS):
            ax = axes[i, j]
            for k in ("Oracle", "LW", "SURE-G", "SURE-DF"):
                for n_i, N in enumerate(NS):
                    x = [N / T for T in TS]
                    y = [prial(res[(sname, dist, N, T)], k) for T in TS]
                    ax.plot(x, y, color=COL[k], ls=LS[k], marker=MRK[k], ms=3.5, lw=1.2 if k != "Oracle" else 1.0,
                            label=NAME[k] if n_i == 0 else None, zorder=3 if k == "SURE-DF" else 2)
            ax.set_xscale("log")
            ax.axhline(0, color="0.6", lw=0.8)
            if i == 0:
                ax.set_title(LABEL_DIST[dist] + " data")
            if i == 1:
                ax.set_xlabel(r"$N/T$")
            if j == 0:
                ax.set_ylabel("PRIAL over $S$ (%)\n" + ("single-factor $\\Sigma$" if sname == "factor"
                                                        else "spread $\\Sigma$"))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "sec6_prial." + ext), dpi=200)
    plt.close(fig)


def fig_alpha(res, sname="factor"):
    """Mean +/- sd of (alpha_hat - alpha*) for each estimator, across N/T, single-factor Sigma."""
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5), sharey=True)
    offs = {"LW": 0.9, "SURE-G": 1.0, "SURE-DF": 1.1}
    for j, dist in enumerate(DISTS):
        ax = axes[j]
        for k in ("LW", "SURE-G", "SURE-DF"):
            for n_i, N in enumerate(NS):
                cells = [res[(sname, dist, N, T)] for T in TS]
                x = np.array([N / T for T in TS]) * offs[k]
                m = [np.mean(c["alphas"][k] - min(c["a_star"], 1)) for c in cells]
                s_ = [np.std(c["alphas"][k]) for c in cells]
                ax.errorbar(x, m, yerr=s_, color=COL[k], marker=MRK[k], ms=3.5, lw=1.0, capsize=0,
                            elinewidth=0.7, label=NAME[k] if n_i == 0 else None)
        ax.axhline(0, color="black", lw=0.9, ls="--")
        ax.set_xscale("log")
        ax.set_title(LABEL_DIST[dist] + " data")
        ax.set_xlabel(r"$N/T$")
    axes[0].set_ylabel(r"$\hat\alpha - \alpha^*$ (mean $\pm$ sd)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "sec6_alpha." + ext), dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ experiment B: risk curves
def exp_curves(rng, sname="factor", N=100, T=20, reps=20000):
    Sigma = SIGMAS[sname](N)
    grid = np.linspace(0, 1, 201)
    # t5 is left out here: with infinite 8th moments the Monte Carlo means of V-hat, C-hat are unstable.
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.7), sharey=False)
    lines = []
    for ax, dist in zip(axes, ("gauss", "t10")):
        c = run_cell(rng, Sigma, T, dist, reps=reps)
        st = c["stats"]
        R = risk(grid, c["V"], c["C"], c["D"])
        m = {k: st[k].mean() for k in st}
        sure_g = grid ** 2 * m["d2"] + (1 - 2 * grid) * m["V_g"] + 2 * grid * m["C_g"]
        sure_df = grid ** 2 * m["d2"] + (1 - 2 * grid) * m["V_df"] + 2 * grid * m["C_df"]
        ax.plot(grid, R, color=COL["Oracle"], ls=(0, (3, 2)), lw=1.1, label=r"true risk $R(\alpha)$", zorder=4)
        ax.plot(grid, sure_g, color=COL["SURE-G"], lw=1.5, zorder=3, label=r"mean Gaussian SURE$(\alpha)$")
        ax.plot(grid, sure_df, color=COL["SURE-DF"], lw=4.0, alpha=0.55, zorder=2, label=r"mean distribution-free SURE$(\alpha)$")
        ax.text(min(c["a_star"], 1) + 0.02, R[0] * 0.04, r"$\alpha^*$", color="0.35", fontsize=8)
        ax.axvline(min(c["a_star"], 1), color="0.5", lw=0.8, ls=":")
        ax.set_title(LABEL_DIST[dist] + " data")
        ax.set_xlabel(r"shrinkage intensity $\alpha$")
        ax.set_ylim(0, R[0] * 1.15)
        gap = lambda s: np.max(np.abs(s - R)) / R[0]
        se_v = lambda k: st[k].std() / np.sqrt(reps)
        lines.append("curves %-6s %-5s N=%d T=%d reps=%d: alpha*=%.4f R(0)=%.2f R(alpha*)=%.2f | "
                     "max|meanSURE_G-R|/R(0)=%.4f  max|meanSURE_DF-R|/R(0)=%.4f | "
                     "MC mean loss at alpha*=%.2f | V=%.2f meanVg=%.2f(se %.2f) meanVdf=%.2f(se %.2f) | C=%.4f meanCg=%.4f meanCdf=%.4f(se %.4f)"
                     % (sname, dist, N, T, reps, c["a_star"], R[0], risk(c["a_star"], c["V"], c["C"], c["D"]),
                        gap(sure_g), gap(sure_df), c["losses"]["Oracle"].mean(), c["V"], m["V_g"], se_v("V_g"), m["V_df"], se_v("V_df"),
                        c["C"], m["C_g"], m["C_df"], se_v("C_df")))
    axes[0].set_ylabel("Frobenius risk")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "sec6_sure_curves." + ext), dpi=200)
    plt.close(fig)
    return lines


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    out = ["Section 6 simulations, seed %d, %d reps per grid cell, alpha clipped to [0,1]" % (SEED, REPS),
           "PRIAL = 100 (E L(S) - E L(est)) / E L(S); Oracle uses the fixed alpha* = (V-C)/D (clipped).",
           "Vg/V etc. = Monte Carlo mean of the estimate divided by the true value (1 = unbiased).", ""]
    out += [check_lw_identity(np.random.default_rng(1)), ""]
    out += exp_curves(rng)
    out.append("")
    res = exp_grid(rng)
    out += table_lines(res)
    clip = {k: sum(c["n_clip"][k] for c in res.values()) for k in ("LW", "SURE-G", "SURE-DF")}
    out += ["", "clipped alphas (out of %d): %s" % (REPS * len(res), clip)]
    out += [""] + latex_rows(res)
    fig_prial(res)
    fig_alpha(res)
    out.append("runtime %.1f s" % (time.time() - t0))
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(OUT, "sec6_results.txt"), "w") as f:
        f.write(txt + "\n")
