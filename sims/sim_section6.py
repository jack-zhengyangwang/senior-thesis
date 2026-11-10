"""Section 6 simulations: Gaussian SURE (Section 4) vs distribution-free SURE (Section 5) vs Ledoit-Wolf.

Setting: x_1..x_T iid, mean zero, Cov = Sigma; S = (1/T) sum_t x_t x_t'; target F = nu I, nu = tr(S)/N;
estimator Sigma_hat = alpha F + (1 - alpha) S; loss = squared Frobenius norm.

PRIAL is computed as 100 * (V - E L(est)) / V, where V = E L(S) is known in closed form and E L(est) is
estimated with L(S) as a control variate (its exact mean V is known), which tames the t5 tails. Differences between methods are paired (same
replications for every method); the oracle PRIAL is exact: 100 * (V - R(alpha*)) / V = 100 (V - C)^2 / (D V) when alpha* lies in [0, 1].

Every experiment / cell has its own generator default_rng([SEED, ...]), so adding or changing one
experiment does not shift the numbers of another.

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
REPS = 5000            # replications per run
RUNS = {"gauss": 1, "t10": 1, "t5": 10}   # t5: 10 independent runs of REPS, to measure run-to-run spread
CHUNK = 250
NS = (10, 50, 100)
TS = (20, 60, 250)
DISTS = ("gauss", "t10", "t5")
DF = {"t10": 10, "t5": 5}
LABEL_DIST = {"gauss": "Gaussian", "t10": r"$t_{10}$", "t5": r"$t_5$"}
KEY_SIGMA = {"factor": 1, "spread": 2}
KEY_DIST = {"gauss": 1, "t10": 2, "t5": 3}

# estimator order, colors (Okabe-Ito), markers -- fixed across all figures
EST = ("S", "LW", "SURE-G", "SURE-DF", "Oracle")
COL = {"S": "0.55", "LW": "#E69F00", "SURE-G": "#0072B2", "SURE-DF": "#009E73", "Oracle": "black"}
NAME = {"S": r"sample $S$", "LW": "LW", "SURE-G": "Gaussian SURE", "SURE-DF": "distribution-free SURE",
        "Oracle": r"oracle $\alpha^*$"}
# nested markers so that overlapping estimators stay visible: big hollow square (LW) > hollow circle (G)
# > small filled triangle (DF)
STY = {
    "LW": dict(color=COL["LW"], ls=(0, (4, 1.5)), lw=1.1, marker="s", ms=6.5, mfc="none", mew=1.1, zorder=2),
    "SURE-G": dict(color=COL["SURE-G"], ls="-", lw=1.1, marker="o", ms=4.8, mfc="none", mew=1.1, zorder=3),
    "SURE-DF": dict(color=COL["SURE-DF"], ls="-", lw=0.9, marker="^", ms=3.2, zorder=4),
    "Oracle": dict(color=COL["Oracle"], ls=(0, (2, 1.5)), lw=0.9, marker=None, zorder=1),
}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "0.9", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "savefig.bbox": "tight"})


# ------------------------------------------------------------------ population covariances
def sigma_factor(N):
    """Single-factor (market) model: Sigma = sigma_m^2 beta beta' + diag(d), sigma_m^2 = 1,
    beta_i and d_i both equally spaced on [0.5, 1.5] in the same order."""
    beta = np.linspace(0.5, 1.5, N)
    d = np.linspace(0.5, 1.5, N)
    return np.outer(beta, beta) + np.diag(d)


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
    """V = sum Var(s_ij) = E L(S), C = sum Cov(f_ij, s_ij), D = E||F - S||^2, alpha* = (V - C)/D."""
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
    return dict(V=V, C=C, D=D, a_star=a_star, a_star_c=a_star_c, losses=cat(losses), alphas=cat(al),
                stats=cat(keep), n_clip=n_clip, reps=reps)


def merge(cells):
    """Pool several runs of the same cell; remember the run boundaries."""
    c = dict(cells[0])
    for f in ("losses", "alphas", "stats"):
        c[f] = {k: np.concatenate([x[f][k] for x in cells]) for k in cells[0][f]}
    c["n_clip"] = {k: sum(x["n_clip"][k] for x in cells) for k in cells[0]["n_clip"]}
    c["reps"] = sum(x["reps"] for x in cells)
    c["runs"] = cells
    return c


# ------------------------------------------------------------------ PRIAL and Monte Carlo errors
def prial_exact_oracle(c):
    return 100 * (c["V"] - risk(c["a_star_c"], c["V"], c["C"], c["D"])) / c["V"]


def _cv(x, c):
    """Control-variate estimate of E[x] using L(S), whose mean V is known exactly:
    mean(x) - beta (mean L(S) - V), beta = regression slope of x on L(S). Returns (estimate, SE)."""
    L0 = c["losses"]["S"]
    beta = np.cov(x, L0)[0, 1] / L0.var(ddof=1)
    resid = x - beta * L0
    return x.mean() - beta * (L0.mean() - c["V"]), resid.std(ddof=1) / np.sqrt(x.size)


def prial(c, k):
    """100 * (V - E L(k)) / V with the exact V = E L(S); E L(k) by the control-variate estimate.
    Oracle: exact. (Under t5 the plain MC mean of a loss is dominated by rare samples; the control variate
    removes most of that noise because those samples have huge L(S) as well.)"""
    if k == "Oracle":
        return prial_exact_oracle(c)
    return 100 * (c["V"] - _cv(c["losses"][k], c)[0]) / c["V"]


def prial_se(c, k):
    return 100 * _cv(c["losses"][k], c)[1] / c["V"]


def diff_se(c, k1, k2):
    """SE of PRIAL(k1) - PRIAL(k2), paired over the same replications."""
    return 100 * _cv(c["losses"][k2] - c["losses"][k1], c)[1] / c["V"]


def run_sd(c, k):
    """Standard deviation of single-run PRIALs across the independent runs (t5 only)."""
    if len(c.get("runs", [])) < 2:
        return np.nan
    return np.std([prial(r, k) for r in c["runs"]], ddof=1)


def run_range(c, k):
    v = [prial(r, k) for r in c["runs"]]
    return min(v), max(v)


# ------------------------------------------------------------------ experiment A: full grid
def exp_grid():
    res = {}
    for sname, sfun in SIGMAS.items():
        for dist in DISTS:
            for N in NS:
                Sigma = sfun(N)
                for T in TS:
                    runs = [run_cell(np.random.default_rng([SEED, 1, KEY_SIGMA[sname], KEY_DIST[dist], N, T, b]),
                                     Sigma, T, dist) for b in range(RUNS[dist])]
                    res[(sname, dist, N, T)] = merge(runs)
    return res


def table_lines(res):
    lines = ["Columns: PRIAL (%%) for LW, SURE-G, SURE-DF, exact Oracle | paired MC SEs: SE(level, max over "
             "LW/G/DF), SE(DF-G), SE(DF-LW) | for t5: single-run SD of PRIAL (LW, G, DF) over %d runs of %d |"
             % (RUNS["t5"], REPS),
             "alpha* and mean(sd) of alpha-hat | bias ratios mean(est)/true", ""]
    hdr = ("%-6s %-5s %4s %4s %6s %6s | %6s %6s %6s %6s | %5s %5s %5s | %5s %5s %5s | %6s %13s %13s %13s | %5s %5s %5s %5s"
           % ("Sigma", "dist", "N", "T", "N/T", "reps", "LW", "G", "DF", "Orac", "SElev", "SEdG", "SEdL",
              "sdLW", "sdG", "sdDF", "alpha*", "LW mean(sd)", "G mean(sd)", "DF mean(sd)",
              "Vg/V", "Vd/V", "Cg/C", "Cd/C"))
    lines.append(hdr)
    for (sname, dist, N, T), c in res.items():
        a = c["alphas"]
        ms = lambda k: "%.3f(%.3f)" % (a[k].mean(), a[k].std())
        st = c["stats"]
        lines.append("%-6s %-5s %4d %4d %6.2f %6d | %6.1f %6.1f %6.1f %6.1f | %5.2f %5.2f %5.2f | %5.1f %5.1f %5.1f | "
                     "%6.3f %13s %13s %13s | %5.2f %5.2f %5.2f %5.2f" % (
                         sname, dist, N, T, N / T, c["reps"],
                         prial(c, "LW"), prial(c, "SURE-G"), prial(c, "SURE-DF"), prial(c, "Oracle"),
                         max(prial_se(c, k) for k in ("LW", "SURE-G", "SURE-DF")),
                         diff_se(c, "SURE-DF", "SURE-G"), diff_se(c, "SURE-DF", "LW"),
                         run_sd(c, "LW"), run_sd(c, "SURE-G"), run_sd(c, "SURE-DF"),
                         c["a_star"], ms("LW"), ms("SURE-G"), ms("SURE-DF"),
                         st["V_g"].mean() / c["V"], st["V_df"].mean() / c["V"],
                         st["C_g"].mean() / c["C"], st["C_df"].mean() / c["C"]))
    return lines


def latex_rows(res, N=100):
    """Rows for the LaTeX tables in thesis/sec6_simulations.tex (pasted from sec6_results.txt)."""
    lab_s = {"factor": "single-factor", "spread": "spread"}
    lab_d = {"gauss": "Gaussian", "t10": "$t_{10}$", "t5": "$t_5$"}
    lines = ["%% Table rows (N = %d): Sigma & data & T & alpha* & mean alpha-hat LW, G, DF & & PRIAL LW, G, DF, Oracle" % N]
    for sname in SIGMAS:
        for dist in DISTS:
            for T in TS:
                c = res[(sname, dist, N, T)]
                a = c["alphas"]
                first_s = dist == "gauss" and T == TS[0]
                lines.append("%s & %s & %d & %.3f & %.3f & %.3f & %.3f & & %.1f & %.1f & %.1f & %.1f \\\\" % (
                    lab_s[sname] if first_s else "", lab_d[dist] if T == TS[0] else "", T, c["a_star"],
                    a["LW"].mean(), a["SURE-G"].mean(), a["SURE-DF"].mean(),
                    prial(c, "LW"), prial(c, "SURE-G"), prial(c, "SURE-DF"), prial(c, "Oracle")))
        if sname == "factor":
            lines.append("\\midrule")
    lines.append("%% Bias table rows (N = %d, T = 20): Sigma & data & EVg/V & EVdf/V & ECg/C & ECdf/C" % N)
    for sname in SIGMAS:
        for dist in DISTS:
            c = res[(sname, dist, N, 20)]
            st = c["stats"]
            lines.append("%s & %s & %.2f & %.2f & %.2f & %.2f \\\\" % (
                lab_s[sname] if dist == "gauss" else "", lab_d[dist], st["V_g"].mean() / c["V"],
                st["V_df"].mean() / c["V"], st["C_g"].mean() / c["C"], st["C_df"].mean() / c["C"]))
        if sname == "factor":
            lines.append("\\midrule")
    return lines


def summary_lines(res):
    """Every summary number quoted in the prose of thesis/sec6_simulations.tex."""
    out = ["SUMMARY (numbers quoted in the text)"]
    cells = lambda s=None, d=None: [(k, c) for k, c in res.items() if (s is None or k[0] == s) and (d is None or k[1] == d)]
    g = cells(d="gauss")
    out.append("gauss: max over cells of (max - min) PRIAL among LW/G/DF = %.2f" %
               max(max(prial(c, k) for k in ("LW", "SURE-G", "SURE-DF")) - min(prial(c, k) for k in ("LW", "SURE-G", "SURE-DF")) for _, c in g))
    out.append("gauss: PRIAL level SE range = %.2f-%.2f; SE(DF-G) range = %.2f-%.2f" % (
        min(prial_se(c, k) for _, c in g for k in ("LW", "SURE-G", "SURE-DF")),
        max(prial_se(c, k) for _, c in g for k in ("LW", "SURE-G", "SURE-DF")),
        min(diff_se(c, "SURE-DF", "SURE-G") for _, c in g), max(diff_se(c, "SURE-DF", "SURE-G") for _, c in g)))
    t = cells(d="t10")
    out.append("t10: PRIAL level SE range = %.2f-%.2f; SE(DF-G) range = %.2f-%.2f; SE(DF-LW) range = %.2f-%.2f" % (
        min(prial_se(c, k) for _, c in t for k in ("LW", "SURE-G", "SURE-DF")),
        max(prial_se(c, k) for _, c in t for k in ("LW", "SURE-G", "SURE-DF")),
        min(diff_se(c, "SURE-DF", "SURE-G") for _, c in t), max(diff_se(c, "SURE-DF", "SURE-G") for _, c in t),
        min(diff_se(c, "SURE-DF", "LW") for _, c in t), max(diff_se(c, "SURE-DF", "LW") for _, c in t)))
    for s in SIGMAS:
        cs = cells(s, "t10")
        rel = [1 - c["alphas"]["SURE-G"].mean() / c["a_star"] for _, c in cs]
        dfd = [abs(c["alphas"]["SURE-DF"].mean() - c["a_star"]) for _, c in cs]
        gain = [prial(c, "SURE-DF") - prial(c, "SURE-G") for _, c in cs]
        zs = [(prial(c, "SURE-DF") - prial(c, "SURE-G")) / diff_se(c, "SURE-DF", "SURE-G") for _, c in cs]
        out.append("t10 %s: G below alpha* by %.0f%%-%.0f%%; |mean DF - alpha*| max %.3f; DF-G PRIAL %.1f-%.1f (min z %.1f)"
                   % (s, 100 * min(rel), 100 * max(rel), max(dfd), min(gain), max(gain), min(zs)))
    t5 = cells(d="t5")
    out.append("t5: single-run PRIAL SD over runs: LW %.1f-%.1f, G %.1f-%.1f, DF %.1f-%.1f" % (
        min(run_sd(c, "LW") for _, c in t5), max(run_sd(c, "LW") for _, c in t5),
        min(run_sd(c, "SURE-G") for _, c in t5), max(run_sd(c, "SURE-G") for _, c in t5),
        min(run_sd(c, "SURE-DF") for _, c in t5), max(run_sd(c, "SURE-DF") for _, c in t5)))
    out.append("t5: min over cells of [min(LW,DF) - G] PRIAL = %.1f; G/oracle PRIAL ratio range %.2f-%.2f" % (
        min(min(prial(c, "LW"), prial(c, "SURE-DF")) - prial(c, "SURE-G") for _, c in t5),
        min(prial(c, "SURE-G") / prial(c, "Oracle") for _, c in t5),
        max(prial(c, "SURE-G") / prial(c, "Oracle") for _, c in t5)))
    for k, c in t5:
        lo, hi = run_range(c, "LW")
        out.append("  t5 %-6s N=%3d T=%3d: LW %.1f (runs %.1f-%.1f, SE %.1f)  DF %.1f  G %.1f  Oracle(exact) %.1f  -> LW-Oracle %+.1f"
                   % (k[0], k[2], k[3], prial(c, "LW"), lo, hi, run_sd(c, "LW") / np.sqrt(len(c["runs"])),
                      prial(c, "SURE-DF"), prial(c, "SURE-G"), prial(c, "Oracle"), prial(c, "LW") - prial(c, "Oracle")))
    out.append("cells where LW or DF beats exact oracle PRIAL (by > 2 SE of the level):")
    for k, c in res.items():
        for e in ("LW", "SURE-DF"):
            se = run_sd(c, e) / np.sqrt(len(c["runs"])) if k[1] == "t5" else prial_se(c, e)
            if prial(c, e) - prial(c, "Oracle") > 2 * se:
                out.append("  %s %s N=%d T=%d %s: %.2f vs oracle %.2f (SE %.2f)" % (k[0], k[1], k[2], k[3], e, prial(c, e), prial(c, "Oracle"), se))
    out.append("max over all cells |PRIAL(LW) - PRIAL(DF)| = %.2f" %
               max(abs(prial(c, "LW") - prial(c, "SURE-DF")) for c in res.values()))
    # factor gauss: oracle vs data-driven
    for k, c in cells("factor", "gauss"):
        if k[3] == 20:
            out.append("factor gauss N=%d T=20: oracle %.1f, LW %.1f, G %.1f, DF %.1f; sd alpha G %.3f DF %.3f; alpha* %.3f mean DF %.3f"
                       % (k[2], prial(c, "Oracle"), prial(c, "LW"), prial(c, "SURE-G"), prial(c, "SURE-DF"),
                          c["alphas"]["SURE-G"].std(), c["alphas"]["SURE-DF"].std(), c["a_star"], c["alphas"]["SURE-DF"].mean()))
    for k, c in cells("factor", "t10"):
        if k[3] == 20:
            out.append("factor t10 N=%d T=20: oracle %.1f, LW %.1f, G %.1f, DF %.1f" % (
                k[2], prial(c, "Oracle"), prial(c, "LW"), prial(c, "SURE-G"), prial(c, "SURE-DF")))
    # C-hat vs V-hat/T, to check the order of the LW-vs-SURE difference
    rat = [(k, np.mean(c["stats"]["C_df"]) / (np.mean(c["stats"]["V_df"]) / k[3])) for k, c in res.items()]
    out.append("E[C-hat] / (E[V-hat]/T): range %.2f-%.2f (gauss N=10,T=250 factor: %.2f)" % (
        min(r for _, r in rat), max(r for _, r in rat), dict(rat)[("factor", "gauss", 10, 250)]))
    tot = sum(c["reps"] for c in res.values())
    clip = {k: sum(c["n_clip"][k] for c in res.values()) for k in ("LW", "SURE-G", "SURE-DF")}
    out.append("clipped alphas out of %d draws: %s -> %s" % (
        tot, clip, {k: "%.2f%%" % (100 * v / tot) for k, v in clip.items()}))
    return out


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
                    ax.plot(x, y, label=NAME[k] if n_i == 0 else None, **STY[k])
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
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "sec6_prial." + ext), dpi=200)
    plt.close(fig)


def fig_alpha(res, sname="factor"):
    """Mean of (alpha_hat - alpha*) +/- sd of alpha_hat, for each estimator, across N/T."""
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.4), sharey=True)
    offs = {"LW": 0.9, "SURE-G": 1.0, "SURE-DF": 1.1}
    for j, dist in enumerate(DISTS):
        ax = axes[j]
        for k in ("LW", "SURE-G", "SURE-DF"):
            st = {kk: v for kk, v in STY[k].items() if kk not in ("ls",)}
            st["ms"] = {"LW": 4.5, "SURE-G": 4.0, "SURE-DF": 3.5}[k]
            for n_i, N in enumerate(NS):
                cells = [res[(sname, dist, N, T)] for T in TS]
                x = np.array([N / T for T in TS]) * offs[k]
                m = [np.mean(c["alphas"][k] - c["a_star_c"]) for c in cells]
                s_ = [np.std(c["alphas"][k]) for c in cells]
                ax.errorbar(x, m, yerr=s_, capsize=0, elinewidth=0.7, ls=STY[k]["ls"],
                            label=NAME[k] if n_i == 0 else None, **st)
        ax.axhline(0, color="black", lw=0.9, ls="--")
        ax.set_xscale("log")
        ax.set_title(LABEL_DIST[dist] + " data")
        ax.set_xlabel(r"$N/T$")
    axes[0].set_ylabel(r"$\hat\alpha - \alpha^*$ (mean $\pm$ sd)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "sec6_alpha." + ext), dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ experiment B: risk curves
def exp_curves(sname="factor", N=100, T=20, reps=20000):
    Sigma = SIGMAS[sname](N)
    grid = np.linspace(0, 0.6, 121)
    # t5 is left out here: with infinite 8th moments the Monte Carlo means of V-hat, C-hat are unstable.
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.7), sharey=False)
    lines = []
    for ax, dist in zip(axes, ("gauss", "t10")):
        c = run_cell(np.random.default_rng([SEED, 2, KEY_DIST[dist]]), Sigma, T, dist, reps=reps)
        st = c["stats"]
        R = risk(grid, c["V"], c["C"], c["D"])
        m = {k: st[k].mean() for k in st}
        sure_g = grid ** 2 * m["d2"] + (1 - 2 * grid) * m["V_g"] + 2 * grid * m["C_g"]
        sure_df = grid ** 2 * m["d2"] + (1 - 2 * grid) * m["V_df"] + 2 * grid * m["C_df"]
        ax.plot(grid, R, color=COL["Oracle"], ls=(0, (3, 2)), lw=1.1, label=r"true risk $R(\alpha)$", zorder=4)
        ax.plot(grid, sure_g, color=COL["SURE-G"], lw=1.5, zorder=3, label=r"mean Gaussian SURE$(\alpha)$")
        ax.plot(grid, sure_df, color=COL["SURE-DF"], lw=4.0, alpha=0.55, zorder=2,
                label=r"mean distribution-free SURE$(\alpha)$")
        ax.axvline(c["a_star_c"], color="0.5", lw=0.8, ls=":")
        ax.text(c["a_star_c"] + 0.012, R[0] * 0.04, r"$\alpha^*$", color="0.35", fontsize=8)
        ax.set_title(LABEL_DIST[dist] + " data")
        ax.set_xlabel(r"shrinkage intensity $\alpha$")
        ax.set_xlim(0, 0.6)
        ax.set_ylim(0, R[0] * 1.15)
        full = np.linspace(0, 1, 201)
        Rf = risk(full, c["V"], c["C"], c["D"])
        gap = lambda mv, mc: np.max(np.abs(full ** 2 * m["d2"] + (1 - 2 * full) * mv + 2 * full * mc - Rf)) / Rf[0]
        se_v = lambda k: st[k].std() / np.sqrt(reps)
        lines.append("curves %-6s %-5s N=%d T=%d reps=%d: alpha*=%.4f R(0)=%.2f R(alpha*)=%.2f | "
                     "max over alpha in [0,1] |meanSURE_G-R|/R(0)=%.4f  |meanSURE_DF-R|/R(0)=%.4f | "
                     "MC mean loss at alpha*=%.2f | V=%.2f meanVg=%.2f(se %.2f) meanVdf=%.2f(se %.2f) | "
                     "C=%.4f meanCg=%.4f meanCdf=%.4f(se %.4f)"
                     % (sname, dist, N, T, reps, c["a_star"], R[0], risk(c["a_star"], c["V"], c["C"], c["D"]),
                        gap(m["V_g"], m["C_g"]), gap(m["V_df"], m["C_df"]), c["losses"]["Oracle"].mean(),
                        c["V"], m["V_g"], se_v("V_g"), m["V_df"], se_v("V_df"),
                        c["C"], m["C_g"], m["C_df"], se_v("C_df")))
    axes[0].set_ylabel("Frobenius risk")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "sec6_sure_curves." + ext), dpi=200)
    plt.close(fig)
    return lines


# ------------------------------------------------------------------ experiment C: why LW beats the oracle under t5
def exp_mechanism(sname="factor", N=100, T=20, reps=200000):
    Sigma = SIGMAS[sname](N)
    c = run_cell(np.random.default_rng([SEED, 3]), Sigma, T, "t5", reps=reps)
    L = c["losses"]
    Rstar = risk(c["a_star_c"], c["V"], c["C"], c["D"])
    d = L["LW"] - L["Oracle"]
    gain = L["S"] - L["LW"]
    order = np.argsort(L["S"])
    top = order[-int(0.001 * reps):]
    bottom = order[: reps // 2]
    aLW = c["alphas"]["LW"]
    return ["mechanism %s t5 N=%d T=%d reps=%d: E[L(LW)-L(oracle)]/R* = %.3f (SE %.3f); "
            "share of LW gain sum(L(S)-L(LW)) from top 0.1%% of L(S) = %.2f; "
            "mean alpha_LW top 0.1%% = %.2f, bottom half = %.2f, all = %.2f; alpha* = %.3f; "
            "MC mean L(S)/V = %.3f"
            % (sname, N, T, reps, d.mean() / Rstar, d.std() / np.sqrt(reps) / Rstar,
               gain[top].sum() / gain.sum(), aLW[top].mean(), aLW[bottom].mean(), aLW.mean(), c["a_star"],
               L["S"].mean() / c["V"])]


def check_lw_identity(rng, N=30, T=15):
    """LW (2004) intensity toward nu I, sum_t ||x_t x_t' - S||^2 / (T^2 ||F - S||^2), equals (T-1)/T * Vhat / ||F-S||^2."""
    X = draw(rng, 1, T, np.linalg.cholesky(sigma_spread(N)), "t10")
    st = statistics(X, sigma_spread(N))
    x = X[0]
    S = x.T @ x / T
    lw = sum(np.sum((np.outer(v, v) - S) ** 2) for v in x) / (T ** 2 * st["d2"][0])
    return "LW identity check: original formula %.12f ; (T-1)/T * Vhat/d2 = %.12f" % (
        lw, (T - 1) / T * st["V_df"][0] / st["d2"][0])


if __name__ == "__main__":
    t0 = time.time()
    out = ["Section 6 simulations, seed %d; %d reps per run; runs per cell %s; alpha clipped to [0,1]" % (SEED, REPS, RUNS),
           "PRIAL = 100 (V - E L(est)) / V, V = E L(S) exact, E L(est) by control variate L(S); Oracle PRIAL exact = 100 (V - R(alpha*)) / V.",
           "SEs: paired over the same replications. t5: the reported PRIAL pools all runs; sd* = SD of single-run PRIALs.",
           "Vg/V etc. = Monte Carlo mean of the estimate divided by the true value (1 = unbiased).", ""]
    out += [check_lw_identity(np.random.default_rng([SEED, 0])), ""]
    out += exp_curves()
    out += exp_mechanism()
    out.append("")
    res = exp_grid()
    out += table_lines(res)
    out += [""] + summary_lines(res)
    out += [""] + latex_rows(res)
    fig_prial(res)
    fig_alpha(res)
    out.append("runtime %.1f s" % (time.time() - t0))
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(OUT, "sec6_results.txt"), "w") as f:
        f.write(txt + "\n")
