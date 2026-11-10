"""Stein's normal-means problem: SURE vs true risk for James-Stein and linear shrinkage.

X ~ N(theta, I_p). Writes sims/out/fig_stein.pdf/png.
Run:  python3 sims/stein_js.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261026)
p, reps = 10, 20000

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# left: risk of MLE vs James-Stein as ||theta|| grows, with mean SURE of JS
norms = np.linspace(0, 8, 33)
js_risk, js_sure = [], []
for r in norms:
    theta = np.zeros(p)
    theta[0] = r
    X = theta + rng.standard_normal((reps, p))
    n2 = np.sum(X ** 2, axis=1)
    js = (1 - (p - 2) / n2)[:, None] * X
    js_risk.append(np.mean(np.sum((js - theta) ** 2, axis=1)))
    js_sure.append(np.mean(p - (p - 2) ** 2 / n2))
ax = axes[0]
ax.axhline(p, color="k", ls="--", lw=1.4, label="MLE  θ̂ = X  (risk = p)")
ax.plot(norms, js_risk, "C3-", lw=2, label="James–Stein, Monte Carlo risk")
ax.plot(norms, js_sure, "C0:", lw=2, label="James–Stein, mean SURE")
ax.set_xlabel("‖θ‖")
ax.set_ylabel("risk  E‖θ̂ − θ‖²")
ax.set_title("p = %d: James–Stein beats X for every θ" % p)
ax.set_ylim(0, p * 1.15)
ax.legend(fontsize=8, loc="lower right")

# right: SURE(lambda) for linear shrinkage (1-lambda)X at one theta
theta = np.full(p, 0.6)
grid = np.linspace(0, 1, 101)
true = (1 - grid) ** 2 * p + grid ** 2 * np.sum(theta ** 2)
ax = axes[1]
acc = np.zeros_like(grid)
for k in range(reps):
    X = theta + rng.standard_normal(p)
    sure = p + grid ** 2 * np.sum(X ** 2) - 2 * p * grid
    acc += sure
    if k < 25:
        ax.plot(grid, sure, color="0.75", lw=0.7)
ax.plot(grid, acc / reps, "C0", lw=2, label="mean SURE(λ)")
ax.plot(grid, true, "k--", lw=1.6, label="true risk R(λ)")
lam = p / (p + np.sum(theta ** 2))
ax.axvline(lam, color="C2", lw=1, label="oracle λ* = %.2f" % lam)
ax.set_xlabel("shrinkage λ   (θ̂ = (1−λ)X)")
ax.set_ylabel("risk")
ax.set_title("Linear shrinkage toward 0, ‖θ‖² = %.1f" % np.sum(theta ** 2))
ax.legend(fontsize=8, loc="upper right")
ax.text(0.02, 0.02, "grey: SURE(λ) from single draws", transform=ax.transAxes, fontsize=7.5, color="0.4")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_stein.pdf"))
fig.savefig(os.path.join(OUT, "fig_stein.png"), dpi=150)
print("JS risk at ||theta||=0: %.3f (theory 2.0), at 8: %.3f" % (js_risk[0], js_risk[-1]))
