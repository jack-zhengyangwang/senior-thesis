# Goal 2: SURE for the single-index target without independence across days

Working notes, 2026-11-11. Each step is marked by whether it needs i.i.d. days.

## Model (replaces i.i.d.)
- Market series x0 (T×1) ~ N(0, σ00·A); stocks x_i = b_i·x0 + e_i.
- Residuals: Cov(e_i, e_j) = Ψ_ij·A (T×T), independent of x0 (strict exogeneity).
- A is a KNOWN T×T correlation matrix (unit diagonal), shared by every series (separable: A ⊗ Σ).
- Known zero mean. i.i.d. is the special case A = I.

## Step-by-step

| Step | Needs i.i.d.? | Result |
|---|---|---|
| 0. Yi identity, E S = Σ | No (only identical daily covariance: A_tt = 1) | unchanged |
| 1. Var(s_ij) | No (Gaussian + separable) | (σ_ij² + σ_iiσ_jj)·tr(A²)/T²; estimator with T_eff = T²/tr(A²) unbiased (MC) |
| 2. s_ij = f_ij + ψ̂_ij | No (algebra) | unchanged |
| 3. Cov(f, ψ̂) = 0 | YES (IK Lemma A.2 needs Wishart = i.i.d. rows) | fails unless H A (I−H) = 0 (Kruskal 1968: market series spans an invariant subspace of A) |
| 4. Cov(f_ij, s_ij) closed form | No | see below; MC verified (sims/check_goal2_cov_formula.py) |
| 5. Unbiased estimation | **STOP: uses independence** | see "Where it stops" |

Step 4 formula (H = x0 x0ᵀ / x0ᵀx0, c = x0ᵀx0/T, q = x0ᵀA x0/x0ᵀx0, q2 = x0ᵀA² x0/x0ᵀx0):

Cov(f_ij, s_ij) = [σ00·Ac + 2 b_i² b_j² σ00²]·tr(A²)/T² + B·E[q2]/T² + b_i b_j Ψ_ij·Cov(c, q)/T

with Ac = b_i²Ψ_jj + b_j²Ψ_ii + 2 b_i b_j Ψ_ij and B = Ψ_iiΨ_jj + Ψ_ij². At A = I it reduces to Theorem 1.

Derivation: condition on x0. Then f = (1/T)[b_ib_j x0ᵀx0 + b_i x0ᵀe_j + b_j x0ᵀe_i + e_iᵀH e_j], ψ̂ = (1/T) e_iᵀ(I−H)e_j.
- E[f|x0] = b_ib_j c + Ψ_ij q/T;  E[ψ̂|x0] = Ψ_ij (T − q)/T.
- Var(f|x0) = Ac·x0ᵀA x0/T² + B·q²/T²  (tr(HAHA) = q²).
- Cov(f, ψ̂|x0) = B·(q2 − q²)/T²  (tr(HA(I−H)A) = q2 − q²).
- Law of total covariance, E[x0ᵀA x0] = σ00 tr(A²), Var(c) = 2σ00² tr(A²)/T²; the Var(q) terms cancel.

## Where it stops (Step 5)
Goal 1's unbiased estimators used three i.i.d. facts that now fail:
1. Sample betas and residual covariance independent given the market (IK Lemma A.2): now He and (I−H)e are correlated given x0.
2. Residual covariance is Wishart with T−1 df: now a quadratic form with matrix (I−H)A(I−H).
3. Market variance c is chi-square with T df: now a weighted chi-square (eigenvalues of A).

Replacements (to do): all three are moments of quadratic forms in Gaussians with KNOWN matrices given x0 (Magnus 1978), so conditional unbiased estimators should exist. E[q2] depends only on A (scale-free) and is a computable constant; Cov(c, q) = σ00·κ(A) with κ(A) computable.
