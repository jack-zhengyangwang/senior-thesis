# Local literature check: SURE-tuned LW shrinkage toward a single-index target

Folder: `/Users/jackscomputer/dev/senior-thesis/papers/` (16 PDFs, all read; Haff 1980 is a scanned image and was read as page images).
Method: abstract, intro, main theorems, conclusion for each; full read of the shrinkage-relevant ones (LW2003, LW2004, Chen et al., Touloumis, Ollila–Raninen, Yi–Zou, Haff 1979).

Page conventions: journal page numbers where the PDF carries them; for arXiv or tech-report versions (Chen et al., Ollila–Raninen, Touloumis, Himeno–Yamada, Yi–Zou tech report) the page is the PDF page. [inferred] marks my reading or interpretation rather than a statement in the paper. [suggestion] marks my own idea, which is not in any paper here.

Notation follows Jack's: N assets, T days, S sample covariance, F target, R(α) = E‖αF + (1−α)S − Σ‖².

---

## 1. Per-paper notes

### Ledoit & Wolf (2003), J. Empirical Finance 10, 603–621. Sharpe single-index target
1. Target: Sharpe single-index F = s₀₀² b bᵀ + D (p.607). Off-diagonal f_ij = s_i0 s_j0 / s₀₀ and diagonal f_ii = s_ii (p.612). Intensity: exact finite-sample oracle α* = Σ[Var(s_ij) − Cov(f_ij, s_ij)] / Σ[Var(f_ij − s_ij) + (φ_ij − σ_ij)²] (eq. 3, p.609). Theorem 1 gives α* = (1/T)(π − ρ)/γ + O(1/T²) (p.609–610), and κ̂ = (p̂ − r̂)/ĉ is consistent (Theorem 2, p.613).
2. Not unbiased. The Cov(f,s) term (ρ) is handled explicitly, but only asymptotically, by the delta method on f_ij = s_i0 s_j0/s₀₀ (Lemma 2, eq. 8–9, p.612). Each piece is a sample-moment "usual estimator" of an asymptotic covariance. The paper notes that Frost and Savarino (1986) ignored this correlation (p.605) and calls ρ a "noteworthy innovation" (p.605, p.611–612).
3. Assumptions: (A1) i.i.d. over time, (A2) N fixed with T → ∞, (A3) finite fourth moments, all on p.606. Not normal ("stock returns are not assumed to be normally distributed", p.606). The mean is estimated: S uses the sample mean (eq. 1–2, p.606).
4. This is the baseline. It says relaxing (A1) is future research (p.606).

### Ledoit & Wolf (2004), JMVA 88, 365–411. Scaled identity, general asymptotics
1. Target μI with μ = ⟨Σ, I⟩. In the finite-sample analysis (§2, p.367–368) the target is deterministic, so no Cov(f,s) term appears. The bona fide estimator (eq. 14, p.380) plugs in m_n = ⟨S_n, I⟩, b̄², d². Intensity is a consistent plug-in under general asymptotics.
2. Not unbiased: only m_n is unbiased (Lemma 3.2, p.379), and the intensity is consistent in quadratic mean. The randomness of the target is not modelled. It is shown to be asymptotically negligible ("conditions under which it is legitimate to neglect the estimation error of our empirical Bayesian prior", p.380).
3. Distribution-free. Requires i.i.d. observations with mean zero (p.367, p.375), p_n/n ≤ K₁, a bounded 8th moment (Assumption 2), and an "uncorrelated products" condition (Assumption 3), all on p.375–376. Assumption 3 holds under normality or ellipticity (p.376).
4. Relaxes N fixed (LW2003 A2) via N/T bounded. Not via Stein. It strengthens the moment condition from 4th to 8th.

### Chen, Wiesel, Eldar & Hero (2010), IEEE TSP (arXiv 0907.4698). RBLW and OAS
1. Target F̂ = (tr Ŝ/p)I, which is data-dependent (eq. 3, p.4). Intensity: the exact Gaussian oracle ρ_O (eq. 6–7, p.5), then (i) RBLW = E[LW | Ŝ], the Rao–Blackwellized LW (Thm 2, eq. 17, p.6–7), and (ii) OAS, the fixed point of iterating the oracle with Σ replaced by the current estimate (Thm 3, eq. 23, p.8).
2. Finite-sample but not an unbiased risk estimate. The oracle eq. 6 keeps the target's randomness: the numerator is E tr[(Σ − Ŝ)(F̂ − Ŝ)], so the Cov(f,s) analogue is included exactly. Under normality it is evaluated in closed form through Wishart moments of tr Ŝ, tr Ŝ², tr²Ŝ (eq. 8–12, p.5). RBLW provably dominates LW in MSE (eq. 18, p.7). OAS has no optimality proof.
3. i.i.d. Gaussian with zero (known) mean, n < p allowed (p.3). Proofs use Haar-measure integrals (Appendix, p.19ff).
4. Relaxes the asymptotic regime: it is exact finite-sample under normality, at the price of adding normality. Not via Stein's lemma; it uses Rao–Blackwell and Wishart moments. The paper restricts itself to the scaled identity and says "the theory developed here can be extended to other non-identity shrinkage targets" (p.19).
- This is the closest analogue to Jack's scaled-identity result: Jack's closed form for the cross term should agree with the Gaussian oracle (7) [inferred].

### Touloumis (2015), CSDA (arXiv 1410.4726). Nonparametric Stein-type shrinkage
1. Targets: νI_p, I_p and D_S (the diagonal of sample variances) (§3, p.4–5). Intensity: plug-in of unbiased U-statistic estimators of tr Σ, tr Σ² and tr(D_Σ²) (Y₁N, Y₂N, Y₃N, p.4–5) into a normal-model formula λ*. Terms in the kurtosis B are dropped as asymptotically negligible (p.3–4).
2. The components are unbiased in finite samples, including when the mean is estimated (p.4). λ̂ itself is only consistent, being a ratio. Data-dependent target: for T = D_S the general formula λ_T (eq. 6, p.5) keeps the cross term E tr{(S − Σ)(Σ − D_S)}, which is the Cov(f,s) analogue. It is computed exactly under model (2), but the B-dependent piece is then discarded. For νI_p the oracle uses the true ν = tr Σ/p (p.2), so the cross term for the scaled identity is not modelled.
3. Nonparametric model X = Σ^{1/2}Z + μ with a "pseudo-independence" moment condition (3) on Z, which need not be independent (p.2–3). N, p → ∞ with trace-ratio conditions (4) (p.3), and t₁ = t₂ = 0 for the alternative targets (p.5). Mean estimated.
4. Relaxes normality (relative to Fisher & Sun 2011) and the known mean. Uses U-statistics, not a Stein identity. The discussion (p.11) flags target choice as future work. No factor target.

### Ollila & Raninen (2019), IEEE TSP (arXiv 1808.10188). Elliptical oracle for βS + αI
1. Target αI with α, β nonrandom (eq. 1–2, p.1). The oracle is α_o = (1 − β_o)η with η = tr Σ/p (Thm 1, p.2–3). The elliptical oracle β_o^Ell depends on Σ only through the sphericity γ and the elliptical kurtosis κ (Thm 3, p.6). Intensity: plug-in of γ̂ (spatial-sign based Ell1, or Ell2) and κ̂ (eq. 24, p.6).
2. The oracle is exact for finite n (Theorem 3), and ϑ̂ = tr Σ²/p has an unbiased estimator for any finite n and p, but only given κ (Thm 4, p.7). κ is then estimated, so overall unbiasedness is lost [inferred]. Because α is a free nonrandom scalar, the Cov(f,s) term never arises. η̂ = tr S/p is plugged in afterwards.
3. i.i.d. elliptical with finite 4th moments (p.3). Mean estimated, with a remark on the known-mean case (p.6). Consistency is under the RMT regime with p/n → c₀ (p.1, p.7).
4. Relaxes LW2004's distribution-free asymptotics to finite-sample exactness under ellipticity. The tool is an exact elliptical var(vec S) formula (Thm 2, eq. 14, p.4), not a Stein identity.

### Yi & Zou (2013), CSDA 58, 339–351. SURE-tuned tapering (tech-report scan)
1. No shrinkage target. The estimator is a weighted MLE σ̂_ij = w_ij^(k) σ̃_ij (tapering or banding) with nonrandom weights. The tuning parameter k is chosen by minimizing SURE (eq. 2.7, p.342).
2. Exact finite-sample unbiased risk estimate. Lemma 1 (eq. 2.1, p.342) is Jack's identity: E‖Σ̂ − Σ‖² = E‖Σ̂ − Σ̃ˢ‖² − ΣVar(σ̃ˢ_ij) + 2ΣCov(σ̂_ij, σ̃ˢ_ij). Because w is nonrandom, Cov reduces to w·Var, and Var(σ̃_ij) has a UMVUE under normality (Lemma 2, eq. 2.4, p.342). There is no data-dependent target, so a genuine Cov(f,s) term is never confronted.
3. i.i.d. N(μ, Σ) (p.339). Mean estimated, with translation invariance (Appendix, p.350). Finite n.
4. Not an LW paper. It supplies the SURE identity Jack uses. The Stein ingredient is Efron's covariance-penalty identity (Appendix proof of Lemma 1, p.350).

### Yi (2013), PhD dissertation, Univ. of Minnesota (114 pp)
- Chapter 2 (PDF p.17–49; printed p.4–35) reproduces Yi & Zou (2013) with no new theory. The SURE identity is Lemma 2.2.1, eq. 2.6 (printed p.10). Gaussian assumption, nonrandom weights.
- Chapter 4, "Extended Factor Analysis" (printed p.62–92): a factor model with correlated errors estimated by penalized EM ("glasso-GEM"). This is factor structure via likelihood, not shrinkage and not SURE (summary, printed p.94).
- It adds nothing on data-dependent targets or the Cov(f,s) term.

### Efron (2004), JASA 99, 619–632. Covariance penalties versus CV
1–2. General framework: Err = err + 2Σcov(μ̂_i, y_i) (eq. 2.8, p.620). SURE replaces cov by σ²E[∂μ̂_i/∂y_i] under the Gaussian (eq. 2.10–2.11, p.620). The parametric bootstrap estimates the covariance penalty by simulating from a fitted model (eq. 2.14–2.16, p.620). Covariance penalties are a Rao–Blackwellized form of cross-validation (abstract, p.619).
3. SURE needs Gaussian errors. The bootstrap version needs only a believable fitted model.
4. Not about covariance matrices. Relevant because the parametric-bootstrap covariance penalty is a ready-made fallback for ΣCov(f_ij, s_ij) when no closed form exists [inferred applicability].

### Haff (1979), JMVA 9, 531–544. Stein–Haff identity for the Wishart
1. Not LW-style. The estimators are Σ̂ = a[S + u t(u) I] with u an average eigenvalue (eq. 1.2, p.532), under Stein, entropy and quadratic losses.
2. The identity E[h(S) tr(TΣ⁻¹)] = 2E[h D*T₍½₎] + 2E tr[∂h/∂S · T₍½₎] + (k − p − 1)E[h tr(S⁻¹T)] (Thm 2.1, eq. 2.1, p.533–534) gives unbiased risk estimators when the loss involves Σ⁻¹ linearly (§4.3, p.540).
- Key warning for Jack: for the elementwise quadratic loss Σ(σ̂_ij − σ_ij)² q_ij, which with q_ij ≡ 1 is Frobenius, "the unbiased risk estimator is unobtainable in closed form unless σ̂_ij = a s_ij" (p.532).
- §4.4 (p.540–543) shows that the identity can still be used, by solving it recursively for terms E[m(u)σ_ii], to get dominance without an unbiased risk estimate (proof of Thm 6, p.542–543).
3. Wishart, i.e. normal data, with k > p + 1 for the inverse moments. Finite-sample.
4. Finite-sample and exact, at the price of normality and n > p.

### Haff (1980), Ann. Statist. 8, 586–597. Empirical Bayes estimation of Σ (scanned)
1. Σ̂ = a[S + u t(u) C] with C a known p.d. matrix and u = 1/tr(S⁻¹C) (eq. 1.3, p.587; eq. 4.1, p.590). This is shrinkage toward a data-scaled fixed matrix. The empirical-Bayes derivation uses an inverse-Wishart prior on Σ⁻¹ (§3, p.589).
2. The risk difference has an unbiased estimator under L₁ (Stein loss), from the Wishart identity (eq. 2.4, p.588). L₂ needs repeated application of the identity because it is quadratic in Σ⁻¹ (p.588). These losses are not Frobenius, and a target correlated with S enters only through the scalar u.
3. S ~ W(Σ, k) with k − p − 1 > 0 (eq. 1.1, p.586). Finite-sample.
4. Stein–Haff identity; requires n > p. No Frobenius loss, no factor target.

### Kubokawa & Srivastava (1999), Ann. Statist. 27, 600–609. Stein–Haff identity for elliptical data
1–2. Estimators: James–Stein and Dey–Srinivasan under Stein loss (p.601–602). No target or linear shrinkage.
- The extended identity (Lemma 1, p.603) reads E^f[tr G(S)Σ⁻¹] = E^F[(n − p − 1) tr G S⁻¹ + 2 tr D_S G]. The right side is an expectation under a different density F(x) = ½∫ₓ^∞ f, so it gives robust dominance ("uniformly for every unknown f", Prop. 1 and Thm 1, p.603–605), not an observable unbiased risk estimate [inferred].
3. Matrix-elliptical model on the whole N×p error matrix, with density |Σ|^{−N/2} f(tr Σ⁻¹eᵀe) (eq. 1.2, p.600). Rows are uncorrelated but not independent unless f is Gaussian [inferred]. Mean estimated (regression model). Finite-sample.
4. This is the only paper in the folder that relaxes independence across observations, partially: a shared radial component, such as common volatility mixing. It does so with a Stein-type identity.

### Landsman & Nešlehová (2008), JMVA 99, 912–927. Stein's lemma for elliptical vectors
1–2. Not a covariance-estimation paper. Thm 2 (p.918): E[h(X₁)(X₂ − μ₂)] = Cov(X₂, X₁)·E[∇h(X₁*)]. Here X₁* is elliptical with the cumulative generator, a different law from X₁ unless X is normal.
- Remark 1 (p.918): Cov(X) need not exist.
- The authors propose plug-in estimation of Cov(h(X₁), X₂) by fitting an elliptical model (p.925). In their t example the MLE plug-in is unstable when ν is estimated (Example 6, p.925).
3. Elliptical, finite mean.
4. This is the tool for moving Stein's lemma beyond normality. The expectation under X* is the obstacle to an observable unbiased estimate [inferred].

### Himeno & Yamada (2013), tech report. Unbiased estimators under non-normality
1–2. No shrinkage. Gives finite-sample unbiased estimators of tr Σ², (tr Σ)² and κ₁₁ = E[(zᵀΣz)²] − 2trΣ² − (trΣ)² from tr S², (tr S)² and Q = Σ‖xᵢ − x̄‖⁴ (Thm 1, p.3). These are consistent under conditions C1–C2 (Thm 2, p.4–6).
3. Model x = μ + Σ^{1/2}z with z ~ F general (eq. 1, p.1). Mean estimated. n/p → c (A1, p.2) for consistency only; unbiasedness is exact.
4. Relaxes normality and the known mean for the Var(s)-type term ΣVar(s_ij), which depends on these trace functionals plus a fourth-moment term [inferred link]. The tool is U-statistic algebra, not Stein.

### Sharpe (1963), Management Science 9, 277–293
- Defines the "diagonal" or single-index model R_i = A_i + B_i I + C_i with uncorrelated C_i (p.281), the structure behind LW2003's F.
- No estimation theory and no shrinkage.

### Stein (1956), Third Berkeley Symposium, 197–206
- The sample mean is inadmissible for n ≥ 3 under squared loss (p.197).
- The heuristic argument for shrinking "has not used the normality" (p.198).
- Origin of shrinkage. No covariance estimation.

### DeMiguel, Garlappi & Uppal (2005 working paper; the 1/N paper)
- Covariance shrinkage only via the Bayes–Stein (Jorion) estimator, which shrinks the mean toward the minimum-variance portfolio mean and inflates Σ̂ (eq. 7–8, PDF p.11).
- Explicitly excludes LW (2003) and other covariance-shrinkage estimators, citing Jagannathan & Ma (PDF p.11).
- No risk estimation for covariance. Relevant only as an empirical benchmark (1/N).

### Reading list: listed but NOT in the folder (content not verified, [inferred] from titles and memory)
- Li & Zou (2014), SURE information criteria for large covariance estimation (arXiv 1406.6514). Likely extends SURE to more covariance estimators. Check whether it covers shrinkage with random targets.
- Lescornel, Loubes & Chabriac (2014), unbiased risk estimation for covariance (ESAIM P&S). A direct novelty threat; must be read.
- Rajaratnam & Vincenzi (2016), Biometrika: a theoretical study of Stein's covariance estimator.
- Konno (2009); Chételat & Wells (2012), p > n with the Stein–Haff identity for singular Wishart.
- Canu & Fourdrinier (2017), JMVA: unbiased risk estimation in the elliptical case. This would address the E^F versus E^f obstacle above. Must read.
- Fourdrinier, Strawderman & Wells (2018), the Springer book on shrinkage estimation.
- Not in the folder but cited by the folder papers: Fisher & Sun (2011) (normal-model targets including D_S, cited by Touloumis p.2 and p.6); Schäfer & Strimmer (2005); Frost & Savarino (1986) (cited by LW2003 p.605).

---

## 2. VERDICT

### (a) Does any paper in the folder already give an unbiased or finite-sample risk estimate for a single-index or factor-model shrinkage target?

No.

- The only paper in the folder that uses the single-index target is Ledoit–Wolf (2003). It handles Cov(f_ij, s_ij) by an asymptotic delta-method estimator (Lemma 2, p.612), with N fixed and T → ∞. It is not unbiased.
- The finite-sample treatments of a data-dependent target in the folder are all simpler targets:
  - Chen et al. (2010): exact Gaussian oracle for (tr S/p)I (eq. 6–7, p.5), plus RBLW and OAS. Not SURE.
  - Touloumis (2015): exact cross term for D_S under a nonparametric model, then simplified asymptotically (p.5).
- Ollila–Raninen (2019) and LW (2004) avoid the cross term by treating the target scale as nonrandom.
- Yi–Zou (2013) and Yi (2013) give exact SURE, but only for nonrandom-weight estimators, so Cov(σ̂, s) = w·Var(s). No factor target appears.
- Supporting evidence that Jack's obstacle is real: Haff (1979, p.532) says closed-form unbiased risk estimators under elementwise quadratic (Frobenius-type) loss are "unobtainable in closed form unless σ̂_ij = a s_ij". Haff's identity yields observable unbiased terms only for losses linear in Σ⁻¹ (p.540).
- Caveat: two reading-list papers not in the folder could pre-empt the idea and must be checked: Lescornel–Loubes–Chabriac (2014) and Li–Zou (2014). Canu–Fourdrinier (2017) matters for any elliptical extension.

So, within the folder, "SURE or unbiased finite-sample tuning of LW shrinkage toward the Sharpe single-index target" is open.

### (b) Gap map: LW assumptions × who relaxed them in the folder × which Stein-type tool could attack the remaining gap

| LW assumption | Relaxed in folder? | By whom / how | Remaining gap | Plausible tool |
|---|---|---|---|---|
| Asymptotic regime (LW03: N fixed, T→∞; LW04: N/T bounded) | Partly: finite-sample results exist for the scaled identity only | Chen (exact Gaussian oracle); Ollila–Raninen (exact elliptical oracle); Touloumis and Himeno–Yamada (unbiased trace functionals); Yi–Zou (exact SURE, nonrandom weights) | Finite-sample unbiased R(α) for a ratio target (single-index or factor). Jack's thesis target. | Stein's lemma at the observation level, applied conditionally on the market series [suggestion, see note 1]; Stein–Haff on a partitioned Wishart; Efron's parametric-bootstrap covariance penalty as a numerical fallback (p.620); leave-one-out unbiasedness [suggestion, note 2] |
| Distribution (LW: distribution-free with 4th or 8th moments; SURE needs a model) | For scaled identity: elliptical (Ollila), linear-transform non-normal (Touloumis, Himeno–Yamada) | Exact moment formulas and U-statistics, not Stein | SURE for a factor target beyond normality | Elliptical Stein lemma (Landsman–Nešlehová Thm 2), but expectations are under X*, not X; ECD Stein–Haff (Kubokawa–Srivastava Lemma 1), same change-of-measure issue; Canu–Fourdrinier (2017, not in folder) is the paper to check. U-statistics avoid this but give unbiasedness only for polynomial functionals. |
| i.i.d. over time (LW03 A1) | Barely | Kubokawa–Srivastava: matrix-elliptical data, uncorrelated but dependent rows via a common radial factor (dominance only, Stein loss). Touloumis: pseudo-independence across variables, not time. | Serial correlation, GARCH, volatility clustering, all for Frobenius loss | ECD Stein–Haff for common-volatility mixing; Gaussian Stein's lemma still holds for jointly Gaussian non-i.i.d. data (Cov(h(X), X_j) = Σ_k Cov(X_j, X_k) E ∂_k h), so a SURE with a known or estimated time-correlation structure is plausible [inferred]; block or HAC-type U-statistics [inferred] |
| Finite 4th moments (LW04 actually needs 8th) | No for covariance estimation | Landsman–Nešlehová drop the Cov(X) requirement for the lemma itself (Remark 1, p.918), but Frobenius MSE needs 4th moments. Ollila's Ell1 uses spatial signs only to estimate γ robustly. | Heavy tails | Hard to attack with SURE (the risk must exist). Elliptical models with estimated κ, or robust-scatter targets, are more promising. |
| Mean known vs estimated | Yes, widely | Touloumis (U-stats with centering corrections), Himeno–Yamada, Ollila, Yi–Zou (translation invariance), LW03. Chen and LW04 assume mean zero. | None material. Jack's mean-zero convention is the restrictive case. | Wishart with n−1 df / U-statistics; straightforward |
| Target independent of S (empirical-Bayes "independence") | Recognized by LW03 (term ρ, p.605, p.611–612); exact only for the scaled identity (Chen) and D_S (Touloumis) | Asymptotic delta method (LW03) | Exact Cov(f,s) for f_ij = s_i0 s_j0/s₀₀ | See notes 1–2 |

Note 1 [suggestion, not from any paper here]. Under joint normality, condition on the market series x₀. Then b_i = s_i0/s₀₀ is linear in x_i and f_ij = b_i b_j s₀₀ is bilinear in (x_i, x_j). The ratio disappears conditionally. By the law of total covariance, Cov(f_ij, s_ij) = E[Cov(f_ij, s_ij | x₀)] + Cov(E[f_ij | x₀], E[s_ij | x₀]).
- The first term involves Gaussian quadratic-form moments with coefficients that depend only on x₀. It should admit a conditionally unbiased estimator.
- The second term reduces to moments of s₀₀, which is chi-square.
- This is the most promising way around the Stein's-lemma dead end.

Note 2 [suggestion]. Independence gives E[f_ij(X₋ₜ) x_it x_jt] = E[f_ij(X₋ₜ)] σ_ij exactly (mean-zero data). So Σ_t f_ij(X₋ₜ) x_it x_jt / T is unbiased for E[f^{(T−1)}] σ_ij. It is not exactly unbiased for E[f^{(T)}] σ_ij, but the bias is O(1/T²) [inferred]. This is a cheap near-unbiased estimator of the cross term to benchmark against.

Note 3 [inferred]. Haff (1979, §4.4, p.542–543) handles E[h(S)σ_ij] by applying the identity recursively. For Jack this means the Stein–Haff route to E[f_ij]σ_ij will produce a series rather than a closed form, which matches where he got stuck.
