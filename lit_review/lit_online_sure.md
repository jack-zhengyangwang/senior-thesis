# Literature check: SURE / finite-sample risk estimation for covariance shrinkage toward data-dependent structured targets

Date: 2026-11-10. Method: WebSearch + WebFetch (arXiv abstract pages, RePEc, CIRJE discussion-paper PDFs). No shadow libraries.
Process note: WebFetch auto-saved two open-access PDFs (CIRJE-F-958, arXiv 1810.08360) into the Claude tool-results cache; I did not download anything myself. I read pages 1-20 of CIRJE-F-958 and pages 1-6 of arXiv 1810.08360.

Labels: [READ] = I read the relevant text/equations. [ABSTRACT] = abstract/metadata page only. [MEMORY] = from prior knowledge, not re-verified in this session.

---

## 1. The single most important find

### Ikeda, Y. & Kubokawa, T. (2016). Linear shrinkage estimation of large covariance matrices using factor models. Journal of Multivariate Analysis 152, 61-81. DOI 10.1016/j.jmva.2016.08.001
Working paper: CIRJE-F-958 (Feb 2015), https://www.cirje.e.u-tokyo.ac.jp/research/dp/2015/2015cf958.pdf [READ, working-paper version]

- Setup: observations (y_i, x_i), y p-dim assets, x q-dim observed factors (q bounded). Factor target
  Σ̂_f = D̂ + Σ̂12 Σ̂22⁻¹ Σ̂21, with D̂ = Λ(Σ̂11.2), where Σ̂11.2 = Σ̂11 − Σ̂12Σ̂22⁻¹Σ̂21 (residual covariance).
  Two cases: (C1) spherical D̂ = p⁻¹tr(Σ̂11.2) I; (C2) diagonal D̂ = diag(Σ̂11.2).
- Estimator: Σ̂_α = αΣ̂11 + (1−α)Σ̂_f (= α Σ̂11.2 + (1−α)D̂ + Σ̂12Σ̂22⁻¹Σ̂21), plus a "double shrinkage" Σ̂(γ,β) that also shrinks the factor part.
- My inference (not stated in the paper): with q = 1 and x = market return, case (C2) is exactly Ledoit-Wolf (2003)'s single-index target. Diagonal = s_ii, off-diagonal = s_i0 s_j0 / s_00. So Σ̂_α in case (C2) is exactly LW-2003 shrinkage.
- How the target–S correlation is handled: this is the trick that gets past Jack's "ratio of sample moments" problem. Under joint normality of (y,x), Lemma A.2 shows that V11.2 = V11 − V12V22⁻¹V21 ~ W_p(n−q, Σ11.2) is independent of (V12, V22). Also V12 | V22 ~ N(Σ12Σ22⁻¹V22, Σ11.2 ⊗ V22). The authors state this holds for any n ≥ q, including p > n. Because S − F = Σ̂11.2 − Λ(Σ̂11.2) (in case C2, the off-diagonal part of the residual covariance), the regression part cancels. Every risk term then reduces to Wishart moments of Σ̂11.2 and the regression part.
- How the intensity is chosen: Lemma 2.1 writes the exact risk p⁻¹E tr(Σ̂(γ,β) − Σ11)² as a quadratic in the weights with moment coefficients J_ij. Theorems 2.1 and 2.2 then approximate those moments under normality, keeping terms up to O(n⁻¹) and O(n⁻²p) remainders, with (n,p) → ∞. In case (C2) with β = 1, the optimum is α* = (b2 − b20)/(b2 − b20 + (p/n) b1²), eq. (2.16), where b1 = p⁻¹tr Σ11.2, b2 = p⁻¹tr Σ11.2², and b20 = p⁻¹Σ_j (Σ11.2)_jj². Plug-in: Section 3 gives estimators of a1, a2, b1, b2, a20, b20, φ11, φ110 that are exactly unbiased under normality (Thm 3.1). These are Srivastava-type corrections such as b̂2 = n²/[p(n−q−1)(n−q+2)]·(tr Σ̂11.2² − (tr Σ̂11.2)²/(n−q)). The paper notes that â2 is unbiased but inconsistent when factor loadings are dense, a2 = O(p).
- Finite-sample status: the weight is NOT an exact finite-sample risk minimizer. It minimizes an approximate risk with O(n⁻¹) terms dropped, evaluated through ratios of unbiased estimators. They never form an unbiased estimate of the whole risk curve R(α) (that is, no SURE curve). Theory is high-dimensional asymptotic. Simulations use N = 100, p = 50-200, q = 3, under normal and chi-square errors.
- Assumptions: i.i.d. rows, joint normality of (y,x) for the risk derivation, factors observed (not PCA), q bounded, n ≥ q, plus trace-order conditions (A1)-(A6).
- Relation to Jack: this is very close in substance. It is a finite-sample, normal-theory, unbiased-component treatment of the exact LW single-index (C2) target, and it handles the target–S dependence. It differs in framing: they approximate the risk to O(n⁻¹) instead of building an exact unbiased risk estimate, and they never mention SURE or Stein's lemma for the tuning.

---

## 2. Other relevant work

| Paper | Target(s) | How intensity is chosen | Unbiased / finite-sample? | Target correlated with S? | Assumptions | Label |
|---|---|---|---|---|---|---|
| Ledoit & Wolf (2003), J. Empirical Finance 10, 603-621. DOI 10.1016/S0927-5398(03)00007-0 | Sharpe single index (market) | Minimize Frobenius risk; π, ρ, γ estimated consistently | No: asymptotic, N fixed, T→∞; ρ̂ is a consistent estimator of the asymptotic covariance between target and S | Yes, via ρ (asymptotic) | iid, finite 4th moments | [ABSTRACT]+[MEMORY] |
| Ledoit & Wolf (2004), "Honey, I shrunk...", J. Portfolio Mgmt | Constant correlation | Same, ρ̂ approximated | No (asymptotic) | Yes, asymptotically, with an approximation in the ρ term | iid | [MEMORY] |
| Ikeda, Kubokawa & Srivastava (2016), CSDA 95, 95-108. DOI 10.1016/j.csda.2015.09.011 (CIRJE-F-970) | Spherical, diagonal; double shrinkage | Ratio of exactly or approximately unbiased estimators of numerator and denominator of the optimal weight | Unbiased components, normal and non-normal | Targets are functions of S (tr S, diag S) | Normal and non-normal; high-dim | [ABSTRACT] |
| Schäfer & Strimmer (2005), SAGMB 4(1) Art. 32 | Six targets A-F, incl. diagonal and constant-correlation-like | λ* = Σ[Var(s_ij) − Cov(t_ij,s_ij)] / Σ(t_ij − s_ij)², plug-in with unbiased Var(s_ij) | Var(s_ij) term is unbiased. For data-dependent targets the Cov term is approximated or neglected (as I recall) | Partly; approximated | Distribution-free, iid | [ABSTRACT]+[MEMORY]; check the target-F row of their Table 1 |
| Chen, Wiesel, Eldar & Hero (2010), IEEE TSP 58(10) | Scaled identity | RBLW (Rao-Blackwellized LW) and OAS | RBLW: exact Gaussian conditional expectation | Yes (tr S) | Gaussian | [MEMORY] |
| Touloumis (2015), CSDA 83, 251-261. arXiv 1410.4726 | Identity, scaled identity, diagonal | Plug-in of U-statistic-type estimators | Consistent (abstract); nonparametric | Yes, but simple targets | Nonparametric, high-dim | [ABSTRACT] |
| Ollila & Raninen (2019), IEEE TSP | Scaled identity | Elliptical-theory optimal intensity | Asymptotic and elliptical | Simple target | Elliptical | [MEMORY] |
| Bodnar, Gupta & Parolya (2014), JMVA 132, 215-228 | Arbitrary target, but nonrandom | Deterministic equivalents | Asymptotic (p/n→c), a.s. optimal | No: target must be nonrandom | RMT | [ABSTRACT] |
| Lancewicki & Aladjem (2014), IEEE TSP 62(24), 6380-6390 | Multiple targets with a "general target" framework | LW-style MSE minimization | Abstract does not say. I believe they use unbiased estimators of Var(s_ij) under a general target class | Not confirmed | Not confirmed | [ABSTRACT] |
| Bartz, Höhne & Müller (2014), arXiv 1412.2041 "Multi-Target Shrinkage" | Multiple targets (e.g., other datasets, estimators) | Quadratic program | Asymptotically optimal (two asymptotic regimes) | Not addressed in abstract | | [ABSTRACT] |
| Oriol (2024/25), arXiv 2405.20086 | Multiple constant targets | Bona fide estimator | Kolmogorov asymptotics | No, the targets are constant | | [ABSTRACT] |
| Tong, Hu, Xi, Xiao, Guo & Yu (2018), Signal Processing 148, 223-233. arXiv 1810.08360 | Any T0, incl. data-built ones (diag(R), Toeplitz) | Closed-form LOOCV with quadratic loss | Distribution-free but heuristic; no unbiasedness claim | No: T0 is held fixed across folds (eq. 11). Zero-mean data. | iid, 4th moments | [READ pp. 1-6] |
| DeMiguel, Martin-Utrera & Nogales (2013), J. Banking & Finance 37(8), 3018-3034 | Covariance (and mean, weights) shrinkage | Smoothed bootstrap calibration | Bootstrap, not unbiased | Implicitly (resampling) | | [ABSTRACT] |
| Zhang/Zhou/Li-type Toeplitz-target papers (IEEE Access 2019, doc 8807200); Sci. Reports 2022, doi 10.1038/s41598-022-21889-8 | Toeplitz / common-covariance targets built from S | Closed-form intensity | Claimed exactly unbiased scalar estimates under (complex) Gaussian and non-Gaussian data | Yes, but the targets are linear in S, so no ratio problem | Gaussian / non-Gaussian | [ABSTRACT/snippet]; I did not verify the author-title match |
| Kwan (2008), Finance Research Letters 5(4), 236-244 | Constant correlation (average-correlation target) | Derives estimation error of the average correlation | Sampling variance expression; finite-sample status unclear | Yes, addresses the target's error | | [ABSTRACT] |
| Yi & Zou (2013), CSDA 58, 339-351 | Tapering (no target) | SURE for Frobenius risk | SURE, unbiased under normality | n/a | Normal | [ABSTRACT] |
| Li & Zou, arXiv 1406.6514 "SURE information criteria" | Tapering/banding/thresholding | SURE-based information criteria | SURE + asymptotics | n/a | | [ABSTRACT] |
| Lescornel, Loubes & Chabriac (2014), ESAIM P&S 18, 251-264. arXiv 1112.4735 | Model selection among projection estimators of a covariance function | Unbiased risk estimation + oracle inequality | URE, non-Gaussian | Not shrinkage toward a target | General | [ABSTRACT] |
| Canu & Fourdrinier (2017), JMVA 158, 60-72; Fourdrinier, Haddouche & Mezoued (arXiv 2012.11920); Rajaratnam & Vincenzi (Biometrika); Konno (2009, JMVA 100); Chételat & Wells (2012, AoS 40(6)) | Orthogonally invariant / eigenvalue / matrix-mean estimators | Stein-Haff unbiased risk estimates | Unbiased (normal or elliptical) | Not linear shrinkage toward a structured target | Normal / elliptical | [ABSTRACT]. Chételat-Wells has a published proof error (arXiv 2311.13140) |
| Ledoit & Wolf nonlinear shrinkage (2017+, incl. factor-model versions; 2022 JFEC review) | Eigenvalue shrinkage, optionally on factor residuals | RMT asymptotics | Asymptotic | n/a | | [MEMORY] |

Not found: any paper that (a) writes a SURE or exact unbiased risk-estimate curve R̂(α) for LW shrinkage toward the single-index, constant-correlation, or any factor target and minimizes it, or (b) gives the exact finite-sample (non-approximated) optimal intensity for the single-index target. I also found no paper applying Yi-Zou-style SURE to linear shrinkage toward a structured target.

---

## 3. Implications for Jack

1. The "stuck" step has a known way around it, provided the market factor is treated as an observed variable that is jointly normal with the returns. Use the regression decomposition S = Σ̂11.2 + Σ̂12Σ̂22⁻¹Σ̂21. Then S − F = offdiag(Σ̂11.2), and Σ̂11.2 ~ W(n−1, Σ11.2)/n is independent of the factor part (Ikeda-Kubokawa Lemma A.2). This lets you compute E‖F−S‖², ΣCov(f_ij,s_ij) and the full risk exactly with Wishart moments, and estimate them unbiasedly. Stein's lemma on the ratio s_0i s_0j / s_00 is not needed.
2. That makes an exact SURE R̂(α) for the single-index target look feasible. It also makes it only an incremental step beyond Ikeda & Kubokawa (2016). Jack must cite IK2016 and say clearly what he adds:
   - an exact (not O(n⁻¹)-approximated) unbiased estimate of the whole risk curve, instead of a ratio of unbiased pieces;
   - fixed-(N,T) exact theory instead of (n,p)→∞;
   - the case where the market index is built from the same N assets (x = wᵀy, which makes the joint covariance singular). IK assume an external factor with positive-definite Σ. I expect the conditioning argument still goes through, but have not checked;
   - non-normal/elliptical robustness, with comparisons to LW-2003, IK2016, and LOOCV (Tong et al.).
3. Before claiming novelty, check two things that I could not verify online: the published JMVA version of IK2016 (it may differ from the working paper, e.g., by adding exact moments), and Schäfer-Strimmer Table 1 (target F, constant correlation).

---

## VERDICT

- Closest existing work: Ikeda & Kubokawa (2016, JMVA 152:61-81). Their case (C2) with one observed market factor is exactly LW-2003 single-index shrinkage. Under normality they derive the risk as a quadratic in α, handle the target–S dependence by Wishart conditioning, and plug in exactly unbiased moment estimators.
- How close: partial overlap, substantial. It is the same estimator, the same loss, normal-theory unbiased ingredients, and it solves the covariance-term difficulty. It differs in that it is not SURE: there is no unbiased risk-curve estimate, the risk is approximated to O(n⁻¹) with (n,p)→∞ framing, the factor is external, and it is never cast as Stein/SURE tuning.
- Remaining gap (genuine but narrow): an exact finite-sample unbiased risk estimate R̂(α) (SURE in the Yi-Zou sense) for LW shrinkage toward the single-index target, with the market factor possibly built from the same data, and its exact minimizer. Not found anywhere. Novelty should be framed as an exact-finite-sample or SURE refinement of IK2016 and LW2003, not as solving an open problem. The constant-correlation target is less covered by this machinery (no regression decomposition, since r̄ is a nonlinear function of S). That target may be the more clearly open case.

## Sources
- https://www.cirje.e.u-tokyo.ac.jp/research/dp/2015/2015cf958.pdf ; https://econpapers.repec.org/RePEc:eee:jmvana:v:152:y:2016:i:c:p:61-81
- https://econpapers.repec.org/RePEc:eee:csdana:v:95:y:2016:i:c:p:95-108 ; https://www.cirje.e.u-tokyo.ac.jp/research/dp/2015/2015cf970ab.html
- https://arxiv.org/abs/1810.08360 ; https://arxiv.org/abs/1412.2041 ; https://arxiv.org/abs/2405.20086 ; https://arxiv.org/abs/1410.4726
- https://ideas.repec.org/a/eee/csdana/v58y2013icp339-351.html ; https://arxiv.org/abs/1406.6514 ; https://arxiv.org/abs/1112.4735
- https://econpapers.repec.org/RePEc:eee:jmvana:v:158:y:2017:i:c:p:60-72 ; https://arxiv.org/abs/2012.11920 ; https://arxiv.org/abs/2311.13140
- https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffff9961f70f/jef.pdf (LW 2003)
- https://econpapers.repec.org/RePEc:eee:finlet:v:5:y:2008:i:4:p:236-244 (Kwan 2008)
- https://econpapers.repec.org/RePEc:eee:jbfina:v:37:y:2013:i:8:p:3018-3034 (DeMiguel et al.)
- https://ieeexplore.ieee.org/document/8807200 ; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9643463/ (Toeplitz targets)
- https://econpapers.repec.org/RePEc:bpj:sagmbi:v:4:y:2005:i:1:n:32 (Schäfer-Strimmer)
- https://manuscript.isc.ac/Inventory/49/10385.htm (Lancewicki-Aladjem metadata)
