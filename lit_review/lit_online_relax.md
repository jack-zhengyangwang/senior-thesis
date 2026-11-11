# Relaxing Ledoit–Wolf assumptions with Stein-type tools: literature map and gap analysis

Prepared 2026-11-10 for Jack's senior thesis (SURE via Yi & Zou identity for LW linear shrinkage).

Evidence tags used throughout:
- [ABS] = read the abstract or landing page online in this session
- [BIB] = confirmed citation only (title, venue, or snippet), not the content
- [REC] = from my own background knowledge, not checked online this session. Verify before citing.

No PDFs were downloaded. No full texts were read, so every theorem-level claim below is at best abstract-level.

---

## 0. Framing: where Stein actually enters Jack's risk formula

R(α) = α² E‖F−S‖² + (1−2α) Σ Var(s_ij) + 2α Σ Cov(f_ij, s_ij)

- **E‖F−S‖²** is a plain expectation. It is unbiasedly estimated by ‖F−S‖² under any distribution.
- **Σ Var(s_ij)** has distribution-free unbiased (or nearly unbiased) estimators under iid sampling. These are U-statistics on the cross-products w_tij = (x_ti − x̄_i)(x_tj − x̄_j): Schäfer & Strimmer 2005 [REC on formula]; Himeno & Yamada 2014 and Touloumis 2015 build unbiased trace and functional estimators without normality [BIB/ABS]. Normality is not needed for this term. It does give a simpler closed form, Var(s_ij) = (σ_ij² + σ_iiσ_jj)/(T−1) for the 1/(T−1) S under Gaussian iid [REC, standard Wishart moment].
- **Σ Cov(f_ij, s_ij)** is the only place Stein's lemma is needed, because F is a nonlinear function of the data (for example, the single-index target). Under Gaussianity, Stein turns it into an expectation of derivatives of F, which can be computed from the data. LW 2003 instead estimates this term asymptotically with a sample-moment ("ρ") estimator that works under finite fourth moments [REC].

Skeptical consequence: under iid non-normal data, a distribution-free route (U-statistics, jackknife) to every term already exists and needs no Stein. So "Stein relaxes LW" is not true for the normality axis. Stein adds normality. Its value is finite-sample exactness, plus being able to handle structured dependence where U-statistic arguments break. Section 7 and the gap map build on this.

---

## 1. Finite-sample vs asymptotic intensity

| Paper | What was relaxed | Tool | Stein-type? |
|---|---|---|---|
| Ledoit & Wolf 2003, J. Empirical Finance 10:603–621, "Improved estimation of the covariance matrix of stock returns with an application to portfolio selection" (doi:10.1016/S0927-5398(03)00007-0) [REC details; single-index target confirmed in search snippets] | baseline: N fixed, T→∞, κ/T intensity | asymptotic moment estimators | No |
| Ledoit & Wolf 2004, JMVA 88:365–411, "A well-conditioned estimator for large-dimensional covariance matrices" (doi:10.1016/S0047-259X(03)00096-4) [REC; snippet confirms a finite-sample section characterising the risk-minimising combination] | gives the finite-sample oracle for the identity target, then estimates it consistently as N,T→∞ | general asymptotics | No |
| Chen, Wiesel, Eldar & Hero 2010, IEEE TSP 58(10):5016–5029, "Shrinkage algorithms for MMSE covariance estimation" (doi:10.1109/TSP.2010.2053029; arXiv:0907.4698) [ABS] | finite-sample improvement under Gaussian data. RBLW = Rao–Blackwellised LW and provably dominates LW in MSE. OAS = oracle-approximating shrinkage (closed form) | Gaussian sufficiency, Wishart moments | Gaussian-exact, but not Stein's lemma as such |
| Yi & Zou 2013, CSDA 58:339–351, "SURE-tuned tapering estimation of large covariance matrices" [ABS] | SURE theory for the Frobenius risk of tapering. This is the source of Jack's identity | SURE under Gaussian data | Yes (SURE/Gaussian) |
| Li & Zou 2014+, arXiv:1406.6514, "SURE information criteria for large covariance matrix estimation and their asymptotic properties" [ABS] | SURE-AIC/BIC analogues for banding. Assumes Gaussian iid | SURE | Yes |
| Ikeda, Kubokawa & Srivastava 2016, CSDA 95:95–108 (CIRJE-F-970) [ABS snippet] | convex combination of S and a target, with the weight estimated by exact or approximately unbiased estimators of its numerator and denominator, in normal and non-normal cases; spherical/diagonal targets; double shrinkage | unbiased moment estimators | Unclear from abstract. Exact unbiasedness under normality is likely Wishart moments, not Stein [inferred] |
| Ikeda & Kubokawa 2016, JMVA 152:61–81, "Linear shrinkage estimation of large covariance matrices using factor models" [ABS snippet] | factor-model target, N,T→∞, plug-in weights | asymptotics | No (from abstract) |
| Naul 2015, Stanford PhD thesis, "Sparse covariance estimation using SURE" (purl.stanford.edu/ry825cs1081) [ABS] | finite-sample and asymptotic analysis of Stein's eigenvalue estimator; extensions with sparsity | Stein's (1975/86) unbiased risk / Stein–Haff | Yes, but eigenvalue shrinkage, not linear-to-target |
| Ledoit & Wolf 2018, Bernoulli, "Optimal estimation of a large-dimensional covariance matrix under Stein's loss" [snippet] | critiques Stein's approach: it minimises an unbiased risk estimate rather than the risk, unbiased risk estimates are not unique, and the result needs isotonisation | RMT nonlinear shrinkage | Discusses Stein critically |

**Status.** For the scaled-identity target under Gaussian iid data, finite-sample optimal or unbiased intensities are well covered: LW 2004's finite-sample oracle, RBLW, OAS, and Ikeda–Kubokawa–Srivastava. For the single-index (Sharpe) target, I found no paper deriving a finite-sample SURE intensity with Stein's lemma for Cov(f_ij, s_ij). A targeted search returned nothing. This is consistent with Jack's core contribution being new, though absence from searches is not proof of absence.

---

## 2. Non-normality: elliptical, heavy tails, distribution-free

| Paper | Relaxed | Tool | Stein? |
|---|---|---|---|
| Ollila 2017, EUSIPCO, "Optimal high-dimensional shrinkage covariance estimation for elliptical distributions" (arXiv:1706.10066) [ABS] | Gaussian → unspecified elliptical with finite 4th moments; identity target; consistency in RMT regime | MSE formula involving an elliptical kurtosis parameter; consistent sphericity estimation | No |
| Ollila & Raninen 2019, IEEE TSP 67(10):2707–2719 (arXiv:1808.10188) [ABS] | elliptical sampling; two estimators of the sphericity parameter driving the oracle shrinkage; beats LW 2004 at small n | oracle MSE under elliptical sampling [I recall an exact finite-n MSE expression for the SCM in terms of κ; REC, not verified] | No |
| Chen, Wiesel & Hero 2011, IEEE TSP 59(9):4097–4107, "Robust shrinkage estimation of high-dimensional covariance matrices" (doi:10.1109/TSP.2011.2138698; arXiv:1009.5331) [ABS] | heavy tails: regularised Tyler fixed point, closed-form data-driven shrinkage coefficient from an MMSE argument | Tyler M-estimation | No |
| Touloumis 2015, CSDA 83:251–261, "Nonparametric Stein-type shrinkage covariance matrix estimators in high-dimensional settings" (arXiv:1410.4726) [ABS] | distribution-free, consistent intensities for 3 targets in N≫T. "Stein-type" means shrinkage in the James–Stein spirit, not Stein's lemma | U-statistics | No |
| Himeno & Yamada 2014, JMVA, "Estimations for some functions of covariance matrix in high dimension under non-normality and its applications" (doi:10.1016/j.jmva.2014.04.020) [BIB + citing snippets] | unbiased estimators of tr Σ², (tr Σ)², kurtosis-type quantities without normality | U-statistics | No |
| Kubokawa & Srivastava 1999, Ann. Statist. 27:600–609, "Robust improvement in estimation of a covariance matrix in an elliptically contoured distribution" [snippet] | extended Stein–Haff identity for elliptical data; minimax dominance carries over from normal | Stein–Haff (elliptical) | Yes, but for decision-theoretic dominance of orthogonally invariant estimators (Stein/KL loss), not LW linear-to-target |
| Tsukuma 2003, AISM 55:447 [snippet] | unbiased risk estimates via Stein and Stein–Haff under normal, then extended to elliptical | Stein + Stein–Haff | Yes, mean/covariance dominance, KL-type loss |
| Landsman & Nešlehová 2008, JMVA 99(5):912–927, "Stein's lemma for elliptical random vectors" [BIB/ABS snippet] | Stein's lemma for multivariate elliptical vectors, motivated by finance (CAPM-type covariance) | generalised Stein | Tool only. No covariance-shrinkage application found |

**Status.** Elliptical shrinkage is done for the identity target (Ollila; Ollila–Raninen) without Stein's lemma. Stein–Haff under elliptical data exists (Kubokawa–Srivastava) but was used for dominance results under invariant losses, not for an LW-style Frobenius SURE with a structured target. No one I found has combined the Landsman–Nešlehová elliptical Stein lemma with linear shrinkage.

A practical obstacle: the elliptical Stein lemma moves the expectation of the derivative to a different ("dual" generator) distribution. That expectation is not a plain function of the observed data, so it does not directly give a computable unbiased estimate [inferred from the form of the lemma, REC].

---

## 3. Time dependence / non-iid

| Paper | Relaxed | Tool | Stein? |
|---|---|---|---|
| Sancetta 2008, JMVA 99(5):949–967, "Sample covariance shrinkage for high dimensional dependent data" (Cambridge WP 0637, 2006) [ABS] | iid → weakly dependent (mixing) time series; feasible optimal shrinkage | HAC/long-run-variance-type estimation of Var(s_ij) | No |
| Bartz & Müller 2014, NeurIPS, "Covariance shrinkage for autocorrelated data" [ABS] | shows Sancetta is consistent but strongly biased in finite samples; bias-corrected estimator robust to lag choice | HAC-type with bias correction | No |
| Steland 2018, Statistical Papers 59:1441–1462 (arXiv:1809.00463) [ABS] | correlated high-dim time series; inference (CIs) for tr Σ, bounds for shrinkage to identity/diagonal | high-dim time-series asymptotics | No (not mentioned) |
| Greenewald & Hero (arXiv:1402.5568) [snippet] | Kronecker/block-Toeplitz spatio-temporal covariance with LW-style shrinkage to diagonal | plug-in LW weight | No |

**Status.** Every dependence paper I found estimates the intensity with long-run-variance (HAC) plug-ins and asymptotics. None uses Stein.

Key point (my analysis, not from a paper): Stein's lemma does not require independence. It holds for any jointly Gaussian vector: E[(X−μ) g(X)] = Cov(X) E[∇g(X)]. If returns are matrix-normal with separable covariance, X ~ MN(0, A, Σ) (T×N, with A the T×T temporal correlation), Stein's lemma still evaluates Cov(f_ij, s_ij). It just picks up A. For example, Var(s_ij) becomes proportional to (σ_ij² + σ_iiσ_jj)·tr((CAC)²) / (T−1)², with C the centring matrix [derivation sketch, REC/inferred; verify]. This is the most natural "relax iid via Stein" opening.

---

## 4. Non-stationarity / time-varying covariance

| Paper | Relaxed | Tool | Stein? |
|---|---|---|---|
| Engle, Ledoit & Wolf 2019, JBES 37(2):363–375, "Large dynamic covariance matrices" (doi:10.1080/07350015.2017.1345683) [ABS snippet] | constant Σ → DCC dynamics. GARCH devolatilisation, then nonlinear shrinkage of the DCC correlation-targeting matrix | GARCH + RMT nonlinear shrinkage | No |
| De Nard, Engle, Ledoit & Wolf 2022, J. Banking & Finance (DCC-NL with intraday OHLC) [snippet] | same plus high-frequency data | same | No |
| Ding & Zheng 2022+, arXiv:2211.10203, "High-dimensional covariance matrices under dynamic volatility models: asymptotics and shrinkage estimation" [ABS snippet] | shows when the LSD of S under scalar BEKK departs from iid Marchenko–Pastur; time-variation-adjusted S restores MP; consistent NLS | RMT | No |

**Status.** Time variation is handled by filtering (GARCH/DCC) followed by shrinkage, all asymptotic. No Stein-type work found.

A conditionally Gaussian model, x_t = τ_t^{1/2} Σ^{1/2} z_t with a common scalar volatility τ_t, makes the data independent but non-identically distributed Gaussian given τ. Stein's lemma then applies exactly, conditional on τ (see Opportunity B).

---

## 5. Estimated mean / T−1 corrections

- LW 2003 demeans and works asymptotically, so the O(1/T) effect of estimating the mean is ignored in the intensity [REC].
- Under Gaussian data, demeaning is handled exactly: (T−1)S ~ Wishart(T−1, Σ). Exact finite-sample results (RBLW, OAS, Wishart moments, Stein–Haff) use T−1 degrees of freedom automatically [REC].
- Schäfer & Strimmer 2005, SAGMB 4(1) art. 32 (doi:10.2202/1544-6115.1175) [BIB]. They use the 1/(n−1) unbiased S and an empirical unbiased-style estimator of Var(s_ij) with n/(n−1)³ scaling [REC on the formula].
- Distribution-free unbiased estimators of tr Σ² etc. (Himeno–Yamada; Srivastava–Yanagihara–Kubokawa 2014 per citing snippet) handle the estimated mean exactly [BIB].

**Status.** Essentially solved. On its own this is a footnote-sized contribution, not a thesis.

---

## 6. N > T and general asymptotics

| Paper | Relaxed | Tool | Stein? |
|---|---|---|---|
| Ledoit & Wolf 2004, JMVA [REC] | N,T→∞ together, identity target | general asymptotics | No |
| Bodnar, Gupta & Parolya 2014, JMVA 132:215–228, "On the strong convergence of the optimal linear shrinkage estimator for large dimensional covariance matrix" (doi:10.1016/j.jmva.2014.08.006; arXiv:1308.2608) [ABS] | p/n→c∈(0,∞), general target, distribution-free, almost-sure optimality | RMT deterministic equivalents | No |
| Ledoit & Wolf 2012 Ann. Statist. / 2018 / 2020 analytical nonlinear shrinkage [REC] | linear → nonlinear eigenvalue shrinkage | RMT | No (2018 discusses Stein critically) |
| Touloumis 2015; Ikeda–Kubokawa(–Srivastava) 2016 | N≫T, non-normal | unbiased U-statistics | No |
| Konno 2009 JMVA (singular Wishart Stein–Haff identities) [snippet] | N>T for Stein–Haff-based estimators | Stein–Haff (singular Wishart) | Yes, but invariant-loss dominance |

**Status.** General asymptotics are thoroughly done. Stein's lemma itself needs no N<T condition. SURE is exact for any (N,T) under Gaussian data, which is a genuine though modest selling point against LW 2003 (N fixed). Konno shows the Stein–Haff machinery also survives N>T.

---

## 7. Stein identities for non-Gaussian or dependent data, usable for unbiased risk estimation

| Tool | Reference | Usable for Jack? |
|---|---|---|
| Multivariate Gaussian Stein lemma with arbitrary covariance | standard; Eldar 2009 IEEE TSP 57(2), "Generalized SURE for exponential families" (arXiv:0804.3010) [ABS] derives SURE for a general (non-iid) Gaussian vector model | Yes. Covers correlated (time-dependent) Gaussian data directly |
| Hudson 1978 (Ann. Statist.) identity for continuous exponential families; Berger 1980, Hwang 1982 (discrete) [REC; Berger/Hwang named in Eldar snippet] | yes in principle, but returns are not a natural exponential family beyond Gaussian | Weak |
| Stein–Haff (Haff 1979; Stein 1975/86) for Wishart; singular Wishart (Konno 2009) | unbiased risk in terms of eigen-derivatives | Useful if Jack recasts the target in terms of S; heavy matrix calculus |
| Elliptical Stein–Haff (Kubokawa & Srivastava 1999) | unbiased risk only up to the mixing variable; dominance results | Partial. Gives inequalities more naturally than unbiased estimates |
| Elliptical Stein lemma (Landsman & Nešlehová 2008) | expectation under a different generator | Not directly computable. Obstacle noted above |
| Scale-mixture conditioning (Gaussian given τ) | standard trick [REC] | Yes. Exact conditional SURE; the unknown τ_t is the obstacle |
| Stein kernels / zero-bias transforms (Goldstein–Reinert, Ledoux–Nourdin–Peccati) | give E[X g(X)] = E[τ(X) g′(X)] for non-Gaussian X, with a Stein kernel τ | Theoretically elegant, but the kernel is unknown for real returns. Estimating it reintroduces nonparametric error. Not undergrad-feasible as an unbiased procedure [inferred] |
| Stein's method for dependent sequences (dependency graphs, exchangeable pairs) | mostly bounds distances to normality (CLT rates), not exact identities | Not useful for unbiased risk |

---

## GAP MAP

| LW assumption | Relaxed already? | By whom (main) | Via a Stein-type tool? |
|---|---|---|---|
| First-order asymptotic intensity (κ/T, N fixed) | Yes for identity target (finite-sample, Gaussian) | LW 2004 finite-sample oracle; Chen et al. 2010 RBLW/OAS; Ikeda–Kubokawa–Srivastava 2016 | Partly: Yi–Zou SURE for tapering/banding only. **Single-index target + Stein: not found** |
| Finite 4th moments / normality not required | LW already avoids normality. Elliptical-specific gains: Ollila 2017, Ollila–Raninen 2019; robust: Chen–Wiesel–Hero 2011; distribution-free: Touloumis 2015, Himeno–Yamada 2014 | | Stein–Haff elliptical exists (Kubokawa–Srivastava 1999) but **not applied to LW linear-to-target SURE** |
| iid over time (serial independence) | Yes, asymptotically | Sancetta 2008; Bartz–Müller 2014; Steland 2018 | **No. All HAC/asymptotic. Gaussian Stein with temporal covariance not found** |
| Stationarity (constant Σ) | Yes, via filtering + NLS | Engle–Ledoit–Wolf 2019; Ding–Zheng 2022 | **No** |
| Mean estimated (T−1) | Essentially yes | Wishart-exact work; Schäfer–Strimmer; Himeno–Yamada | Implicitly (Wishart) |
| N fixed / N<T | Yes | LW 2004; Bodnar–Gupta–Parolya 2014; nonlinear shrinkage; Touloumis | Stein–Haff for N>T: Konno 2009 (invariant loss, not linear-to-target) |

---

## Ranked thesis-sized opportunities

### A. (Best) Exact SURE for single-index linear shrinkage under serially correlated Gaussian returns
- **Model.** Matrix-normal returns X ~ MN(M, A, Σ). A is a T×T temporal correlation, parametric: AR(1) with a common ρ, or a block structure.
- **What it relaxes.** LW's iid assumption. Existing answers (Sancetta, Bartz–Müller) are asymptotic HAC plug-ins with known finite-sample bias, which Bartz–Müller document. Stein's lemma needs Gaussianity but not independence.
- **Deliverable.**
  - Derive Var(s_ij) and Cov(f_ij, s_ij) for the single-index target with A in the formulas.
  - Give an unbiased (A known) or plug-in (A estimated) SURE.
  - Simulation against LW 2003, Sancetta and Bartz–Müller at AR(1) ρ ∈ {0, 0.1, 0.3}.
  - Empirical check on daily returns, where ρ is small, and on overlapping or weekly returns, where dependence is mechanical and known. Overlapping returns are a clean motivating case because A is known exactly.
- **Not done?** No hit in searches. Greenewald–Hero is the closest and uses a plug-in LW weight.
- **Main obstacle.** A is unknown in general. Plugging in ρ̂ breaks exact unbiasedness, so Jack would need to quantify the plug-in bias or argue O(1/T²). The separability assumption (same temporal correlation for every asset) is restrictive and must be stated plainly.

### B. Conditionally Gaussian SURE: elliptical and volatility-clustered returns via scale mixtures
- **Model.** x_t = τ_t^{1/2} Σ^{1/2} z_t. With iid τ the returns are elliptical. With persistent τ_t you get a common-factor GARCH / stochastic-volatility flavour.
- **Method.** Given τ, Stein's lemma applies exactly to independent, heteroskedastic Gaussian rows. This yields a weighted-S SURE and an intensity that depends on τ.
- **What it relaxes.** Normality, toward elliptical/heavy tails, and the identical-distribution part of iid, toward volatility clustering. Both are relaxed with one Stein argument.
- **Not done?** Elliptical identity-target shrinkage exists (Ollila–Raninen), but not single-index and not via conditional Stein. Kubokawa–Srivastava elliptical Stein–Haff is for invariant-loss dominance.
- **Main obstacles.**
  - τ_t is latent. Proxies are a cross-sectional ‖x_t‖²/N (fine when N is large) or a GARCH filter on the market index. Either breaks exact unbiasedness.
  - The estimand must be pinned down: Σ, or E[τ]Σ?
  - Elliptical models give a single common τ for all assets. Real data violate that.

### C. Finite-sample SURE vs LW 2003 in the N ≈ T and N > T regime (single-index target)
- **What it relaxes.** LW 2003's N-fixed asymptotics. Stein's lemma is valid for any N,T, with no extra theory.
- **Deliverable.** Exactness proof plus a simulation and empirical study showing where κ/T fails (small T, large N) and SURE does not.
- **Not done?** Partly done for the identity target (OAS/RBLW, LW 2004). For the single-index target it appears new, but it is the least novel of the three.
- **Main obstacle.** Unbiasedness of the risk does not mean the argmin of SURE is good. SURE's variance can be large when N≫T. Clipping α to [0,1] is a nonlinearity whose effect needs simulation. Also, Gaussianity is assumed, so the comparison with LW on fat-tailed real returns must be honest. This is the expected weak spot, and it pairs well with B.

### D. (Stretch) Robustness of Gaussian SURE under non-normal iid data
- **Idea.** Derive or simulate the bias of the Gaussian Stein term when data are elliptical with kurtosis κ. Then add a kurtosis correction (Ollila-style κ̂, or Himeno–Yamada unbiased estimators) to the Var term while keeping Stein for the Cov term.
- **Main obstacle.** This is honest but mostly a correction exercise. For iid non-normal data, the distribution-free U-statistic route already gives every term without Stein, so the case for Stein here is weak.

**Recommendation.** A, or A combined with C, is the cleanest "relax an LW assumption via Stein" story: Stein's lemma tolerates dependence that the U-statistic and asymptotic machinery handle only approximately. B is the more ambitious finance-relevant variant. Avoid pitching "Stein relaxes normality". It does the opposite, and an examiner will catch it.

---

## Sources consulted (online this session)
- https://arxiv.org/abs/1808.10188 (Ollila & Raninen 2019)
- https://arxiv.org/abs/1706.10066 (Ollila 2017)
- https://arxiv.org/abs/1406.6514 (Li & Zou)
- https://econpapers.repec.org/RePEc:eee:csdana:v:58:y:2013:i:c:p:339-351 (Yi & Zou 2013)
- https://purl.stanford.edu/ry825cs1081 (Naul 2015)
- https://ideas.repec.org/a/eee/jmvana/v99y2008i5p949-967.html (Sancetta 2008)
- https://papers.nips.cc/paper_files/paper/2014/file/fa83a11a198d5a7f0bf77a1987bcd006-Paper.pdf (Bartz & Müller 2014)
- https://arxiv.org/abs/1809.00463 (Steland 2018)
- https://arxiv.org/abs/0907.4698 (Chen, Wiesel, Eldar, Hero 2010)
- https://arxiv.org/abs/1009.5331 (Chen, Wiesel, Hero 2011)
- https://arxiv.org/abs/1410.4726 (Touloumis 2015)
- https://econpapers.repec.org/RePEc:eee:csdana:v:95:y:2016:i:c:p:95-108 (Ikeda, Kubokawa, Srivastava 2016)
- https://econpapers.repec.org/RePEc:eee:jmvana:v:152:y:2016:i:c:p:61-81 (Ikeda & Kubokawa 2016)
- https://econpapers.repec.org/RePEc:eee:jmvana:v:99:y:2008:i:5:p:912-927 (Landsman & Nešlehová 2008)
- https://www.ism.ac.jp/editsec/aism/pdf/055_3_0447.pdf (Tsukuma 2003, snippet only)
- https://arxiv.org/abs/1308.2608 (Bodnar, Gupta, Parolya 2014)
- https://econpapers.repec.org/RePEc:taf:jnlbes:v:37:y:2019:i:2:p:363-375 (Engle, Ledoit, Wolf 2019)
- https://arxiv.org/abs/2211.10203 (Ding & Zheng)
- https://arxiv.org/abs/0804.3010 (Eldar 2009)
- https://arxiv.org/abs/2606.07382 (Chalvidal et al. 2026, stochastic-interpolation shrinkage; no Stein/SURE in abstract)
- https://arxiv.org/pdf/1402.5568 (Greenewald & Hero, snippet only)
- https://www.uzh.ch/cmsssl/econ/dam/jcr:fbe49637-07c1-4237-ad61-4e931246021e/bernoulli_2018.pdf (Ledoit & Wolf 2018, snippet only)
