# Senior Thesis: Progress Log

SURE-tuned Ledoit–Wolf covariance shrinkage. Advisor: Professor Rosenman.

- Draft: Overleaf project "Jack Wang Senior Thesis Scratch" (log in to edit; not mirrored here)
- Original proposal (Aug 2026): `proposal/thesis_proposal.md` and `proposal/thesis_proposal_short.md` (+ .docx/.pdf)
- Papers (local PDFs): `papers/`
  - Yi & Zou, `Yi-Zou2013_techreport.pdf` (journal scan)
  - Yi dissertation ch. 2, `Yi2013_dissertation.pdf` (typed, easier to read)
  - `Chen-Wiesel-Eldar-Hero2010.pdf`, `Touloumis2015.pdf`
  - `Ledoit-Wolf2003.pdf`, `Ledoit-Wolf2004.pdf`, `Sharpe1963.pdf`
- Derivations: `derivations/sure_derivation.tex` / `.pdf` (risk to SURE form, step by step)
- νI target, exact distribution-free SURE, with proofs and simulation: `derivations/sure_scaled_identity.tex` / `.pdf`
- Yi's lemma write-up (English LaTeX of Jack's handwritten Chinese note; LW risk R(δ) and SURE form, open step = estimating Var/Cov terms): `derivations/yi_lemma.tex` / `.pdf`; original scan `derivations/yi_lemma_note_zh.pdf`
- Stein core notes (Stein lemma, LW, two shrinkers, covariance proof, full thesis→Stein map, outline): `derivations/stein_core.tex` / `.pdf`
- Reading list (moved out of Overleaf 2026-11-05): `reading_list.md`
- Source of truth = this folder (`~/dev/senior-thesis`). `~/Desktop/Thesis/` is an old flat copy (iCloud), not maintained.
- Simulation code: `sims/sure_sim.py` (figures and `results.txt` go to `sims/out/`; about 20 s)

## Goal

Estimate $\hat\Sigma = \alpha F + (1-\alpha)S$ and:

1. Write the Frobenius risk of LW shrinkage in SURE form. Done for general F; exact SURE for νI.
2. Plot SURE(α) against the true risk curve. Done for νI (`fig_sure_vs_risk`).
3. Derive a condition under which shrinkage reduces risk. Done: R(α) < R(0) iff 0 < α < 2α\*, and α\* ≥ 0 always.
4. Compare with the LW, RBLW, OAS and Touloumis estimators. Done for LW, RBLW and OAS (`fig_compare`); Touloumis not yet.

## Core formula

Lemma 1 (Yi & Zou) needs only $\mathbb{E}S=\Sigma$. Plugging in the LW shrinker gives:

$$R(\alpha)=\alpha^2\,\mathbb{E}\|F-S\|_F^2+(1-2\alpha)\sum_{ij}\mathrm{Var}(s_{ij})+2\alpha\sum_{ij}\mathrm{Cov}(f_{ij},s_{ij})$$

- SURE minimizer: $\hat\alpha=\sum(\hat V-\hat C)/\|F-S\|_F^2$.
- Risk is reduced iff $0<\alpha<2\alpha^*$, with the oracle $\alpha^*$, not $\hat\alpha$ (proved 2026-10-26). The $2\hat\alpha$ version holds only for the estimated SURE curve.
- LW (2004) toward νI equals $\frac{T-1}{T}\hat V/\|F-S\|^2$. SURE is $(\hat V-\hat C)/\|F-S\|^2$. So LW is SURE with two finite-sample terms dropped (proved, and checked numerically).
- At a fixed α, SURE is unbiased for R(α). The minimizer α̂ is a ratio of estimates, so it is not unbiased for α\*.

## Papers read

| Paper | Takeaway for the thesis |
|---|---|
| Ledoit & Wolf (2003) | Single-index target. α\* = Σ[Var(s) − Cov(f,s)] / E‖F−S‖². ρ̂ is consistent, not unbiased. Written up in Section 3. |
| Yi & Zou (2013), CSDA 58:339–351 | SURE tunes the tapering bandwidth k. Lemma 1 (risk decomposition) is pure algebra. Lemma 2 gives a UMVUE of Var(s_ij) and assumes Gaussian data. |
| Chen, Wiesel, Eldar & Hero (2010) | Oracle ρ (eq. 6) holds for any distribution. RBLW uses Rao–Blackwell, which needs Gaussian data. OAS. Target Tr(S)/p·I. |
| Touloumis (2015) | Non-Gaussian model with kurtosis term B. U-statistics Y1N and Y2N are unbiased for tr Σ² and tr²Σ. Targets νI, I, D_S. R package ShrinkCovMat. |

**Where to find Lemma 1 in Yi & Zou:**

- Journal scan: statement on PDF p. 4 (journal p. 342, eq. 2.1); proof on PDF p. 12 (journal p. 350, Appendix, eq. A.1).
- Dissertation: statement on PDF p. 23 (Lemma 2.2.1); proof on PDF p. 45 (Proof 2.1, eq. 2.13).

## Which terms can be estimated unbiasedly

| Term | Gaussian | Distribution-free |
|---|---|---|
| E‖F−S‖² | Yes, the observed ‖F−S‖² | Yes |
| ΣVar(s_ij) | Yes, Lemma 2 | Yes, (M̂ − Û₂)/T |
| ΣCov(f,s), fixed target | Yes, it is 0 | Yes, it is 0 |
| ΣCov(f,s), target νI | Yes, RBLW | Yes, (M̂ − Û₃)/(pT) (proved + simulated 2026-10-26) |
| ΣCov(f,s), target D_S | Yes, = Σ_i Var(s_ii) | Yes |
| ΣCov(f,s), single-index or constant-correlation | No | No (nonlinear in S; LW uses a consistent estimator) |

Proposed distribution-free estimators. These assume zero mean and S = (1/T)ΣXXᵀ.

- $\hat M=\frac1T\sum_t\|X_t\|^4$
- $\hat U_2=\frac{1}{T(T-1)}\sum_{t\ne u}(X_t^\top X_u)^2$
- $\hat U_3=\frac{1}{T(T-1)}\sum_{t\ne u}\|X_t\|^2\|X_u\|^2$

## Open issues

- **Decide on the target (raise with Rosenman).** Linear targets (I, νI, D_S) give an exact SURE. The single-index target from Section 3 only allows an approximate SURE.
- **Fix how S is defined.** Section 3's S divides by T. With an estimated mean, E[S] ≠ Σ and Lemma 1 picks up a bias term. Use T−1, or assume a known mean.
- **Handle an unknown mean.** The distribution-free estimators above need U-statistic corrections.
- **Simulation, done for νI on 2026-10-26.** Under Gaussian data SURE ≈ OAS. Next: AR(1) r ∈ {0.1, 0.9}, FBM, targets I and D_S, and Touloumis.
- **Clean up the Overleaf draft:**
  - The intro says constant-correlation target; Section 3 uses single-index.
  - A stray "Tip: Working on a plan…" line.
  - A duplicated sentence.
  - `\cite{ledoit2004}` next to "(2003)".
  - `\\` used as paragraph breaks.

## Next paper

Schäfer & Strimmer (2005), "A Shrinkage Approach to Large-Scale Covariance Matrix Estimation and Implications for Functional Genomics", *Stat. Appl. Genet. Mol. Biol.* 4(1), Art. 32.

- It estimates each term separately, with no normality assumption, for six targets.
- Check whether its Var̂(s_ij) is exactly unbiased.

## Log

- **2026-10 (before the 23rd):**
  - Read Yi & Zou, Chen et al., Touloumis.
  - Wrote lit-review subsections in Overleaf for Yi & Zou (my own notes), Chen et al. (2010) and Touloumis (2015), plus the "keeping the fourth-moment term" paragraph.
  - Fixed compile errors (Unicode outside math mode; missing `\[`).
- **2026-10-23:**
  - Mapped which SURE terms can be estimated unbiasedly (table above).
  - Picked Schäfer & Strimmer as the next paper.
  - Worked through the risk-to-SURE derivation: `derivations/sure_derivation.pdf`.
  - Located Lemma 1 and its proof in Yi & Zou.
- **2026-10-26:**
  - Proved the distribution-free estimators for the νI target are unbiased: V̂ = (M̂−Û₂)/T and Ĉ = (M̂−Û₃)/(pT). Known mean, finite 4th moments.
  - Fixed the risk-reduction condition: it is (0, 2α\*), not (0, 2α̂). Showed α\* ≥ 0 by Cauchy–Schwarz.
  - Showed LW (2004) = (T−1)/T · V̂/‖F−S‖². SURE = (V̂−Ĉ)/‖F−S‖².
  - Gaussian check: α\* from the SURE decomposition matches Chen et al. eq. (7) exactly.
  - Simulation (p=100, AR(1) r=0.5, Gaussian and t₁₀):
    - The mean SURE curve matches the true R(α) to within 0.2%.
    - SURE stays within 3% of oracle MSE at every T.
    - LW is +73% (Gaussian) and +87% (t) at T=6.
    - Under t, RBLW and OAS are 1.2–4.6× the oracle because they use the Gaussian fourth moment.
  - Write-up: `derivations/sure_scaled_identity.pdf`.
- **2026-10-27:**
  - Found `sims/stein_js.py` → `sims/out/fig_stein.png` (made 10-26, not logged before). It runs the Stein normal-means warm-up: James–Stein vs the MLE, and SURE(λ) vs the true risk for (1−λ)X.
  - Wrote the Stein → thesis map: Stein's lemma turns E[(X−θ)g] into an observable term. The covariance analogue is the ΣCov(f,s) term. The thesis estimates that term by U-statistics instead of a Gaussian identity, so the SURE is distribution-free.
  - Parallel worth a paragraph: linear SURE in the normal-means problem gives λ̂ = p/‖X‖², while JS uses p−2. LW vs SURE has the same pattern, since LW drops the Ĉ and (T−1)/T terms.
  - Merged the Stein-connection write-up into `derivations/stein_core.pdf` (orig 10-26 notes pp.1–3 + new pp.4–8). Copy in ~/Downloads.
  - Added "Every piece of the thesis, mapped to Stein" (pp. 8–10): Pythagoras = JS empirical Bayes, 2δ* window = JS 2(p−2) window, positive-part/clipping, Stein lemma vs U-stats, Gaussian check, Touloumis, Stein-lemma route for single-index target (approximate SURE), sims side by side. PDF copied to ~/Desktop/Thesis/.
  - Rewrote pp. 5–7 of stein_core in plain words: error = (a) distance moved − (b) noise in S + (c) copy-the-noise penalty, with sanity checks (Σ̂=S, Σ̂=F), D/V/C table, plain proof, and what M̂/Û₂/Û₃ average. Map now pp. 9–11. Desktop + Downloads copies updated.
  - Added pp. 5–6 to stein_core: 6-step derivation of E‖Σ̂−Σ‖² = (a) E‖Σ̂−S‖² − (b) V + (c) 2ΣCov(σ̂,s); notes it uses only E S = Σ (why T vs T−1 matters) and no distribution. Desktop + Downloads updated (14 pp).
  - Q: what justifies E[S]=Σ? Answer: holds exactly if mean known (S=1/T ΣXXᵀ) or mean estimated with 1/(T−1); fails with 1/T + demeaning (bias −Σ/T). Bias adds −b²+2b(Eσ̂−σ), unobservable, and is O(1/T) = same order as V, so not negligible. Needs only identical 2nd moments (stationarity), not normality/independence; V̂,Ĉ formulas do need independence.
  - Expanded Step 5 (p. 6): proof that Cov(A,B)=E[A(B−EB)] (only one variable needs centring), hence E[σ̂(s−σ)]=Cov(σ̂,s) given Es=σ; σ̂ may be biased/data-dependent.
  - Added pp. 10–11 to stein_core: role of Yi & Zou Lemma 1 (= three-piece formula; replaces unknown bias² with observable E‖Σ̂−S‖²; any estimator), Yi vs thesis table, LW rewrite → δ²D+(1−2δ)V+2δC → SURE, tapering eq. 2.11 as special case, why V cannot be dropped for LW. Now 16 pp.
  - Expanded Step 2 (pp. 11–13): linearity argument; D̂ free; V=(E‖X‖⁴−trΣ²)/T and E Û₂=trΣ² via independent pairs; C=Var(trS)/p=(E‖X‖⁴−(trΣ)²)/(pT) and E Û₃=(trΣ)²; summary table; cautions. Now 18 pp.
  - 2026-10-28: Rewrote Step 2 (pp. 11–13) in simple terms: pieces (a) D̂, (b) V̂, (c) Ĉ each as measures/compute/why; one idea = same-day minus different-day products; hand-worked example p=2,T=4 (δ̂=1/2); Gram-matrix computation; empirical-risk check (δ=0.5: 143.7 vs 143.6). Jack: ping before citing papers beyond Yi&Zou, Stein, LW.
- **2026-10-28 (cont.) — paused to ask Rosenman:**
  - Walked through: S = XᵀX/T (mean 0) = avg of daily Y_t = X_tX_tᵀ; Var(s_ij)=Var(y)/T; U-statistics (avg of kernel over pairs t≠u, exactly unbiased; n−1 sample variance is one). V̂ = (1/T)·Σ_ij [T−1 sample variance of daily y_t,ij].
  - Verified S definitions: LW 2004 §2 = XXᵗ/n, mean zero (same as ours; LW also scales Frobenius by 1/p). Yi & Zou lemma uses demeaned /(n−1); their taper uses MLE /n → the (n−1)/n factor. LW 2004 PDF in scratchpad (not yet in papers/). LW 2003 S definition NOT checked.
  - Unverified idea: bootstrap Var of s_ij in closed form = (T−1)/T·V̂ = LW's b̄²; bootstrap could give approx SURE for nonlinear (single-index) target.
  - Math check of stein_core (subagent): core algebra correct. TO FIX (not yet done): sign in sure_derivation.tex:93 (+2b·E(σ̂−s)); δ*=b²/d² ≠ (V−C)/D for νI (=(b²−C)/(d²−C)); "LW = SURE minus two terms" wording (LW drops Ĉ and subtracts V̂/T); add "common Σ across days" assumption; over-claims about outside papers (Chen/Touloumis/Haff); 0.2%→0.23%; 143.7 vs 143.6 comes from scratchpad sure_demo.py (move to sims/).
  - Also pending: add derivative step δ* to Yi section; decide on outside-paper citations (keep / cut to Yi&Zou, Stein, LW + James–Stein / further reading).
  - Questions for Rosenman: which target (νI exact vs single-index approx); mean-zero vs demeaned T−1; which paper for "expectation of quadratic form as numerical integral"; bootstrap as route for nonlinear target.
- 2026-11-05: Consolidated into dev. Copied the 6 proposal files from ~/Desktop/Thesis into `proposal/`; checked by md5 that all 14 Desktop files now exist in dev with the same content. Dev is the only copy going forward.
- 2026-11-05: Translated + typeset handwritten Chinese Yi-lemma note → `derivations/yi_lemma.pdf` (3 pp). Fixed in write-up: note's δ²E‖Σ̂−S‖² typo → δ²E‖F−S‖²; "only the third term depends on (i,j)" → all three are summed. Added remarks: why E S = Σ holds (mean-zero, /T), unbiased V̂ar(s_ij) = 1/(T(T−1)) Σ_t (x_ti x_tj − s_ij)², Cov(f,s)=0 for fixed F.
- 2026-11-05 (Overleaf edits): deleted lit-review 2.4 Chen-Wiesel-Eldar-Hero and 2.5 Touloumis; compacted LW proof (~1.5 pp) into §3 "Proofs of Theoretical Results" = 3.1 LW + 3.2 Yi's Lemma; fixed two overfull equations; turned plain-text cites into \citet/\citep (ledoit2003, stein1981, sharpe1963, yi2013) so all 5 .bib entries render; Reading List removed from draft → `reading_list.md` (several titles already truncated in source). Draft now 8 pp. Old version in Overleaf History.
- 2026-11-05: Added Overleaf §4 "Stein's Approach to LW Shrinkage": multivariate Stein lemma on x_t ~ N(0,Σ) gives Cov(h,s_ij) = (1/T)Σ_tΣ_k σ_ik E[x_tj ∂h/∂x_tk] — unknown Σ reappears as multiplier (vs known σ² in normal means). Var(s_ij) = (σ_iiσ_jj+σ_ij²)/T, unbiased V̂ar = [T s_ii s_jj + (T−2)s_ij²]/((T+2)(T−1)); exact for linear targets (νI), approximate for single-index. Verified by Monte Carlo: `sims/stein_lemma_check.py`. Fixed all 8 Overleaf overfull/underfull warnings → log clean (0/0/0), 11 pp.
- 2026-11-05: Rewrote Overleaf §4 as a step-by-step Stein derivation (Step 1 expand → Step 2 Stein's lemma → Step 3 unbiased estimates) for LW with νI target under normality: R(α) = (1−2α)[(trΣ)²+trΣ²]/T + α²E‖F−S‖² + 4α trΣ²/(NT); SURE uses V̂ = [T(trS)²+(T−2)trS²]/((T+2)(T−1)), b̂ = T[T trS² − (trS)²]/((T+2)(T−1)); α̂ = (V̂ − 2b̂/(NT))/‖F−S‖². Verified by MC (`sims/stein_sure_nuI_check.py`). §4.6: single-index only approximate. Log clean, 12 pp.

## Plan for 2026-11-09 session (whole-thesis pass)

Draft now = 1 Intro, 2 Lit review, 3 Proofs (LW + Yi), 4 Stein route (νI, Gaussian). Missing: the distribution-free SURE (main contribution), simulations, conclusion.

1. (~20 min) Finish walkthrough of §4: 4.2 setup → 4.3 expand → 4.4 Stein lemma → 4.5 SURE → 4.6 single-index. 4.1 done.
2. (~40 min) Overleaf fix list: intro says constant-correlation, §3 uses single-index; define S consistently (mean-zero, /T) across §§3–4; stray "Tip:" line, duplicated sentence, ledoit2004-vs-2003 cite. Port 10-28 math-check fixes (sign at sure_derivation.tex:93, δ* for νI, "LW = SURE minus two terms" wording, "common Σ across days" assumption).
3. (~75 min) New §5 Distribution-free SURE for νI: M̂, Û₂, Û₃; unbiasedness proofs; α̂ = (V̂−Ĉ)/‖F−S‖²; LW = (T−1)/T·V̂/‖F−S‖²; risk reduced iff 0<α<2α*. Source: derivations/sure_scaled_identity.tex.
4. (~60 min) New §6 Simulations: port fig_sure_vs_risk + fig_compare; ADD the §4 Gaussian-SURE α̂ as an estimator → Gaussian vs distribution-free SURE under N and t₁₀.
5. (~20 min) Conclusion skeleton + questions for Rosenman (target choice; mean-zero vs T−1; bootstrap for nonlinear target).
Needs Jack's OK: citing Chen et al. (RBLW/OAS) and Touloumis in §6.

## End-of-session reminders (2026-11-10)
- Give Jack a VERY high-level summary of the whole roadmap (Stein 4.1 → LW in Stein form → Step 1 expand → Step 2 Stein lemma → Step 3 estimate → §5 distribution-free → §6 sims → port to Overleaf).
- Possible §4 remark: variance term could be estimated directly by 4th moments (Yi/§5 route) and Stein used only on the target term f_ij (hybrid; unbiased only under normality).
- Walkthrough paused at §4.4 Step 2.4 (lem:stein-cov); next: apply with h = f_ij (skipping h = s_ij), then Step 3.
- Port §5/§6 into Overleaf (preamble envs, 3 figure uploads, §4 label + ref fix, Chen et al. citation decision, §3 ledoit2004→2003).

## 2026-11-10 (late): restructure agreed with Jack
- §1–4 stay (draft up to the risk function). §4 shows Stein's route reaches Yi's risk; §4.6 says only Cov(f_ij,s_ij) is stuck for single-index.
- NEW §5 = list of methods for Var(s_ij) and Cov(f_ij,s_ij), each labelled [literature] or [proposed]. Yi & Zou Lemma 2 = normal-theory UMVUE for Var (no target cov term needed: tapering weights fixed). LW 2003 = asymptotic ρ̂.
- NEW §6 = empirical comparison (fold old §5/§6 drafts + single_index_directions sims into §5/§6).
- NEW §7 = relaxing assumptions (Wishart = normality, gives Stein–Haff tools; elliptical, unknown mean, time dependence).
- Citations: anything in Overleaf biblio.bib (ledoit2003/2004, stein1981, sharpe1963, yi2013). Others pending Jack's skim (papers being pulled to papers/).
- Overleaf fixed today: Yi identity LaTeX, §2.1 contradiction + d/p + norms, intro target T→F single-index, §2.2 p/δ→N/α, §3.1 cite 2003 + mean-zero S, §4.3 explicit link to Yi's lemma. "Stein (1956)" still plain text (not in bib).

## 2026-11-10 (night): bib + §2 + outline in Overleaf
- Papers: all 8 candidates now in papers/ (Stein1956, Haff1979, Haff1980, Kubokawa-Srivastava1999, Efron2004, Landsman-Neslehova2008, Himeno-Yamada2013_techreport, Ollila-Raninen2019). Desktop/Thesis duplicates sent to Trash.
- Overleaf biblio.bib: added the 8 entries (thesis/added_refs.bib).
- Overleaf §2: "Stein (1956)" -> \citet{stein1956}; new opening paragraph of §2.2 on Haff (1980) (+ \citep{haff1979}).
- Overleaf outline (headings only) added: §5 Methods for Estimating the Risk Terms, §6 Empirical Comparison, §7 Relaxing the Assumptions, §8 Conclusion. [literature]/[proposed] tags are % comments. Compiles: 0 errors, 0 warnings, 15 pp.
- Intro (Overleaf): grammar fixes + approved word choices; research question sentence ("by following Stein's approach to risk estimation, rather than Ledoit and Wolf's asymptotic formula, can we obtain an unbiased, finite-sample estimate of the risk of linear shrinkage, tracing out its performance curve and choosing the shrinkage intensity that minimizes it?"); bridge sentence on LW's N-fixed asymptotics before it. Stein 1956 cited (was mis-cited as 1981).
- Placeholders (\textit{[Placeholder: ...]}) added: intro findings/roadmap paragraph + every §5–§8 heading. Still open intro items: 1/N + Michaud citations (not in bib), split paragraph 1, trim overlap with §2.1.
- Intro P1 split into "why Σ matters" / "why it's hard". Cited \citep{michaud1989, ledoit2003} (inversion) and \citep{demiguel2009} (1/N); 1/N sentence reworded to "portfolios built from sample estimates ... typically beaten out of sample" (was an overclaim about the covariance estimate). Bib +demiguel2009, +michaud1989.
- papers/: +DeMiguel-Garlappi-Uppal2005_workingpaper.pdf (Warwick-hosted Dec-2005 draft; published RFS 2009 needs library). Highlights added: DGU abstract p2; LW2003 p12 (inverting amplifies error, Michaud 1989); LW2004 p1–2. Michaud 1989 original NOT obtained (CFA members only) — cited via LW2003.
- §2.1 tightened (2629 -> 2130 chars). FIXED a math error: old draft said Cov(θ̂_i, X_i) = σ² E[∂g_i/∂X_i]; correct is Cov(g_i, X_i) = σ² E[∂g_i/∂X_i], i.e. Cov(θ̂_i, X_i) = σ² + σ² E[∂g_i/∂X_i].
- Michaud 1989 dropped (Jack: obvious); intro now \citep{ledoit2003} only; michaud1989 removed from bib.
- §2.2 rewritten around "how Stein maps to matrices" (S plays X, F plays the fixed point); Haff compressed; LW2003 tightened (removed unverified "seven competing estimators"); added 1-sentence LW2004 (identity target, N and T grow together).
- Intro bridge sentence corrected: LW asymptotics are N fixed (2003) OR N,T together (2004) — old wording ("N held fixed") was wrong for LW2004.

## 2026-11-10 (late night): novelty check + new direction
- 3 lit agents (reports in lit_review/). Closest prior work: Ikeda & Kubokawa (2016, JMVA 152:61–81; working paper papers/Ikeda-Kubokawa2015_workingpaper.pdf, highlighted Lemma 2.1, eq 2.16, Thm 3.1, Lemma A.2). Their factor target = LW2003 single-index (our inference); Wishart independence (Lemma A.2) removes the ratio problem; α* is O(1/n) approx, not exact SURE.
- DECISION (Jack): build thesis around Theorem 1 = exact SURE for single-index target under iid normality; attempt Theorem 2 = serially correlated Gaussian returns. Don't pitch "Stein relaxes normality".
- Overleaf restructured: new §2.4 Finite-Sample Shrinkage Intensities, §2.5 Dependence and Non-Normality, §4 new subsection (Proposition: LW2004 dropped term ≤ 2/N of variance term), §4.6 renamed "Why the direct route stalls…"; §5 Exact SURE single-index, §6 Serially Correlated Returns, §7 Numerical Results, §8 Discussion and Conclusion. Old methods list -> benchmarks in §7.3. Compiles 0 errors.
- TODO: add ikeda2016 to biblio.bib after Jack reads it; check published JMVA version; pull Schäfer–Strimmer 2005, Canu–Fourdrinier 2017, Kwan 2008.
- Yi–Zou Lemma 2 eq (2.4) as printed is NOT unbiased in Monte Carlo (n=10: E=0.2116 vs true 0.2500; n=30: 0.0733 vs 0.0776; also biased if applied to the n−1 sample cov). Possible typo; check Yi dissertation. Our Wishart-derived estimator [(m−2)s_ij² + m s_ii s_jj]/((m+2)(m−1)) (m = df; m=T known mean, m=n−1 demeaned) is unbiased (0.2496, 0.0776). Script: sims/check_yz_lemma2.py.
- 2026-11-10: Cov(f_ij,s_ij)=Var(f_ij) (i≠j) via IK Lemma A.2 — MC confirmed (sims/check_cov_equals_var.py).
- Var(f_ij) = [σ00·A + 2 b_i² b_j² σ00²]/T + B/T², A = b_i²Ψ_jj + b_j²Ψ_ii + 2b_ib_jΨ_ij, B = Ψ_iiΨ_jj + Ψ_ij².
- Unbiased estimator V̂ built from c=s00, b̂=s0/s00, Ψ̃ = T/(T−1)·(S − s0 s0ᵀ/s00); MC unbiased at T=10,20,60 (sims/check_var_f_estimator.py). Needs a written proof.
- END-TO-END CHECK PASSED (sims/sim_sure_single_index.py, N=20, T=40, 60k reps): E[SURE(α)] = R(α) at every α within 2 s.e. (diffs −0.03…−0.09 vs 2se≈0.40, risk≈58–62). Curves: sims/out/sure_single_index_curves.{png,pdf}. True-curve argmin α=0.575; SURE-chosen α mean 0.592 (sd 0.084). Individual SURE curves vary mostly in level, less in minimizer.
- R notebook sims/sure_single_index.qmd (mirrors SRP-2025 James–Stein workflow): shrinkage fn + one-sample check → SURE fn → closed-form true risk R(α) → one-setting MC (empirical 34.765, avg SURE 34.766, closed form 34.758) → risk over α (3 curves overlap) → loop over misspecification ρ=0..0.6 → performance curve (risk vs Σ_{i≠j}Ψ_ij²): SURE-tuned ≈ oracle-α everywhere, beats S everywhere, beats F once misspecification > ~2.5. Plots in sims/out/sure_*.png. Closed-form R(α) for single-index target also derived (true_risk()).
- 2026-11-11: Overleaf §2.4 Finite-Sample Shrinkage Intensities (Chen 2010, Touloumis 2015, Ikeda–Kubokawa 2016) and §2.5 Dependence and Non-Normality (Ollila 2019, Himeno–Yamada, Landsman, Kubokawa–Srivastava) written; bib +chen2010, +touloumis2015, +ikeda2016. Visible "[citation pending: Sancetta (2008)]" in §2.5 awaiting Jack's OK. % TODO in §2.4: Jack to confirm IK case (C2) = LW2003 target. Simulation NOT added to Overleaf: Jack to confirm the performance-curve x-axis with Rosenman.
- 2026-11-11 Goal 2 feasibility (sims/check_goal2_ar1.py; AR(1) known A shared by all series): E S = Σ still; Var(s_ij) = (σ_ij²+σ_iiσ_jj)tr(A²)/T² ✓; variance estimator with T_eff = T²/tr(A²) unbiased ✓, iid estimator underestimates (×3.4 at φ=.9,T=10). Cov(f, s−f) ≈ 0 but algebra says not exactly 0: conditional on x0, terms ∝ tr(HA(I−H)A) and Cov(c,q), q = x0'Ax0/x0'x0. Theorem 2 = Theorem 1 with T→T_eff plus small correction terms (to derive).
- 2026-11-11 Goal 2 derivation: Cov(f,s) closed form without iid (derivations/goal2_noniid.md), MC verified at T=10/30, φ=0/.5/.9 (sims/check_goal2_cov_formula.py); iid formula off by ×6–8 at φ=.9. Stops at unbiased estimation (needs replacements for IK independence, Wishart residuals, chi-square market variance).

- 2026-11-11: Goal 2 step 5 SOLVED. Theorem 2 (exact SURE under known serial correlation A, single-index target) written with full proofs in thesis/sec6_serial.tex and pasted into Overleaf Section 6 (compiles, 0 errors). Verified by Monte Carlo: E[SURE] = R(alpha) within 2 s.e. for A = I, AR(1) phi = 0.7, overlapping k = 5 (sims/check_goal2_sure.py); pairwise Cov-hat unbiased at T = 10 (sims/check_goal2_cov_hat.py). Citations still pending: Kruskal 1968, Magnus 1986. Sims NOT in Overleaf (waiting on Rosenman).

## Handoff 2026-11-12 (end of long session)
Overleaf structure now (labels in parentheses):
- §4 Stein's Approach (general recipe): 4.1 Setup (single-index target, market as series 0, four-block Sigma-bar/S-bar following IK), 4.2 Step 1 expand risk, 4.3 Step 2 Stein lemma stated for f_ij (lem:stein-cov; proof uses independence across days), 4.4 Step 3 variance term via Yi-Zou (known-mean version).
- §5 Targets with an Exact SURE (sec:levels01): Level 0 fixed target I (+ Figure 1, fig:level0, level0_risk_ratio_thesis.png), Level 1 scaled identity, 5.3 term dropped by LW2004 (lem:dropped: 2/(N(N+1)) <= C/V <= 1/N).
- §6 Single-Index Target: Where the Derivative Route Stops (sec:level2): Stein formula for Cov(f_ij,s_ij) (MC-verified, sims/check_stein_single_index.py); Step 3 never closes because of 1/s_00.
- §7 Exact SURE single-index (sec:thm1), §8 Serially Correlated (sec:thm2), §9 Numerics, §10 Conclusion: SKELETONS ONLY (Jack's request). Full §8 Goal 2 text saved in thesis/sec6_serial.tex.
Sims: sims/sure_levels01.qmd (Levels 0/1 R notebook), sims/level0_risk_ratio.R (fast, 14 s; reuses draws across delta).
Next: write §7 (Goal 1 proofs) into Overleaf; idea: quadratic-form view f_ij = x_i'Hx_j/T, H = market projection, as §7 setup unifying §7/§8. Bao-Kan (2013) only useful for exact Goal 2 TRUE-risk curves (not SURE); not yet downloaded or cited.
Pending decisions: add Magnus 1978/1986, Kruskal 1968, Bartz-Muller 2014, Sancetta 2008, Richardson-Smith 1991 to biblio.bib; §2.5 "[citation pending]". Rosenman meeting: confirm performance-curve x-axis and benchmarks.
