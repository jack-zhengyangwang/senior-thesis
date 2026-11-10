---
title: "SURE-Tuned Covariance Shrinkage for Portfolio Selection"
subtitle: "Senior Thesis Proposal"
author: "Jack [Last Name]"
date: "August 2026"
documentclass: article
geometry: margin=1.1in
fontsize: 12pt
linkcolor: blue
toc: true
numbersections: true
---

# Research Question

Can Stein's Unbiased Risk Estimate (SURE) serve as a general-purpose framework for tuning shrinkage intensity in covariance matrix estimation for portfolio optimization, and does it offer practical advantages over the standard Ledoit--Wolf analytical approach when applied to arbitrary shrinkage targets?

## Motivation

The sample covariance matrix is the workhorse input for mean-variance portfolio optimization. However, when the number of assets $N$ is large relative to the number of time-series observations $T$, the sample covariance matrix is notoriously ill-behaved—it is near-singular, its eigenvalues are poorly estimated, and the resulting portfolio weights exhibit extreme instability. DeMiguel, Garlappi, and Uppal (2009) demonstrated that, out of sample, the naive $1/N$ portfolio routinely outperforms optimization-based strategies whose inputs are sample covariance estimates.

The most successful solution to this problem is shrinkage estimation. Ledoit and Wolf (2003, 2004) proposed shrinking the sample covariance matrix toward a structured target (the identity matrix, or a single-factor market model). Their estimator takes the form

$$\hat{\boldsymbol{\Sigma}}_{\text{LW}} = (1 - \rho) \, \mathbf{S} + \rho \, \mathbf{T},$$

where $\mathbf{S}$ is the sample covariance matrix, $\mathbf{T}$ is the shrinkage target, and $\rho \in [0,1]$ is the shrinkage intensity. Ledoit and Wolf derive $\rho$ analytically by minimizing the asymptotic Frobenius risk, yielding a closed-form formula that depends on moments of the data.

This analytical approach has a limitation: the derivation of $\rho$ is specific to each choice of target $\mathbf{T}$. If one wishes to use a different shrinkage target—say, a sector-block-diagonal matrix, a constant-correlation model, or a sparsity-inducing target—the risk-minimization must be re-derived. The literature provides optimal shrinkage intensities for a handful of targets, but there exists no general, off-the-shelf method to tune $\rho$ for an arbitrary target.

**Stein's Unbiased Risk Estimate (SURE)** provides such a method. Introduced by Stein (1981), SURE gives an unbiased estimate of the risk of an estimator without requiring knowledge of the true parameter. In the covariance context, this means one can compute SURE for the shrinkage estimator $\hat{\boldsymbol{\Sigma}}(\rho) = (1-\rho)\mathbf{S} + \rho\mathbf{T}$ as a function of $\rho$, and minimize it directly to choose the shrinkage intensity. No asymptotic approximations are needed, and the approach generalizes immediately to any target $\mathbf{T}$.

The core question this thesis asks is: **does the SURE-tuned approach match or improve upon Ledoit--Wolf's analytically derived shrinkage in finite samples, and does it extend cleanly to targets for which no analytical shrinkage formula exists?**

---

# Background

## Stein's Paradox and the James--Stein Estimator

Consider the canonical problem of estimating a $p$-dimensional mean vector $\boldsymbol{\mu}$ from a single observation $\mathbf{x} \sim \mathcal{N}_p(\boldsymbol{\mu}, \sigma^2 \mathbf{I})$ (with $\sigma^2$ known). The maximum likelihood estimator (MLE) $\hat{\boldsymbol{\mu}}_{\text{MLE}} = \mathbf{x}$ is unbiased, consistent, and efficient. Yet Stein (1956) proved a shocking result: when $p \geq 3$, the MLE is **inadmissible** under quadratic loss. James and Stein (1961) constructed an explicit estimator that uniformly dominates it:

$$\hat{\boldsymbol{\mu}}_{\text{JS}} = \left(1 - \frac{(p-2)\sigma^2}{\|\mathbf{x}\|^2}\right)\mathbf{x}.$$

This estimator shrinks the observation toward zero by an amount proportional to $1/\|\mathbf{x}\|$; when the data is close to the origin (as it tends to be in high dimensions), shrinkage is stronger. Efron and Morris (1973) provided an elegant empirical Bayes interpretation: if $\boldsymbol{\mu}$ has a prior distribution $\boldsymbol{\mu} \sim \mathcal{N}(0, \tau^2 \mathbf{I})$, the posterior mean is a shrinkage estimator, and the James--Stein rule estimates the shrinkage factor from the data itself.

## Stein's Unbiased Risk Estimate (SURE)

Stein (1981) generalized the insight. For any estimator $\hat{\boldsymbol{\mu}} = \mathbf{x} + \mathbf{g}(\mathbf{x})$ where $\mathbf{g}$ is (almost) differentiable, the quantity

$$\text{SURE}(\hat{\boldsymbol{\mu}}) = -p\sigma^2 + \|\mathbf{g}(\mathbf{x})\|^2 + 2\sigma^2 \sum_{i=1}^p \frac{\partial g_i(\mathbf{x})}{\partial x_i}$$

is an **unbiased estimator** of the true risk $\mathbb{E}\|\hat{\boldsymbol{\mu}} - \boldsymbol{\mu}\|^2$. The remarkable feature of SURE is that it can be computed from the observed data alone — it does not require the unknown $\boldsymbol{\mu}$. This means one can select tuning parameters by minimizing SURE, and the resulting estimator adapts to the unknown characteristics of the data.

SURE has proven powerful in modern statistics. Donoho and Johnstone (1995) used it to tune wavelet thresholding parameters. Candès, Sing-Long, and Trzasko (2013) generalized it to singular value thresholding for matrix estimation. Yet its application to covariance matrix shrinkage for portfolio selection remains largely unexplored in the finance and accounting literature.

## Covariance Shrinkage for Portfolios

The Ledoit--Wolf estimator and its variants represent the state of the art for practical portfolio construction. The key papers are:

- **Ledoit & Wolf (2003):** Shrink the sample covariance toward the identity matrix.
- **Ledoit & Wolf (2004a):** Shrink toward the single-factor (market model) covariance matrix, which is more realistic for equity returns.
- **Ledoit & Wolf (2004b):** A practitioner-oriented summary.

The analytical shrinkage constant in Ledoit--Wolf (2004a) is

$$\hat{\rho}_{\text{LW}} = \frac{\hat{\pi} - \hat{\gamma}}{\hat{\gamma} \cdot T},$$

where $\hat{\pi}$ estimates the sum of asymptotic variances of the sample covariance entries, and $\hat{\gamma}$ estimates the Frobenius distance between the sample and target covariance matrices. This formula is elegant but target-specific: each new target requires deriving new expressions for $\pi$ and $\gamma$.

---

# Contribution

This thesis makes three contributions:

1. **Methodological:** It demonstrates that SURE provides a principled, target-agnostic method for tuning shrinkage intensity in covariance estimation. The approach is computationally straightforward—minimize a scalar function of $\rho$—and applies to any shrinkage target without additional analytical derivation.

2. **Comparative:** It provides a systematic finite-sample comparison of SURE-tuned shrinkage versus analytically-derived Ledoit--Wolf shrinkage across multiple targets (identity, single-factor, constant-correlation, and sector-block-diagonal). The comparison evaluates both estimation accuracy (Frobenius loss) and economic performance (out-of-sample portfolio variance).

3. **Practical:** It introduces shrinkage targets borrowed from the accounting and finance structure of the data—specifically, a target that respects GICS sector groupings—for which no analytical shrinkage formula exists, and shows that SURE tuning handles them naturally.

---

# Proposed Methodology

## SURE for Covariance Shrinkage

Let $\mathbf{X} \in \mathbb{R}^{T \times N}$ be a matrix of $T$ observations on $N$ asset returns, with $\mathbf{X}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ i.i.d. The sample covariance matrix is

$$\mathbf{S} = \frac{1}{T}\mathbf{X}^\top \mathbf{X}.$$

For a given shrinkage target $\mathbf{T}$ (assumed positive definite), define

$$\hat{\boldsymbol{\Sigma}}(\rho) = (1-\rho)\mathbf{S} + \rho\mathbf{T}, \quad \rho \in [0,1].$$

The Frobenius risk is $R(\rho) = \mathbb{E}\|\hat{\boldsymbol{\Sigma}}(\rho) - \boldsymbol{\Sigma}\|_F^2$. The Ledoit--Wolf approach derives an asymptotic approximation to $R(\rho)$, differentiates, and solves for $\rho^*$ analytically.

The SURE approach is different: construct an unbiased estimate of $R(\rho)$ that is computable from $\mathbf{X}$ alone, and minimize it numerically:

$$\hat{\rho}_{\text{SURE}} = \arg\min_{\rho \in [0,1]} \; \widehat{\text{SURE}}(\rho).$$

The key theoretical work is deriving the SURE expression for $\hat{\boldsymbol{\Sigma}}(\rho)$. Haff (1979, 1980) and Candès et al. (2013) provide the building blocks: the SURE estimate for covariance estimators involves the eigenvalues of $\mathbf{S}$ and derivative terms from the shrinkage function applied to each eigenvalue. I will follow the general formulation in Candès et al. (2013), who derive SURE for any spectral estimator that applies a shrinkage function $\eta(\lambda)$ to each sample eigenvalue.

## Shrinkage Targets

I will evaluate SURE tuning across four targets, ordered by increasing structure:

1. **Identity:** $\mathbf{T} = \mathbf{I}$. The simplest target. Allows direct comparison with Ledoit--Wolf (2003).
2. **Single-factor market model:** $\mathbf{T} = \sigma_m^2 \boldsymbol{\beta}\boldsymbol{\beta}^\top + \mathbf{D}$, where $\boldsymbol{\beta}$ contains market betas and $\mathbf{D}$ is a diagonal matrix of residual variances. Allows comparison with Ledoit--Wolf (2004a).
3. **Constant-correlation model:** All pairwise correlations set to the average sample correlation. A middle ground between the identity and factor models, popular in practice (Elton & Gruber, 1973).
4. **Sector-block-diagonal:** A block-diagonal matrix where within-sector covariances are set to their sample values and between-sector covariances are shrunk to zero. This reflects the accounting intuition that firms in the same GICS sector share common factors, while cross-sector links are weaker. **No analytical LW formula exists for this target** — SURE is the natural tuning method.

## Simulation Design

The simulation proceeds in three blocks:

### Block 1: Gaussian Returns
- Generate returns from $\mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_{\text{true}})$ where $\boldsymbol{\Sigma}_{\text{true}}$ is calibrated to real CRSP data (e.g., a 100-stock subset of the S&P 500, estimated using 5 years of daily data as the "truth").
- Vary the ratio $N/T$ (e.g., $N=50, 100, 200$; $T=60, 120, 250$) to probe the regime where shrinkage matters most.
- For each configuration, compute $\hat{\boldsymbol{\Sigma}}$ via LW-analytical and via SURE-tuned shrinkage for each target.
- Evaluate: Frobenius loss $\|\hat{\boldsymbol{\Sigma}} - \boldsymbol{\Sigma}_{\text{true}}\|_F^2$ and out-of-sample portfolio variance of the global minimum-variance portfolio (GMVP).

### Block 2: Heavy-Tailed Returns
- Repeat Block 1 but generate returns from a multivariate $t$-distribution with $\nu = 3, 5$ degrees of freedom.
- This tests the robustness of SURE tuning under fat tails, where the normality assumption underlying SURE is violated. Gaussian SURE will be biased; I will also implement a robust SURE correction following Xie, Kou, and Brown (2012).

### Block 3: Empirical Application
- Use CRSP daily returns for S&P 500 constituents, 2010--2025.
- Rolling-window estimation: estimate $\hat{\boldsymbol{\Sigma}}$ using the past $T=250$ days, form the GMVP, compute the realized portfolio variance over the next $h=21$ days (one month).
- Compare cumulative out-of-sample variance across estimators.
- Additionally test on non-US data (e.g., FTSE 100 or Nikkei 225 constituents) as a robustness exercise.

## Performance Metrics

- **Frobenius loss:** Measures statistical accuracy of the covariance estimate.
- **Out-of-sample GMVP variance:** The economic metric. The GMVP is the natural test case because it isolates the covariance estimate from the noise in mean-return estimates.
- **Turnover:** Average monthly portfolio turnover. Shrinkage should produce more stable weights.
- **SURE vs. Oracle gap:** In simulation, compare SURE-chosen $\rho$ to the oracle $\rho^*$ (chosen knowing $\boldsymbol{\Sigma}_{\text{true}}$) to quantify how much SURE loses relative to infeasible optimal shrinkage.

---

# Reading List

## 1. Stein & SURE Foundations

- **Stein, C. (1956).** "Inadmissibility of the Usual Estimator for the Mean of a Multivariate Normal Distribution." *Proceedings of the Third Berkeley Symposium on Mathematical Statistics and Probability*, 1, 197--206.
- **James, W. & Stein, C. (1961).** "Estimation with Quadratic Loss." *Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 361--379.
- **Stein, C. (1981).** "Estimation of the Mean of a Multivariate Normal Distribution." *Annals of Statistics*, 9(6), 1135--1151.
- **Efron, B. & Morris, C. (1973).** "Stein's Estimation Rule and Its Competitors---An Empirical Bayes Approach." *Journal of the American Statistical Association*, 68(341), 117--130.

## 2. SURE for Covariance Matrix Estimation

- **Haff, L.R. (1979).** "Estimation of the Inverse Covariance Matrix: Random Mixtures of the Inverse Wishart and Identity Matrices." *Annals of Statistics*, 7(6), 1264--1276.
- **Haff, L.R. (1980).** "Empirical Bayes Estimation of the Multivariate Normal Covariance Matrix." *Annals of Statistics*, 8(3), 586--597.
- **Haff, L.R. (1991).** "The Variational Form of Certain Bayes Estimators." *Annals of Statistics*, 19(3), 1163--1190.
- **Daniels, M.J. & Kass, R.E. (2001).** "Shrinkage Estimators for Covariance Matrices." *Biometrics*, 57(4), 1173--1184.
- **Yang, R. & Berger, J.O. (1994).** "Estimation of a Covariance Matrix Using the Reference Prior." *Annals of Statistics*, 22(3), 1195--1211.

## 3. The Ledoit--Wolf Benchmark

- **Ledoit, O. & Wolf, M. (2003).** "Improved Estimation of the Covariance Matrix of Stock Returns with an Application to Portfolio Selection." *Journal of Empirical Finance*, 10(5), 603--621.
- **Ledoit, O. & Wolf, M. (2004a).** "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices." *Journal of Multivariate Analysis*, 88(2), 365--411.
- **Ledoit, O. & Wolf, M. (2004b).** "Honey, I Shrunk the Sample Covariance Matrix." *Journal of Portfolio Management*, 30(4), 110--119.

## 4. Modern SURE and Risk Estimation

- **Candès, E.J., Sing-Long, C.A., & Trzasko, J.D. (2013).** "Unbiased Risk Estimates for Singular Value Thresholding and Spectral Estimators." *IEEE Transactions on Signal Processing*, 61(19), 4643--4657.
- **Donoho, D.L. & Johnstone, I.M. (1995).** "Adapting to Unknown Smoothness via Wavelet Shrinkage." *Journal of the American Statistical Association*, 90(432), 1200--1224.
- **Xie, X., Kou, S.C., & Brown, L.D. (2012).** "SURE Estimates for a Heteroscedastic Hierarchical Model." *Journal of the American Statistical Association*, 107(500), 1465--1479.

## 5. Portfolio Selection and Estimation Error

- **DeMiguel, V., Garlappi, L., & Uppal, R. (2009).** "Optimal Versus Naive Diversification: How Inefficient is the $1/N$ Portfolio Strategy?" *Review of Financial Studies*, 22(5), 1915--1953.
- **Kan, R. & Zhou, G. (2007).** "Optimal Portfolio Choice with Parameter Uncertainty." *Journal of Financial and Quantitative Analysis*, 42(3), 621--656.

## 6. Supplementary

- **Elton, E.J. & Gruber, M.J. (1973).** "Estimating the Dependence Structure of Share Prices---Implications for Portfolio Selection." *Journal of Finance*, 28(5), 1203--1232. (Constant-correlation model.)
- **Ledoit, O. & Wolf, M. (2017).** "Nonlinear Shrinkage of the Covariance Matrix for Portfolio Selection: Markowitz Meets Goldilocks." *Review of Financial Studies*, 30(12), 4349--4388. (For context on nonlinear shrinkage as a potential extension.)

---

# Timeline

| Weeks | Activity |
|-------|----------|
| 1--3 | Read foundations (Stein, James--Stein, Efron & Morris, SURE). Read Haff (1979, 1980). |
| 4--6 | Read Ledoit--Wolf papers in detail. Read Candès et al. (2013) for SURE for spectral estimators. Read DeMiguel et al. for portfolio methodology. |
| 7--8 | **Literature review draft.** Write background section. Derive the SURE expression for the linear shrinkage estimator $\hat{\boldsymbol{\Sigma}}(\rho) = (1-\rho)\mathbf{S} + \rho\mathbf{T}$. |
| 9--10 | **Implement simulation Block 1.** Gaussian returns, identity and single-factor targets. compare SURE vs. LW analytical tuning. |
| 11--12 | **Implement Block 2.** Heavy-tailed returns. Test robustness. Prepare initial simulation results for supervisor feedback. |
| 13--15 | **Implement Block 3.** Empirical application with CRSP data. Rolling-window portfolio exercise. Add constant-correlation and sector-block-diagonal targets. |
| 16--17 | Write methodology and results sections. Finalize all figures and tables. |
| 18--19 | Write introduction and conclusion. Full draft to supervisor. |
| 20 | Revisions and final submission. |

---

# Software and Data

- **Language:** Python (NumPy, SciPy, pandas, scikit-learn) with key shrinkage routines implemented directly. R as secondary option.
- **Data:** CRSP daily stock returns via WRDS; Compustat for sector classifications (GICS).
- **Optimization:** SURE minimization is a one-dimensional bounded optimization over $\rho \in [0,1]$ — solved via golden-section search or Brent's method (both in SciPy).
