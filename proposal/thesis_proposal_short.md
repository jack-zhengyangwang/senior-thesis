---
title: "SURE-Tuned Covariance Shrinkage for Portfolio Selection"
subtitle: "Senior Thesis Proposal"
author: "Jack Wang"
date: "August 2026"
documentclass: article
geometry: margin=1.1in
fontsize: 12pt
---

# Research Question

Can Stein's Unbiased Risk Estimate (SURE) be used to tune the shrinkage intensity in covariance
matrix estimation for portfolio selection, and how does it compare to the standard Ledoit--Wolf
formula — especially for shrinkage targets that have no closed-form solution?

# Motivation

The sample covariance matrix is the standard input to portfolio optimization, but it behaves badly
when the number of assets $N$ is large relative to the number of observations $T$: it is
near-singular and produces wildly unstable weights. DeMiguel, Garlappi and Uppal (2009) show that
optimized portfolios built on it are routinely beaten out of sample by the naive $1/N$ rule.

The standard fix is to shrink toward a structured target,
$\hat{\boldsymbol{\Sigma}} = (1-\rho)\mathbf{S} + \rho\mathbf{T}$. Ledoit and Wolf choose the
intensity $\rho$ by minimizing an *asymptotic* approximation to the risk. SURE (Stein, 1981)
instead gives a *finite-sample unbiased* estimate of risk, computable from the data alone, which
can be minimized numerically over $\rho$ for any target. The question is whether that finite-sample
property buys anything in practice — precisely in the small-$T$, large-$N$ regime where the
asymptotic approximation is weakest.

# Plan

- **Derive** the SURE expression for linear shrinkage using the Stein--Haff identity for Wishart
  matrices, and check early whether it collapses to Ledoit--Wolf for a fixed target.
- **Test three targets:** the identity and single-factor market model (both have Ledoit--Wolf
  benchmarks), and a sector-block-diagonal target, which has no closed-form constant and is the
  real test of whether SURE generalizes.
- **Benchmark against** the sample covariance, $1/N$, Ledoit--Wolf, and cross-validated $\rho$.
- **Simulate** across a grid of $N$ and $T$ (including $N > T$), using data-generating processes
  where each target is in turn the correct one.
- **Apply** to CRSP daily returns for S&P 500 constituents, rolling-window minimum-variance
  portfolios, with significance tested via Ledoit and Wolf (2011).

# Reading List

- Stein, C. (1981). "Estimation of the Mean of a Multivariate Normal Distribution." *Annals of Statistics*, 9(6), 1135--1151.
- Haff, L.R. (1980). "Empirical Bayes Estimation of the Multivariate Normal Covariance Matrix." *Annals of Statistics*, 8(3), 586--597.
- Ledoit, O. & Wolf, M. (2003). "Improved Estimation of the Covariance Matrix of Stock Returns." *Journal of Empirical Finance*, 10(5), 603--621.
- Ledoit, O. & Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices." *Journal of Multivariate Analysis*, 88(2), 365--411.
- Schäfer, J. & Strimmer, K. (2005). "A Shrinkage Approach to Large-Scale Covariance Matrix Estimation." *Statistical Applications in Genetics and Molecular Biology*, 4(1).
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). "Optimal Versus Naive Diversification." *Review of Financial Studies*, 22(5), 1915--1953.
- Ledoit, O. & Wolf, M. (2011). "Robust Performance Hypothesis Testing with the Variance." *Wilmott*, 55, 86--89.
