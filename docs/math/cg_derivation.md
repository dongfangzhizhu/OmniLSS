# Cole-Green (1992) CG Derivation Notes (Week 1 Draft)

> Date: 2026-05-19  
> Chinese version: [cg_derivation_cn.md](./cg_derivation_cn.md)

## Purpose

This document is the Week 1 mathematical derivation workspace for the full CG implementation, focusing on cross-parameter second derivatives required by the joint scoring matrix.

## Notation Setup

For parameters \(\theta = (\theta_1, \dots, \theta_K)\), with predictors \(\eta_k = X_k \beta_k\):

- score component: \(u_k = \partial \ell / \partial \eta_k\)
- self second derivative: \(h_{kk} = \partial^2 \ell / \partial \eta_k^2\)
- cross derivative: \(h_{jk} = \partial^2 \ell / (\partial \eta_j \partial \eta_k)\), \(j \neq k\)

The CG outer step uses block Newton updates based on a joint matrix:

\[
H =
\begin{bmatrix}
H_{11} & H_{12} & \cdots & H_{1K} \\
H_{21} & H_{22} & \cdots & H_{2K} \\
\vdots & \vdots & \ddots & \vdots \\
H_{K1} & H_{K2} & \cdots & H_{KK}
\end{bmatrix},
\quad
s =
\begin{bmatrix}
s_1 \\
s_2 \\
\vdots \\
s_K
\end{bmatrix}
\]

with each block \(H_{jk} = X_j^\top W_{jk} X_k\), where \(W_{jk}\) is built from \(-h_{jk}\).

## Implementation Mapping (Planned)

- `d_ll / d_eta_k` from existing RS score functions.
- `d²_ll / d_eta_k²` from existing Hessian diagonal support.
- `d²_ll / (d_eta_j d_eta_k)` via JAX automatic differentiation (`jax.hessian` or `jax.jacfwd(jax.jacrev(...))`).
- Fisher-style expected information remains optional for convergence diagnostics.

## Verification Plan

1. AD vs finite difference (`check_grads`) for NO/GA/WEI.
2. Pairwise mixed-partial checks with R internal references (e.g., NO: \(\mu,\sigma\)).
3. Acceptance target: mixed partial error < 1e-6 against R-oriented reference outputs.

## Progress Checklist

- [x] Initial notation and block-matrix scaffold.
- [ ] Full derivation of update equation including Schur complement strategy.
- [ ] Distribution-specific derivative expansions (NO, GA, WEI, NBI).
- [ ] Finalized report-ready proof narrative.
