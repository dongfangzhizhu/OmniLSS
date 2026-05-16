# Technical Note 2 — Numerical Stability in Distributional Regression

## Scope
Phase 0 numerical-stability baseline for OmniLSS runtime.

## Stabilization primitives
- `safe_exp`, `safe_log`, `safe_softplus`, `safe_divide`
- `sanitize_gradient`, `regularize_hessian`, `step_halving`

## Failure modes addressed
- NaN/Inf gradients from extreme likelihood curvature
- singular/ill-conditioned weighted least squares systems
- exploding updates during IRLS/JAX refinement

## Current policy
All optimizer paths must expose stability diagnostics in model slots.
