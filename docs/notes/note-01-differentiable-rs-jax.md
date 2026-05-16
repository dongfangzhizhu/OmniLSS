# Technical Note 1 — Differentiable RS Algorithm in JAX

## Scope
This note documents the differentiable RS implementation strategy in OmniLSS Phase 0.

## Key points
- RS outer loop uses deterministic convergence criteria on deviance change.
- JAX-compatible kernels are isolated so score/hessian paths remain differentiable.
- Stability guards: gradient sanitization, step halving, and condition monitoring.

## Practical diagnostics
- `gradient_norm`
- `condition_number`
- `step_size_by_param`
- `deviance_history`

## Limitations
- Small design matrices can favor CPU LAPACK over GPU kernels.
- Warm-start overhead can dominate end-to-end runtime for RS_JAX.
