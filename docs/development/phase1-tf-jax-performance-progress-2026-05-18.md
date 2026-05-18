# Phase 1 Progress: TF JAX Score/Hessian Performance

中文版本: [phase1-tf-jax-performance-progress-2026-05-18_cn.md](phase1-tf-jax-performance-progress-2026-05-18_cn.md)

## Completed in this step

- JIT-wrapped the generic AD score/Hessian vmap helper so AD-built families cache vectorized derivative kernels.
- Replaced TF `FamilyJAXSpec` score functions with handwritten Student-t derivatives for `mu`, `sigma`, and `nu`.
- Replaced TF diagonal Hessians with negative Fisher-information-style approximations for stable IRLS updates.
- Added regression tests comparing handwritten TF scores against JAX autodiff and checking TF Hessians remain finite and negative.

## Notes

The TF JAX path no longer builds vmapped gradient/Hessian closures inside the TF spec.  This reduces tracing pressure in repeated RS calls while keeping score formulas numerically aligned with autodiff references.
