# Phase 1 Progress: Same-Family JAX `vmap` Batch Kernel

中文版本: [phase1-jax-vmap-batch-progress-2026-05-18_cn.md](phase1-jax-vmap-batch-progress-2026-05-18_cn.md)

## Completed in this step

- Added a real `jax.vmap` execution path inside `batch_jax_rs_fit` for same-family batches whose per-parameter design matrices have identical shapes.
- Kept the deterministic per-model fallback for mixed-family batches or shape-heterogeneous models, preserving the existing public API and result order.
- Returned the same `JaxRSResult` objects as before, while the accelerated path computes parameters, etas, coefficients, deviance, iterations, and convergence flags in one vmapped XLA program.
- Added regression coverage that monkeypatches the per-model fallback and verifies same-family/same-shape batches complete through the vectorized path.

## Notes

This completes the Phase 1 requirement for batched multi-model fitting with `jax.vmap` as the GPU acceleration path. Remaining accelerator performance work should focus on benchmark-backed crossover thresholds and minimizing compile-time overhead for large production batches.
