# Phase 1 Progress: Batched JAX RS API

中文版本: [phase1-batched-jax-rs-progress-2026-05-18_cn.md](phase1-batched-jax-rs-progress-2026-05-18_cn.md)

## Completed in this step

- Added `omnilss.algorithms.jax_rs_batch.batch_jax_rs_fit` as the batch-oriented API for independent JAX RS fits.
- Supported both broadcast design matrices and one design tuple per model.
- Supported one family spec for all models or one `FamilyJAXSpec` per model, with mixed-family inputs grouped deterministically by family name.
- Supported observation weights as `None`, `[n]`, or `[K, n]`.
- Added `gamlss_rs_jax_batch` to the formula integration layer so callers can fit multiple datasets/families without manually looping over `gamlss_rs_jax`.
- Exported the new batch APIs from `omnilss.algorithms`.
- Added tests confirming batched results match repeated `jax_rs_fit_core` calls and that the formula-layer batch helper returns fitted models.

## Notes

The current implementation prioritizes correctness and deterministic CPU/GPU behavior through a per-family loop. The public API and grouping structure are now in place for a future same-family GPU `jax.vmap` kernel without changing callers.
