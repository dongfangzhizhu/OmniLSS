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

This API now has two execution modes: same-family/same-shape batches use the vectorized `jax.vmap` kernel described in [phase1-jax-vmap-batch-progress-2026-05-18.md](phase1-jax-vmap-batch-progress-2026-05-18.md), while mixed-family or shape-heterogeneous inputs retain the deterministic per-family fallback.
