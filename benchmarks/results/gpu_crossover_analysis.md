# GPU Crossover Analysis (Week 3 Draft)

> Date: 2026-05-19  
> Chinese version: [gpu_crossover_analysis_cn.md](./gpu_crossover_analysis_cn.md)

## Objective

Document practical backend routing thresholds for OmniLSS based on honest cold/hot benchmark separation.

## Current Evidence Base

- Cold/hot separation helper implemented: `benchmark_jax(...)`.
- `honest_benchmark(...)` now emits `cold_s` and hot median `median_s`.
- Unit tests validate helper behavior and timing invariants.

## Preliminary Recommendation (to be revised with full sweep)

- Prefer **NumPy RS** for small/medium problems with low design dimension.
- Consider **JAX RS** only when repeated calls amortize compilation and matrix dimensions are sufficiently large.
- Working provisional threshold rule for experiments:

```text
Recommend JAX backend when n > 50k AND p > 10 (or multi-parameter CG blocks are large)
```

## Required Next Measurements

1. Run explicit `p`-sweep experiments (`p in {2, 5, 10, 20, 50}`) at fixed `n`.
2. Run `n`-sweep experiments (`n in {1k, 5k, 10k, 50k, 100k, 500k}`).
3. Record hardware details in every run:
   - CPU model
   - GPU model
   - JAX version
4. Compare cold-start and hot-start separately against:
   - NumPy RS (CPU)
   - R GAMLSS baseline

## Status

Week 3 benchmark repair is in progress; this document is the running analysis ledger.
