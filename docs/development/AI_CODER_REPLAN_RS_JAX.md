# OmniLSS RS_JAX Replan (Code-Driven, English Default)

## Scope and source of truth
- This plan is implementation-driven and must follow code reality in:
  - `omnilss/src/omnilss/algorithms/jax_rs_core.py`
  - `omnilss/src/omnilss/algorithms/jax_rs_integration.py`
  - `omnilss/src/omnilss/algorithms/rs_algorithm.py`
  - `benchmarks/comprehensive_performance_test.py`
- Do **not** use README-level claims as evidence.

## One-line technical essence
`RS_JAX` is currently a hybrid pipeline (`NumPy` initialization + JAX `while_loop` IRLS core), where warm-start/init cost and small-`p` WLS efficiency dominate end-to-end behavior.

## Week 1 (stability + observability)
- [x] Day 1: Add RS_JAX timing decomposition hooks (`warm_start`, `core_rs`, `host_device_sync`).
- [x] Day 2: Emit diagnostics (`beta_norm_first_step`, `eta_clip_ratio`).
- [x] Day 3: Add first-step beta explosion guard + fallback init.
- [x] Day 4: Add regression tests for telemetry/fallback schema and sanity bounds.
- [x] Day 5: Update benchmark report template sections to explicitly surface all RS_JAX diagnostics.

## Week 2 (routing + performance policy)
- [x] Day 1: Add device-aware routing metadata foundation (`p_total`, suggested backend).
- [x] Day 2: Add explicit routing comparison group in `benchmarks/comprehensive_performance_test.py`.
- [x] Day 3: Add comparison mode under fixed precision constraints.
- [x] Day 4: Generate family × device recommendation matrix from benchmark outputs.
- [ ] Day 5: Freeze v1 strategy defaults (CPU-first for small `p`, no blind GPU path).

## Non-negotiable engineering rules
1. No speed claim without stage-level timing.
2. Optimize end-to-end latency first, not isolated kernel microbenchmarks.
3. Any “GPU is faster” conclusion must include `family`, `n`, `p`, `batch_size`, and warm-start share.
4. Keep `max_inner=1` as default unless damping/line-search is implemented and validated.
