# OmniLSS Four-Week Development Plan

> Date Created: 2026-05-20  
> Core Principle: **CG algorithm completion first**, architecture refactoring second, performance honesty throughout the entire process  
> Chinese version: [four-week-execution-plan-2026-05-20_cn.md](./four-week-execution-plan-2026-05-20_cn.md)

---

## Execution Checklist (Always Review Before Starting Any Task)

- [x] Confirm the current milestone and day-level checklist before implementing.
- [x] Confirm documentation parity requirement (English + Chinese with cross-links).
- [x] Start with Week 1 Day 1–2 mathematical groundwork.
- [x] Complete Week 1 Day 1–2 literature/derivation outputs (draft scaffold completed).
- [x] Complete Week 1 Day 3–4 cross-derivative infrastructure (initial implementation completed).
- [x] Complete Week 1 Day 5 numerical verification report (local AD verification report published).
- [ ] Complete Week 2 CG full-loop implementation and validation (in progress: outer-loop scaffold + validation report added; R-alignment pending).
- [x] Started Week 3 Day 13 benchmark repair: added reusable `benchmark_jax(...)` helper with explicit cold/hot separation and wired `honest_benchmark(...)` to report `cold_s` + hot median.
- Added benchmark helper test coverage: `benchmarks/test_comprehensive_performance_benchmark_helper.py` validates non-negative cold/hot timings and expected cold>=hot behavior for JITed path.
- [ ] Complete Week 3 warm-start decoupling + benchmark repair (in progress: benchmark cold/hot helper integrated).
- [ ] Complete Week 4 integration/release preparation.

---

## Progress Log (Sequential Advancement)

### 2026-05-19 update

- Added bilingual four-week execution plan documents.
- Added bilingual CG derivation workspace documents.
- Implemented initial Week 1 Day 3–4 infrastructure:
  - `omnilss/src/omnilss/derivatives/cross_derivatives.py`
  - `omnilss/tests/test_cross_derivatives.py`
- Added finite structural verification (shape/symmetry/finiteness) for NO/GA/WEI.
- Published bilingual Week 1 Day 5 cross-derivative verification reports in `docs/reports/`.
- Published bilingual Week 2 progress report: `docs/reports/CG_FULL_VERIFICATION_2026_05_19.md` / `_cn.md`.
- Started Week 2 Day 6–7 implementation: added `omnilss/src/omnilss/algorithms/cg_algorithm_full.py` with `build_joint_scoring_matrix(...)` and `solve_joint_system(...)`.
- Added Week 2 tests: `omnilss/tests/test_cg_algorithm_full.py` for block assembly, linear solve consistency, and outer-step deviance decrease.
- Added `run_cg_outer_loop(...)` scaffold with relative global deviance convergence (`c_crit`) and per-iteration step-size history.

> Note: The active roadmap document defines Week 1–Week 4 only. Work is being advanced strictly in sequence from Week 1 onward; items beyond Week 4 are not yet defined in this plan and therefore cannot be marked complete.

---


### 2026-05-20 update

- Re-checked the execution checklist before implementation and continued in sequence from Week 2 tasks.
- Hardened Week 2 CG outer-step line-search behavior: when no improving step is found down to `min_step_size`, the step is now explicitly rejected with `accepted_step_size=0.0` and parameters unchanged.
- Added regression test coverage for the rejected-step path to ensure non-improving directions do not mutate `eta` and do not increase deviance.
- Extended Week 2 CG outer-loop observability: `run_cg_outer_loop(...)` now reports explicit `termination_reason` (`relative_deviance_converged` or `max_outer_reached`) for deterministic validation bookkeeping.
- Added Week 2 tests covering both termination paths to stabilize future R-alignment and report-generation assertions.
- Hardened Week 2 loop-stall handling: `run_cg_outer_loop(...)` now exits early with `termination_reason="no_progress_step_rejected"` when line-search rejects the step (`accepted_step_size=0.0`), avoiding non-productive max-outer cycling.
- Extended Week 2 validation harness assertions to require explicit termination bookkeeping across NO/GA/WEI/NBI and added a dedicated no-progress termination validation case.
- Added Week 2 regression checks that preserve `eta` on no-progress termination and confirm zero-deviance fixed-point runs are still classified as converged.

## Background and Architectural Understanding

Before defining the roadmap, we must acknowledge three confirmed architectural facts:

1. **The warm-start trap**: the current RS_JAX warm-start introduces the full overhead of NumPy RS into the measurement pipeline, completely masking GPU acceleration in WLS. On an RTX 3060, even the pure JAX core (without warm-start) is still 1.7–4.3× slower than NumPy RS.
2. **GPU disadvantage on small matrices**: for IRLS WLS with small design matrices `[n, p=2]`, CPU LAPACK (`np.linalg.lstsq`) is significantly faster than `jnp.linalg.lstsq`, which suffers from kernel launch overhead. GPU batch parallelism cannot be utilized effectively in sequential IRLS.
3. **`max_inner=1` is correct**: multiple inner IRLS iterations oscillate near the warm-start values, consistent with R’s `glim.fit`. This is not a bug and should not be modified.

This means the real value of JAX is not “making RS faster,” but rather:

- enabling **CG** implementation (CG requires AD-based cross-parameter second derivatives, which is JAX’s unique advantage),
- and enabling **large design matrices / multi-parameter families** for batched GPU inference.

---

- Added benchmark report environment metadata capture (Python/platform/processor/JAX/backend/device) to support Week 3 hardware-detail requirements.
