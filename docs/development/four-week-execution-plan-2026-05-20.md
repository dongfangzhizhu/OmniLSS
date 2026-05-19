# OmniLSS Four-Week Development Plan

> Date Created: 2026-05-20  
> Core Principle: **CG algorithm completion first**, architecture refactoring second, performance honesty throughout the entire process  
> Chinese version: [four-week-execution-plan-2026-05-20_cn.md](./four-week-execution-plan-2026-05-20_cn.md)

---

## Execution Checklist (Always Review Before Starting Any Task)

- [x] Confirm the current milestone and day-level checklist before implementing.
- [x] Confirm documentation parity requirement (English + Chinese with cross-links).
- [x] Start with Week 1 Day 1–2 mathematical groundwork.
- [ ] Complete Week 1 Day 1–2 literature/derivation outputs.
- [ ] Complete Week 1 Day 3–4 cross-derivative infrastructure.
- [ ] Complete Week 1 Day 5 numerical verification report.
- [ ] Complete Week 2 CG full-loop implementation and validation.
- [ ] Complete Week 3 warm-start decoupling + benchmark repair.
- [ ] Complete Week 4 integration/release preparation.

---

## Background and Architectural Understanding

Before defining the roadmap, we must acknowledge three confirmed architectural facts:

1. **The warm-start trap**: the current RS_JAX warm-start introduces the full overhead of NumPy RS into the measurement pipeline, completely masking GPU acceleration in WLS. On an RTX 3060, even the pure JAX core (without warm-start) is still 1.7–4.3× slower than NumPy RS.
2. **GPU disadvantage on small matrices**: for IRLS WLS with small design matrices `[n, p=2]`, CPU LAPACK (`np.linalg.lstsq`) is significantly faster than `jnp.linalg.lstsq`, which suffers from kernel launch overhead. GPU batch parallelism cannot be utilized effectively in sequential IRLS.
3. **`max_inner=1` is correct**: multiple inner IRLS iterations oscillate near the warm-start values, consistent with R’s `glim.fit`. This is not a bug and should not be modified.

This means the real value of JAX is not “making RS faster,” but rather:

- enabling **CG** implementation (CG requires AD-based cross-parameter second derivatives, which is JAX’s unique advantage),
- and enabling **large design matrices / multi-parameter families** for batched GPU inference.

---

## Milestone Overview

| Week   | Core Goal                                                     | Deliverables                                            | Acceptance Criteria                               |
| ------ | ------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------- |
| Week 1 | CG theoretical modeling + cross-derivative infrastructure     | `cross_derivatives.py` + mathematical validation report | Numerical alignment with R, error < 1e-6          |
| Week 2 | Full CG implementation + outer loop                           | `cg_algorithm.py` (full version)                        | NO/GA/WEI convergence with bias matching R        |
| Week 3 | Architecture refactor: decouple warm-start, repair benchmarks | Refactored RS + honest benchmarking suite               | Benchmark report includes both cold/hot scenarios |
| Week 4 | Integration / testing / release preparation                   | v0.3.0-rc, PyPI draft                                   | CI fully green, license migration completed       |

---

## Weekly Plan (Canonical Task List)

The detailed week/day tasks, acceptance thresholds, dependencies, risks, DoD, and target before/after states are adopted exactly from the approved plan text provided by product direction for 2026-05-20.

To avoid divergence, implementation progress should be tracked by:

1. checking the execution checklist above,
2. updating task checkboxes in progress reports,
3. linking evidence docs in `docs/math/` and `docs/reports/`.

---

## Immediate Next Development Tasks (Activated Now)

### Week 1 / Day 1–2

- [x] Create this execution plan document.
- [x] Create Chinese counterpart and cross-link both documents.
- [x] Create `docs/math/cg_derivation.md` as the initial derivation output target.
- [x] Create `docs/math/cg_derivation_cn.md` and cross-link both derivation documents.
- [ ] Continue expanding derivations with full block-matrix notation and distribution-specific notes.
