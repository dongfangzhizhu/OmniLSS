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
- [ ] Complete Week 1 Day 5 numerical verification report.
- [ ] Complete Week 2 CG full-loop implementation and validation.
- [ ] Complete Week 3 warm-start decoupling + benchmark repair.
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

> Note: The active roadmap document defines Week 1–Week 4 only. Work is being advanced strictly in sequence from Week 1 onward; items beyond Week 4 are not yet defined in this plan and therefore cannot be marked complete.

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
