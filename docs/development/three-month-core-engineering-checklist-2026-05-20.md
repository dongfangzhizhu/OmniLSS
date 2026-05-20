# OmniLSS Three-Month Core Engineering Checklist (2026-05-20)

This checklist tracks execution order and completion status for
`docs/development/three-month-core-engineering-roadmap-2026-05-20.md`.

## Week-by-week Progress

- [x] Week 1 / Task 1: remove `build_ad_family` runtime monkey-patching of class state.
- [x] Week 1 / Task 2: introduce a runtime backend protocol (`omnilss.runtime.backend.RuntimeBackend`).
- [x] Week 1 / Task 3: introduce deterministic policy/config scaffold (`omnilss.runtime.config`).
- [x] Week 2 / Task 1: damped IRLS with backtracking line search (initial integration).
- [x] Week 2 / Task 2: Hessian regularization module and singular-distribution tests.
- [x] Week 2 / Task 3: eta / z / weight clipping integration.
- [x] Week 3 / Task 1: optimizer abstraction layer (runtime scaffold).
- [x] Week 3 / Task 2: convergence framework (grad/deviance/parameter/curvature).
- [x] Week 3 / Task 3: iteration trace JSON export + replay.
- [x] Week 4 / Family metadata + constraints + validation engine.
- [x] Week 5 / benchmark framework (correctness/performance/convergence/stress).
- [x] Week 6 / CPU-first WLS (Cholesky-based).
- [ ] Week 7 / profiling + flamegraph tooling.
- [ ] Week 8 / error hierarchy + structured observability.
- [ ] Week 9 / statistical validation suite.
- [ ] Week 10 / artifact locking + deterministic replay.
- [ ] Week 11 / public API freeze and immutable return schemas.
- [ ] Week 12 / stress and failure-mode hardening.

## Execution Rule

Always review this checklist before starting the next coding block, and only advance in sequence unless explicitly instructed otherwise.
