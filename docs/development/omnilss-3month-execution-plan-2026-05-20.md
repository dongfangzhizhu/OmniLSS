# OmniLSS Three-Month Execution Plan (Core `omnilss` Library)

> Goal: within 12 weeks, bring `omnilss` to a production-embeddable state that is stable, reproducible, observable, and extensible.

## 0) Shared constraints (must remain explicit)

- For small design matrices (typical `p=2`), current `RS_JAX` does **not** show a GPU advantage. `while_loop + jnp.linalg.lstsq` plus launch/sync overhead dominates.
- Warm-start reduces first-step IRLS divergence but introduces NumPy RS overhead; it must not be treated as a universal default.
- `max_inner=1` remains the safe default that aligns with `gamlss::glim.fit`; keep this as policy.
- Near-term strategy stays: **CPU-first + selective JAX + strong observability**, rather than blanket GPU usage.

---

## 1) Definition of Done

- Core families (`NO`, `GA`, `WEI`, `TF`, `NBI`, `ZIP`) are usable for fitting and prediction.
- Fit failures expose structured error codes and actionable diagnostics.
- Auto-routing decisions are explainable (`RS` vs `RS_JAX`).
- CI covers correctness, property/replay tests, reproducibility, and performance regression signals.

---

## 2) Ordered 12-week execution checklist

> Rule: always verify checklist state before starting the next task; proceed strictly in week order.

- [x] **Week 1 — Solver Router v1**
  - [x] Add routing policy object (`method`, `reason`, `threshold`, `backend`).
  - [x] Add `auto_select_method_trace()` while preserving `auto_select_method()` behavior.
  - [x] Persist routing trace into `additional_slots['method_routing']` for `gamlss(..., method='auto')`.
- [x] **Week 2 — IRLS/RS stability hardening**
  - [x] Add `eta` clipping and step damping.
  - [x] Add weight/Hessian epsilon floor.
  - [x] Enforce `max_inner=1` in default policy and logs.
- [x] **Week 3 — Error semantics and telemetry**
  - [x] Fit telemetry: per-iteration deviance, step norm, eta range, nan count, stage timings.
  - [x] Structured error envelope (`code/stage/parameter/hint`).
  - [x] Remove silent exceptions in model-construction and prediction-schema critical paths.
- [x] **Week 4 — Baseline performance gates**
  - [x] Family × n × backend benchmark matrix.
  - [x] Aggregate P50/P95, convergence rate, and accuracy deltas.
  - [x] Add performance degradation threshold alerts.
- [x] **Week 5–6 — Family stabilization**
  - [x] Prioritize `NO/GA/WEI/TF`; then `NBI/ZIP/IG/BE`.
  - [x] Optimize TF autodiff overhead using caching/reuse.
  - [x] Unify fixed-parameter behavior (broadcast/length validation/error codes).
- [x] **Week 7 — Prediction contract hardening**
  - [x] Enforce term order and smooth metadata validation.
  - [x] Convert legacy fallback to explicit opt-in (default off).
- [x] **Week 8 — Test-system upgrade**
  - [x] Property-based + replay + stress tests.
  - [x] Family-level minimum coverage baseline.
- [x] **Week 9 — Auto Backend Selector v2**
  - [x] Tune routing rules using benchmark data.
  - [x] Emit recommendation reason + confidence.
- [x] **Week 10 — API stabilization**
  - [x] Freeze core API surface.
  - [x] Add contract tests against breaking schema changes.
- [x] **Week 11 — RC hardening**
  - [x] Full regression run and P0/P1 cleanup.
- [x] **Week 12 — GA handoff**
  - [x] Deliver developer guide, performance guide, and next-phase plan.

---

## 3) Current status (2026-05-20)

All Week 1–Week 12 tasks above are marked complete in sequence and aligned with the engineering checklist at:

- `docs/development/three-month-core-engineering-checklist-2026-05-20.md`

No out-of-order backlog remains in this three-month block. New work should be planned as a follow-up phase document.

---

## 4) Task template for future blocks

For each task, always include:

- Changed file list
- Acceptance criteria (functionality + tests + performance)
- Rollback strategy
- Risk + observability metrics
