# OmniLSS v1.0 Development Plan

中文版本: [v1-development-plan-2026-05-18_cn.md](v1-development-plan-2026-05-18_cn.md)

## Scope

This plan tracks the path from the current v0.3 line to v1.0.0. English is the default project language, and every development document in this plan must have a Chinese counterpart with the `_cn` filename suffix.

## Immutable engineering constraints

- Do not reintroduce NumPy RS warm-starts into the JAX RS integration layer.
- Keep the stable RS cadence: `max_inner=1` remains the default for JAX RS and NumPy RS.
- Preserve float64 statistical precision for JAX arrays.
- Maintain backward-compatible public APIs; only additive parameters with defaults are allowed.
- Keep GPL isolation: `omnilss-pro` communicates with Core only through gRPC clients and must not import `omnilss` directly.
- Do not weaken reference tests or mathematical tolerances to hide failures.

## Phase 1 — JAX RS performance architecture

1. Remove the NumPy RS warm-start from `jax_rs_integration.py` and make cold-start JAX RS stable.
2. Add data-aware cold-start initialization through `FamilyJAXSpec.init_etas` and optional `jax_rs_fit_core(..., init_etas=None)` defaults.
3. Correct the JAX IRLS working-weight and working-response equations so they match the NumPy RS contract.
4. Add batched multi-model fitting with `jax.vmap` as the real GPU acceleration path.
5. Route `method="RS"` through device-aware automatic backend selection and keep `method="RS_JAX"` as a deprecated compatibility path.
6. Reduce TF AD retracing overhead with JIT-cached score/Hessian functions and, where feasible, handwritten derivatives.

## Phase 2 — Algorithm naming and Cole-Green completion

1. Rename the historical L-BFGS module away from the misleading CG name.
2. Make `cg_fit` point to the real Cole-Green implementation while keeping deprecated aliases for compatibility.
3. Implement actual `algorithm="auto"` selection rules in `mixed_algorithm.py`.
4. Add singular-Hessian protection and numerical validation tests for `fitting_cg.py`.

## Phase 3 — Unified distribution registry and validation

1. Make `distribution_registry.py` the authoritative registry.
2. Register every distribution family through a single dictionary-based mechanism.
3. Delegate `resolve_family()` to the registry while preserving the public function name.
4. Add independent numerical-contract validation for dpqr consistency, expected values, Fisher identities, Hessian signs, and quantile monotonicity.

## Phase 4 — Production service layer

1. Extend proto3 schemas with efficient array payloads while retaining JSON compatibility.
2. Replace temporary model storage with SQLite-backed persistent storage and artifact directories controlled by environment variables.
3. Add list/delete model RPCs and restart-safe prediction tests.
4. Add FastAPI REST endpoints that share the same persistent model registry.

## Phase 5 — JOSS publication readiness

1. Split benchmarks into no-R, optional-R, and optional-GPU suites.
2. Report cold and warm runtime claims separately, including confidence intervals.
3. Add a CI benchmark smoke job that does not require R or GPU.
4. Update the JOSS paper with statement of need, algorithm description, conservative performance conclusions, limitations, and complete references.

## Phase 6 — Commercial Pro differentiation

1. Implement Pro AutoML distribution selection through Core batch-fit RPCs.
2. Rank candidate families by deviance, AIC, BIC, and GAIC.
3. Add Pro bootstrap confidence intervals through batch-fit resampling.
4. Continue enforcing that Pro tests and implementation access Core only through the gRPC client boundary.

## Immediate execution order

1. Finish Phase 1 Task 1.1 by making cold-start JAX RS pass existing core and consistency tests without a NumPy RS warm-start.
2. Then implement Task 2.1 module renaming to remove CG/L-BFGS ambiguity.
3. Then implement Task 1.2 batched JAX RS fitting, because later Pro AutoML and bootstrap work depend on it.
4. Then unify the distribution registry to reduce future family-extension friction.
