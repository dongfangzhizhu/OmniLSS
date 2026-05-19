# OmniLSS v1.0 Development Completion Report — 2026-05-19

中文版本: [v1-completion-report-2026-05-19_cn.md](v1-completion-report-2026-05-19_cn.md)

## Scope

This report closes the implementation tasks listed in [v1-development-plan-2026-05-18.md](v1-development-plan-2026-05-18.md). English remains the default project language; this report has a Chinese counterpart with the `_cn` suffix and reciprocal links.

## Completion summary

- **Phase 1 — JAX RS performance architecture:** cold-start JAX RS uses data-aware initialization, corrected IRLS equations, same-family `jax.vmap` batching, device-aware `method="RS"` routing, deprecated explicit `RS_JAX` compatibility, and lower-retracing TF score/Hessian kernels.
- **Phase 2 — Algorithm naming and Cole-Green completion:** the historical L-BFGS path is exposed with L-BFGS naming, package-level `cg_fit` routes to the real Cole-Green backend, `algorithm="auto"` uses implemented selection rules, and singular-Hessian protection has validation coverage.
- **Phase 3 — Unified distribution registry and validation:** `distribution_registry.py` is the authoritative lookup surface, built-ins are registered through one dictionary table, `resolve_family()` delegates to the registry, the old legacy if-chain resolver table has been replaced by a registry-backed compatibility shim, and numerical-contract validation coverage is in place.
- **Phase 4 — Production service layer:** proto3 array payloads preserve JSON compatibility, model artifacts use SQLite-backed persistent storage, gRPC and REST expose list/delete model lifecycle operations, and restart-safe prediction coverage validates persistent model reuse.
- **Phase 5 — JOSS publication readiness:** benchmark suites are split into no-R, optional-R, and optional-GPU paths; cold and warm timing claims are separated with confidence intervals; CI has a no-R/no-GPU benchmark smoke job; and the JOSS paper has updated need, method, validation, benchmark, limitation, and reference content.
- **Phase 6 — Commercial Pro differentiation:** Pro AutoML ranks candidate families through Core `BatchFit` client calls by deviance, AIC, BIC, and GAIC; Pro bootstrap confidence intervals use batch-fit resampling; and Pro implementation/tests preserve the gRPC client boundary without importing `omnilss`.

## Validation focus

The completion pass specifically rechecked the immediate execution order from the plan:

1. Cold-start JAX RS remains independent of NumPy RS warm-starts.
2. CG/L-BFGS naming no longer routes package-level Cole-Green calls through the historical L-BFGS alias.
3. Batched JAX RS has a real same-family `jax.vmap` path and deterministic fallback behavior.
4. Distribution family lookup is now registry-first while the historical private resolver name remains a compatibility shim, not a separate resolver chain.
