# Month 1 Legacy Prediction Runtime Audit (2026-05-18)

> Chinese version: [month1-legacy-prediction-runtime-audit-2026-05-18_cn.md](month1-legacy-prediction-runtime-audit-2026-05-18_cn.md)
>
> Parent plan: [six-month-execution-plan-2026-05-17.md](six-month-execution-plan-2026-05-17.md)
>
> Weekly checklist: [six-month-weekly-implementation-checklist-2026-05-17.md](six-month-weekly-implementation-checklist-2026-05-17.md)

## Scope

This note closes the Week 2 legacy prediction-entry audit for the current runtime surface. The goal is to ensure older helper functions do not bypass the saved design-matrix schema, structured prediction errors, or family capability gates introduced in Week 1.

## Findings and Changes

- `omnilss.prediction_runtime` no longer imports a missing `predictAll` symbol from `omnilss.prediction`.
- `predict_mean()`, `predict_distribution()`, `predict_quantile()`, and `predict_interval()` now delegate to schema-safe `predict_params()`.
- Runtime wrappers require the family `prediction` capability before computing predictions. Experimental prediction routes remain allowed for development compatibility, while unsupported or unknown families fail through the capability registry.
- Quantile and interval helpers now use all predicted distribution parameters instead of constructing a `mu`-only parameter dictionary.
- Regression tests cover the legacy wrappers and compare them with the canonical `GAMLSSModel.predict_params()` and `GAMLSSModel.predict_quantiles()` paths.

## Week 2 Status

Week 2 can now be treated as complete for the current production path:

1. Public artifact-schema examples and validator CLI were added previously.
2. Legacy prediction runtime wrappers have been audited and routed through schema-safe prediction.
3. Remaining work is future expansion rather than a Week 2 blocker: every newly discovered legacy prediction alias must be migrated to the same schema-safe path before release.

## Link to Week 3

This audit also prepares Week 3 by making prediction runtime behavior consult the capability registry. Method-route capability maps are now exposed in the machine-readable matrix so docs, tests, and strict routing share the same route-feature mapping.
