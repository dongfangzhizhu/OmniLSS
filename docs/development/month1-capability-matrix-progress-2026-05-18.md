# Month 1 Capability Matrix Progress (2026-05-18)

> Chinese version: [month1-capability-matrix-progress-2026-05-18_cn.md](month1-capability-matrix-progress-2026-05-18_cn.md)
>
> Parent plan: [six-month-execution-plan-2026-05-17.md](six-month-execution-plan-2026-05-17.md)

## Scope

This note records progress against Month 1 / Workstream D3: **RS/CG/JAX Capability Matrix**.

## Implemented Progress

- The runtime capability registry now exposes a machine-readable `capability_matrix()` snapshot containing the feature list, method-route feature map, and per-family evidence statuses.
- `tools/generate_capability_matrix.py` now generates `family-capability-matrix-2026-05-18.json` for reuse by documentation, release bundles, and API clients.
- The gRPC server now exposes `CapabilityService.CapabilityMatrix` so service clients can retrieve the same runtime capability matrix through the API boundary.
- Production-safe families can mark core routes as validated; the current baseline validates `NO` for `rs_fit`, `prediction`, `r_consistency`, and `production_safe`.
- `gamlss(..., strict_capabilities=True)` now rejects experimental method/family routes and admits only validated capability features. The fitting code uses the same `require_method_route()` mapping that is emitted in the matrix.
- Default development behavior remains unchanged: experimental routes are still allowed unless strict capability mode is requested.
- Method-routing tests cover both strict acceptance for the validated `NO` RS route, strict rejection for an experimental GA RS route, and the public method-route feature map.
- Legacy prediction runtime wrappers now require the family `prediction` capability and delegate to schema-safe `predict_params()` before producing means, quantiles, intervals, or distribution payloads. See [month1-legacy-prediction-runtime-audit-2026-05-18.md](month1-legacy-prediction-runtime-audit-2026-05-18.md).

## Remaining D3 Work

- Keep HTTP `/capabilities` response-shape docs synchronized with the expanded method-route feature map.
- Expand validation evidence until more core families can move from experimental to validated for specific routes.
