# Month 1 Capability Matrix Progress (2026-05-18)

> Chinese version: [month1-capability-matrix-progress-2026-05-18_cn.md](month1-capability-matrix-progress-2026-05-18_cn.md)
>
> Parent plan: [six-month-execution-plan-2026-05-17.md](six-month-execution-plan-2026-05-17.md)

## Scope

This note records progress against Month 1 / Workstream D3: **RS/CG/JAX Capability Matrix**.

## Implemented Progress

- The runtime capability registry now exposes a machine-readable `capability_matrix()` snapshot containing the feature list, the method-routing map, strict-mode policy flags, and per-family evidence statuses.
- `tools/generate_capability_matrix.py` now generates `family-capability-matrix-2026-05-18.json` for reuse by documentation, release bundles, and API clients.
- The gRPC server now exposes `CapabilityService.CapabilityMatrix` so service clients can retrieve the same runtime capability matrix through the API boundary.
- The gRPC capability service now also exposes `RouteCapability`, allowing service clients to preflight method/family admission with the same report used by HTTP metadata and `gamlss()` strict routing.
- Production-safe families can mark core routes as validated; the current baseline validates `NO` for `rs_fit`, `prediction`, `r_consistency`, and `production_safe`.
- `gamlss(..., strict_capabilities=True)` now rejects experimental method/family routes and admits only validated capability features. The fitting code uses the same `require_method_route()` mapping that is emitted in the matrix.
- Default development behavior remains unchanged: experimental routes are still allowed unless strict capability mode is requested.
- Method-routing tests cover both strict acceptance for the validated `NO` RS route and strict rejection for an experimental GA RS route.
- Generated JSON artifacts, HTTP metadata responses, gRPC capability matrix responses, and gRPC route-capability reports now carry the same method-to-feature routing contract as `gamlss()`.
- `method_route_capability_report()` now provides a JSON-friendly route-admission report for service boundaries before future async fit jobs are scheduled, `gamlss()` uses the same helper for runtime gating, and the HTTP metadata boundary exposes the report for client preflight checks.

## Remaining D3 Work

- Expand validation evidence until more core families can move from experimental to validated for specific routes.
- Wire `method_route_capability_report()` into future async job admission once the service job runtime is introduced.
