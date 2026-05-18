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
- The capability matrix schema version is now exposed as `CAPABILITY_MATRIX_VERSION = 3`, so adding the `method_routes` compatibility key is visible to clients instead of silently changing the version-2 payload.
- The public `method_route_feature()` and `require_method_route()` helpers are now restored, and `capability_matrix()` again includes the backward-compatible `method_routes` alias so generated artifacts, tests, service metadata, and documented strict routing share one importable contract.
- Added `tools/validate_capability_matrix.py` and `validate_capability_matrix_payload()` so generated matrix artifacts can be checked for schema version, method-route alias drift, policy drift, family coverage, and feature-status validity before release bundles or service metadata reuse them.
- The matrix validator now returns structured reports for unreadable files and malformed JSON as well as schema drift, so release checks receive one machine-readable failure envelope instead of uncaught file/parse exceptions.
- `method_route_capability_report()` now provides a JSON-friendly route-admission report for service boundaries before future async fit jobs are scheduled, `gamlss()` uses the same helper for runtime gating, and the HTTP metadata boundary exposes the report for client preflight checks.

## D3 Closure and Deferred Work

The Week 3 capability-gate implementation is complete for the current Month 1 scope: runtime gates, generated artifacts, HTTP/gRPC metadata, schema versioning, and validator checks share the same method-routing contract.

Deferred work is explicitly assigned to later roadmap items:

- Expanding validation evidence so more core families can move from experimental to validated belongs to Month 2 validation work.
- Wiring `method_route_capability_report()` into async job admission belongs to Month 3 service job-runtime work.
