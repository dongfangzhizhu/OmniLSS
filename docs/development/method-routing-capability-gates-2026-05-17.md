# Method Routing Capability Gates (2026-05-17)

[中文版本](method-routing-capability-gates-2026-05-17_cn.md)

This note advances Workstream D3 from the [six-month execution plan](six-month-execution-plan-2026-05-17.md) and completes the first follow-up from the [family capability registry note](family-capability-registry-2026-05-17.md): `gamlss()` now checks the family capability registry before entering method-specific fitting backends.

## Runtime behavior

`gamlss()` validates the resolved method and family before starting expensive fitting work. The same mapping is exported by `method_capability_features()` and embedded in `capability_matrix()["method_capability_features"]`:

| Method | Capability feature checked | Experimental allowed? | Unsupported behavior |
|---|---|---:|---|
| `RS` | `rs_fit` | Yes | Raises `FamilyCapabilityError` |
| `RS_JAX` | `rs_jax_fit` | Yes | Raises `FamilyCapabilityError` before the JAX backend is called |
| `CG` | `cg_fit` | Yes | Raises `FamilyCapabilityError` |
| `MIXED` | `cg_fit` | Yes | Raises `FamilyCapabilityError` |
| `joint` / `JOINT` | `cg_fit` | Yes | Raises `FamilyCapabilityError` before initialization/refinement |
| `lbfgs` / `LBFGS` | `cg_fit` | Yes | Raises `FamilyCapabilityError` before initialization/refinement |
| `auto` | Resolves to `RS` or `RS_JAX`, then checks that resolved route | Yes | Unsupported `RS_JAX` routes cannot proceed |

The current policy still permits experimental features because the existing project defaults are research/development oriented. The important change is that `unsupported` routes now fail through the capability registry before backend work begins. Service boundaries can call `method_route_capability_report()` to receive the same decision as a JSON-friendly admission report with `ok`, `feature`, `status`, `code`, and `message` fields.

## Why this matters

- Unsupported method/family combinations fail fast.
- The routing layer now uses the same capability source of truth as documentation, tests, generated JSON artifacts, gRPC responses, and HTTP metadata responses.
- `RS_JAX` no longer relies solely on backend-local support checks; unsupported families are rejected at the public `gamlss()` boundary.
- Future service APIs can reuse the same gate before scheduling asynchronous jobs.

## Follow-up work

1. Completed: `strict_capabilities=True` rejects `experimental` features unless callers explicitly use the default research/development mode.
2. Completed: JSON artifacts attach family capability snapshots and compatibility reports.
3. Completed: HTTP/gRPC service endpoints expose the runtime capability matrix.
4. Completed: generated machine-readable capability matrix artifacts include the runtime method-routing map.
5. Completed: `method_route_capability_report()` exposes the route-admission decision for future service job scheduling.
