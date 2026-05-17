# Method Routing Capability Gates (2026-05-17)

[中文版本](method-routing-capability-gates-2026-05-17_cn.md)

This note advances Workstream D3 from the [six-month execution plan](six-month-execution-plan-2026-05-17.md) and completes the first follow-up from the [family capability registry note](family-capability-registry-2026-05-17.md): `gamlss()` now checks the family capability registry before entering method-specific fitting backends.

## Runtime behavior

`gamlss()` validates the resolved method and family before starting expensive fitting work:

| Method | Capability feature checked | Experimental allowed? | Unsupported behavior |
|---|---|---:|---|
| `RS` | `rs_fit` | Yes | Raises `FamilyCapabilityError` |
| `RS_JAX` | `rs_jax_fit` | Yes | Raises `FamilyCapabilityError` before the JAX backend is called |
| `CG` | `cg_fit` | Yes | Raises `FamilyCapabilityError` |
| `MIXED` | `cg_fit` | Yes | Raises `FamilyCapabilityError` |
| `joint` / `JOINT` | `cg_fit` | Yes | Raises `FamilyCapabilityError` before initialization/refinement |
| `lbfgs` / `LBFGS` | `cg_fit` | Yes | Raises `FamilyCapabilityError` before initialization/refinement |
| `auto` | Resolves to `RS` or `RS_JAX`, then checks that resolved route | Yes | Unsupported `RS_JAX` routes cannot proceed |

The current policy still permits experimental features because the existing project defaults are research/development oriented. The important change is that `unsupported` routes now fail through the capability registry before backend work begins.

## Why this matters

- Unsupported method/family combinations fail fast.
- The routing layer now uses the same capability source of truth as documentation and tests.
- `RS_JAX` no longer relies solely on backend-local support checks; unsupported families are rejected at the public `gamlss()` boundary.
- Future service APIs can reuse the same gate before scheduling asynchronous jobs.

## Follow-up work

1. Add an optional strict production mode where `experimental` features are rejected unless explicitly enabled.
2. Attach capability snapshots to serialized model metadata.
3. Expose the capability matrix through HTTP/gRPC service endpoints.
4. Generate a machine-readable capability matrix artifact from the runtime registry.
