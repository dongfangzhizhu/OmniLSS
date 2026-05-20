# Method Routing Reason Guide

This guide explains how to inspect and interpret automatic method routing
(`RS` vs `RS_JAX`) in OmniLSS.

## Where to read routing metadata

When `gamlss(..., method="auto")` or `gamlss(..., method="RS")` is used,
OmniLSS stores routing details in:

- `model.additional_slots["method_routing"]`

Expected keys:

- `requested_method`
- `selected_method`
- `reason`
- `reason_detail`
- `backend`
- `threshold`
- `n_obs`
- `family`
- `schema_version`

## Reason codes

Common values of `reason` include:

- `cpu_backend_prefers_numpy_rs`
- `family_not_jax_supported`
- `gpu_crossover_not_reached`
- `gpu_crossover_reached`
- `tpu_crossover_not_reached`
- `tpu_crossover_reached`
- `auto_method_disabled`
- `force_jax_enabled`
- `explicit_method_requested`

Use `reason_detail` for user-facing text.

For explicit `method="RS_JAX"`, the reason is `explicit_method_requested` to indicate that auto crossover logic was intentionally bypassed.

## Practical guidance

- If reason is `gpu_crossover_not_reached`, keep `RS` unless benchmark data
  demonstrates a lower threshold for your hardware.
- If reason is `family_not_jax_supported`, `RS` is expected and preferred.
- Avoid forcing `RS_JAX` in small-`n` workloads.


## Model API helpers

For model-level diagnostics, use:

- `model.get_method_routing()`
- `model.get_method_routing_reason()`
- `model.get_method_routing_reason_detail()`
- `model.get_method_routing_summary()`
- `model.explain_method_routing()`
- `model.get_method_routing_confidence()`
