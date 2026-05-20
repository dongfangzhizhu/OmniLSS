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

Use `reason_detail` for user-facing text.

## Practical guidance

- If reason is `gpu_crossover_not_reached`, keep `RS` unless benchmark data
  demonstrates a lower threshold for your hardware.
- If reason is `family_not_jax_supported`, `RS` is expected and preferred.
- Avoid forcing `RS_JAX` in small-`n` workloads.
