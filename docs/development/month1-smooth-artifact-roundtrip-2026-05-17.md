# Month 1 Smooth Artifact Roundtrip Progress (2026-05-17)

[中文版本](month1-smooth-artifact-roundtrip-2026-05-17_cn.md)

## Context

This note advances the Month 1 trustworthy-core workstream in the [six-month execution plan](six-month-execution-plan-2026-05-17.md), especially the requirement that smooth fit → predict → serialize → load → predict paths either roundtrip safely or fail with structured schema errors.

## Implementation Progress

- Smooth fit metadata now carries prediction-critical basis details: knot sequence, spline degree, penalty order, and the concrete basis smoother used when a formula alias such as `s()` resolves to a basis implementation.
- JSON model artifacts persist compact smooth metadata outside the training arrays, keeping training data redacted by default while retaining enough schema information to rebuild supported smooth prediction design blocks.
- The design-matrix schema now embeds `smooth_basis_metadata` for parameters that require smooth metadata, so schema consumers can audit whether a serialized artifact is capable of schema-safe smooth prediction.
- The prediction path accepts both live `SmoothDesignInfo` objects and JSON-loaded smooth metadata, and it raises `PredictionSchemaError` if required smooth metadata or variables are missing.
- A regression test now verifies `pb(x)` JSON save/load roundtrip predictions within the Month 1 acceptance tolerance of `rtol <= 1e-7`.

## Current Boundary

- Schema-safe smooth prediction is validated for `pb`/`ps` B-spline basis reconstruction.
- Other smoother classes remain guarded by explicit structured errors until their basis metadata and prediction reconstruction paths are validated.
- Full training data remains opt-in for JSON artifacts; prediction uses schema metadata rather than serialized training arrays.

## Next Steps

1. Extend schema-safe smooth prediction tests to `s(x, smoother="ps")` aliases and multi-parameter formulas.
2. Add negative tests for missing knots, missing smooth variables, and unsupported smoothers.
3. Promote the compact smooth metadata shape into the broader artifact-schema documentation once the remaining unsupported smoother boundaries are fully enumerated.
