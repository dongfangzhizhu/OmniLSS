# Month 1 Service Prediction Error Envelope Progress (2026-05-18)

[中文版本](month1-service-prediction-error-envelope-2026-05-18_cn.md)

This note advances the Week 2 prediction-hardening work in the [six-month weekly implementation checklist](six-month-weekly-implementation-checklist-2026-05-17.md) and keeps the service boundary aligned with the schema-safe prediction runtime.

## Scope

- Top-level Python API exports for schema-safe prediction errors.
- gRPC `PredictResponse.error` behavior when prediction schema validation fails.
- Public API documentation for client routing.

## Completed Changes

- `PredictionSchemaError` and `build_prediction_design_matrix()` are available from the top-level `omnilss` package, so clients do not need to import private helpers.
- gRPC response error strings preserve the structured prediction schema envelope as JSON for `PredictionSchemaError` failures while retaining plain-text compatibility for other exceptions.
- The gRPC boundary documentation now tells clients to parse `PredictResponse.error` as JSON first and route on `type` plus `code` when present.

## Remaining Follow-Up

- Add the same structured envelope to future HTTP prediction endpoints when the service moves beyond metadata-only endpoints.
- Consider a protobuf `error_json` or typed error message in the next non-backward-compatible RPC schema revision.
