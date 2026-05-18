# gRPC Runtime Boundary (Phase 0)

[中文版本](grpc-boundary-phase0_cn.md)

## Service surface
- `Fit` request/response
- `Predict` request/response
- `Sample` request/response
- `CapabilityMatrix` request/response for the same runtime capability matrix exposed by package and HTTP metadata APIs. The payload includes family feature evidence, `method_capability_features`, and `strict_capability_policy`.

## Boundary principle
Client-facing services call into core runtime through a narrow RPC contract.

## Goal
Enable remote serving and reduce coupling between product layers.

## Prediction error envelope

`PredictResponse.error` remains a string for protobuf compatibility. When prediction fails because the request cannot reproduce the saved model schema, the server serializes the `PredictionSchemaError.to_dict()` payload as JSON inside that string:

```json
{
  "type": "prediction_schema_error",
  "code": "unseen_factor_levels",
  "parameter": "mu",
  "term": "factor(grp)",
  "reason": "unseen factor levels ['c']",
  "message": "Factor term 'factor(grp)' contains unseen levels ['c']"
}
```

Clients should first attempt to parse `error` as JSON and route on `type == "prediction_schema_error"` plus `code`. Non-schema runtime failures may still be returned as plain text while the Phase 0/Month 1 boundary remains backward-compatible.
