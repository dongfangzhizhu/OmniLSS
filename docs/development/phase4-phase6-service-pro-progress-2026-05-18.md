# Phase 4 and Phase 6 Service/Pro Progress — 2026-05-18

中文版本: [phase4-phase6-service-pro-progress-2026-05-18_cn.md](phase4-phase6-service-pro-progress-2026-05-18_cn.md)

## Completed work

- The FastAPI server now uses the same SQLite-backed persistent model registry and artifact directory controls as the Core gRPC server. Operators can set `OMNILSS_MODEL_STORE_DIR` and `OMNILSS_MODEL_DB_PATH` to keep fitted model artifacts restart-safe.
- The REST API now exposes model lifecycle endpoints through `GET /models` and `DELETE /models/{model_id}` in addition to fit, predict, and diagnostics.
- The REST distribution-selection endpoint now fits candidate families and reports deviance, AIC, BIC, GAIC, parameter count, and iterations instead of returning the first candidate as a placeholder.
- The Pro gRPC contract mirror now includes list/delete model RPCs and efficient repeated-double array payloads for prediction while retaining JSON compatibility.
- The Pro client now exposes `list_models()` and `delete_model()` and sends both JSON and array-column prediction payloads.
- Pro AutoML now ranks candidate families by deviance, AIC, BIC, and GAIC through Core client calls only, and adds bootstrap deviance confidence intervals through repeated resampled Core fits.

## Boundary notes

- `omnilss-pro` still does not import `omnilss`; all Pro automation operates through an injected `OmniLSSCoreClient`-compatible boundary.
- REST and gRPC model lifecycle state share the Core registry implementation, so lifecycle behavior remains consistent across service surfaces.
