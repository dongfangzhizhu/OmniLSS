# Phase 4 Progress: FastAPI endpoints sharing persistent model registry

中文版本: [phase4-fastapi-registry-progress-2026-05-18_cn.md](phase4-fastapi-registry-progress-2026-05-18_cn.md)

## Completed in this step

- Added an optional FastAPI service entrypoint (`omnilss.api.http.fastapi_server.create_app`).
- Reused the gRPC persistent model registry (`omnilss.api.grpc.server.REGISTRY`) instead of introducing a parallel store.
- Added `GET /models` to list active model IDs.
- Added `DELETE /models/{model_id}` to remove stored models by ID.
- Added a health endpoint `GET /health` for parity with the stdlib HTTP boundary.
- Added focused tests validating that FastAPI endpoints use the shared registry object.

## Remaining Phase 4 work

- proto3 array payload extension with JSON compatibility guarantees.
- gRPC list/delete RPC definitions and restart-safe prediction integration tests.
