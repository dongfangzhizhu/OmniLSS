# Phase 6 Progress: Core Batch-Fit RPC for Pro AutoML — 2026-05-19

中文版本: [phase6-core-batch-fit-rpc-progress-2026-05-19_cn.md](phase6-core-batch-fit-rpc-progress-2026-05-19_cn.md)

## Completed in this step

- Extended the Core and Pro `fit.proto` contracts with `BatchFitRequest`, `BatchFitResponse`, and the `FitService.BatchFit` RPC while preserving the existing single-model `Fit` RPC.
- Regenerated the Core and Pro protobuf modules and fallback gRPC service wrappers so non-`grpcio-tools` environments expose the new batch method consistently.
- Added a Core server `BatchFit` implementation that executes each `FitRequest`, persists each successful model in the existing registry, and returns per-model `FitResponse` records.
- Added `OmniLSSCoreClient.batch_fit()` on the Pro side and changed Pro AutoML ranking and bootstrap helpers to use Core batch-fit calls instead of per-candidate/per-resample `fit()` loops.
- Added tests proving the Core service handles multi-request batch fitting and Pro AutoML uses the batch boundary.

## Boundary notes

`omnilss-pro` still imports only its mirrored protobuf stubs and the gRPC runtime. It does not import `omnilss`; all AutoML and bootstrap work crosses the Core boundary through `OmniLSSCoreClient.batch_fit()`.
