# gRPC Runtime Boundary (Phase 0)

[中文版本](grpc-boundary-phase0_cn.md)

## Service surface
- `Fit` request/response
- `Predict` request/response
- `Sample` request/response

## Boundary principle
Client-facing services call into core runtime through a narrow RPC contract.

## Goal
Enable remote serving and reduce coupling between product layers.
