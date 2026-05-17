# Phase 1 Serving Runtime Prototype

[中文版本](serving-runtime-prototype_cn.md)

Endpoints:
- Quantile inference API
- Interval prediction API
- Uncertainty API

SLO target:
- P95 latency tracking enabled in benchmark telemetry


Metadata endpoints implemented in the lightweight stdlib HTTP boundary:

- `GET /health` / `GET /healthz`: service liveness check.
- `GET /capabilities` / `GET /capability-matrix`: runtime family capability matrix.

These endpoints are prototype-safe metadata endpoints only; fit/predict HTTP endpoints still require authentication, request IDs, limits, and structured logging before production exposure.
