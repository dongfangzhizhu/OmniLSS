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
- `GET /metrics`: Prometheus-style counters for metadata endpoint requests.

All HTTP metadata responses include `X-Request-ID`; inbound request IDs are propagated when provided. Embedded deployments can also pass `serve(..., event_sink=...)` to receive structured request events for prototype-safe logging or test observability; event-sink exceptions are isolated and do not break metadata responses.

## HTTP error envelope

Unsupported or unknown HTTP metadata requests return a structured JSON envelope while preserving `X-Request-ID`:

```json
{
  "success": false,
  "error": {
    "type": "http_error",
    "code": "method_not_allowed",
    "message": "HTTP POST is not enabled for '/predict'; fit/predict endpoints require authn, limits, and structured logging before exposure"
  },
  "request_id": "example-request-id"
}
```

Prototype HTTP POST handling first applies `Content-Length` validation and the payload-limit gate, returning `400 invalid_content_length` for malformed lengths and `413 payload_too_large` for oversized requests. POST routes that pass the size gate intentionally return `405 method_not_allowed` until authentication, payload limits, and structured logging are implemented.

These endpoints are prototype-safe metadata endpoints only; fit/predict HTTP endpoints still require authentication, request IDs, limits, and structured logging before production exposure.
