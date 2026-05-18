# Phase 1 Serving Runtime Prototype（中文）

[English version](serving-runtime-prototype.md)

计划中的服务端能力：

- Quantile inference API
- Interval prediction API
- Uncertainty API

SLO 目标：

- 在 benchmark telemetry 中启用 P95 latency tracking。

轻量标准库 HTTP 边界已实现的 metadata endpoints：

- `GET /health` / `GET /healthz`：service liveness check。
- `GET /capabilities` / `GET /capability-matrix`：runtime family capability matrix。
- `GET /metrics`：以 Prometheus 风格 counter 暴露 metadata endpoint request 计数。

所有 HTTP metadata response 都包含 `X-Request-ID`；如果入站 request 提供 request ID，则会透传。嵌入式 deployment 也可以通过 `serve(..., event_sink=...)` 接收结构化 request event，用于 prototype-safe logging 或测试观测；event sink exception 会被隔离，不会破坏 metadata response。

## HTTP 错误 envelope

不支持或未知的 HTTP metadata request 会返回结构化 JSON envelope，并保留 `X-Request-ID`：

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

当前 prototype 在处理 POST 时会先执行 `Content-Length` validation 与 payload limit gate；非法 `Content-Length` 会返回 `400 invalid_content_length`，超过配置上限会返回 `413 payload_too_large`。在 authentication、payload limits 和 structured logging 完成之前，未超限的 prototype HTTP POST routes 会有意返回 `405 method_not_allowed`。

这些 endpoints 目前只是 prototype-safe metadata endpoints；fit/predict HTTP endpoints 在生产暴露前仍需要 authentication、request IDs、limits 和 structured logging。
