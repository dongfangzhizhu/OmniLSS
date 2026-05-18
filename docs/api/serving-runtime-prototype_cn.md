# Phase 1 Serving Runtime Prototype（中文）

[English version](serving-runtime-prototype.md)

计划中的服务端能力：

- Quantile inference API
- Interval prediction API
- Uncertainty API

- [ ] 完成全文中文翻译
- [ ] 与英文版本保持同步


轻量标准库 HTTP 边界已实现的元数据端点：

- `GET /health` / `GET /healthz`：服务存活检查。
- `GET /capabilities` / `GET /capability-matrix`：运行时分布族能力矩阵。
- `GET /metrics`：以 Prometheus 风格 counter 暴露元数据端点请求计数。

所有 HTTP 元数据响应都会包含 `X-Request-ID`；如果入站请求提供 request ID，则会透传。

这些端点目前只是 prototype-safe 的元数据端点；fit/predict HTTP 端点在生产暴露前仍需要认证、request ID、限制和结构化日志。
