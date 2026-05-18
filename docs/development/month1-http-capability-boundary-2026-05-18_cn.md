# 第 1 月 HTTP 能力矩阵边界进展（2026-05-18）

> English version: [month1-http-capability-boundary-2026-05-18.md](month1-http-capability-boundary-2026-05-18.md)
>
> 父计划：[six-month-execution-plan-2026-05-17_cn.md](six-month-execution-plan-2026-05-17_cn.md)

## 范围

本文推进第 1 月 / 工作流 D3，并为第 3 月 / 工作流 D7 做准备：通过轻量 HTTP 服务边界暴露能力元数据。

## 已实现进展

- 新增 `omnilss.api.http.server`，这是一个不依赖外部 Web 框架的标准库 HTTP 元数据服务。
- 新增 `GET /health` 和 `GET /healthz`，用于编排系统的冒烟检查。
- 新增 `GET /capabilities` 和 `GET /capability-matrix`，返回与 package API、生成 JSON 工件和 gRPC 服务一致的运行时 `capability_matrix()` 负载。
- HTTP 响应新增 `X-Request-ID` 透传/生成能力，便于请求追踪。
- 新增可选 `event_sink` structured request event hook，用于 prototype-safe logging 和测试观测；sink exception 会被隔离。
- 新增 `GET /metrics`，以 Prometheus 风格 counter 暴露元数据请求计数。
- 未知 endpoint、暂不开放的 POST route、非法 `Content-Length` 和超过 payload limit 的 POST request 现在返回结构化 HTTP error envelope，并继续透传 `X-Request-ID`。
- 新增测试：在本地临时端口启动 HTTP 服务，并验证 health、capability matrix、request ID、metrics、payload-limit、invalid content length、structured event hook 和结构化 error envelope 响应。

## 剩余工作

- 在暴露 fit/predict HTTP 端点前增加认证、生产级结构化日志 sink 和速率限制；现有 POST payload limit gate 与 prototype `event_sink` 会保留，在此之前未超限 POST route 保持 `405 method_not_allowed`。
- 当服务边界不再只是 prototype 时，在 `docs/api/` 中记录公开 HTTP 响应 schema。
- 每当 registry schema 变化时，通过测试保持 HTTP 与 gRPC capability 响应同步。
