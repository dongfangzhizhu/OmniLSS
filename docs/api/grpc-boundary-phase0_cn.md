# gRPC Runtime Boundary（Phase 0）

[English version](grpc-boundary-phase0.md)

## 服务表面

- `Fit` request/response
- `Predict` request/response
- `Sample` request/response
- `CapabilityMatrix` request/response，与 package 和 HTTP metadata API 暴露同一份 runtime capability matrix。payload 包含 family feature evidence、`method_capability_features` 和 `strict_capability_policy`。
- `RouteCapability` request/response，用于 method/family preflight check。payload 与 `method_route_capability_report()` 和 HTTP `/route-capability` 返回的 JSON report 一致，包含 strict 模式 admission decision，方便 client 在未来提交 fit job 前先检查路由。

## 边界原则

面向客户端的服务通过窄 RPC contract 调用 core runtime，以减少产品层与统计核心之间的耦合。

## 目标

支持远程 serving，并为后续安全、异步任务、模型 registry 和可观测性工作保留清晰边界。

## Route capability preflight

`RouteCapabilityRequest` 接收 `family`、`method` 和 `strict`。成功响应会把 route-admission report 序列化到 `RouteCapabilityResponse.report_json`；如果请求缺少 family 或 method，则返回 `success=false` 与纯文本 error，同时保持相同的 response shape。

## Prediction 错误 envelope

为保持 protobuf 兼容，`PredictResponse.error` 仍然是字符串。当 prediction 因请求无法复现已保存模型 schema 而失败时，服务端会把 `PredictionSchemaError.to_dict()` payload 序列化为 JSON 字符串：

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

客户端应先尝试将 `error` 解析为 JSON，并基于 `type == "prediction_schema_error"` 以及 `code` 进行路由。Phase 0 / 第 1 月边界保持向后兼容，因此非 schema runtime failure 仍可能以纯文本返回。
