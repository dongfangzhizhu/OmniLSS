# 第 1 月服务端 Prediction 错误 Envelope 进展（2026-05-18）

[English version](month1-service-prediction-error-envelope-2026-05-18.md)

本文推进 [六个月每周实现清单](six-month-weekly-implementation-checklist-2026-05-17_cn.md) 中第 2 周 prediction hardening 工作，并确保 service boundary 与 schema-safe prediction runtime 保持一致。

## 范围

- schema-safe prediction error 的顶层 Python API export。
- prediction schema validation 失败时 gRPC `PredictResponse.error` 的行为。
- 面向客户端路由的公开 API 文档。

## 已完成变更

- `PredictionSchemaError` 与 `build_prediction_design_matrix()` 可从顶层 `omnilss` package 导入，客户端无需依赖 private helper。
- 对于 `PredictionSchemaError` failure，gRPC response error string 会以 JSON 形式保留结构化 prediction schema envelope；其他 exception 仍保持纯文本兼容。
- gRPC boundary 文档现在说明客户端应先尝试把 `PredictResponse.error` 解析为 JSON，并在字段存在时基于 `type` 与 `code` 路由。

## 后续跟进

- 当 HTTP service 从 metadata-only endpoint 扩展到 prediction endpoint 后，为 HTTP prediction endpoint 添加同样的结构化 envelope。
- 在下一次允许破坏兼容性的 RPC schema 修订中，考虑新增 protobuf `error_json` 或 typed error message。
