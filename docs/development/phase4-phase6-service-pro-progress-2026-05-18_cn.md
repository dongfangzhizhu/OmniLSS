# 第 4 阶段与第 6 阶段服务/Pro 进展 — 2026-05-18

English version: [phase4-phase6-service-pro-progress-2026-05-18.md](phase4-phase6-service-pro-progress-2026-05-18.md)

## 已完成工作

- FastAPI 服务器现在使用与 Core gRPC 服务器相同的 SQLite 持久化模型注册表和模型产物目录控制。运维人员可以设置 `OMNILSS_MODEL_STORE_DIR` 与 `OMNILSS_MODEL_DB_PATH`，使拟合后的模型产物具备重启安全性。
- REST API 现在除 fit、predict、diagnostics 外，还提供 `GET /models` 与 `DELETE /models/{model_id}` 模型生命周期端点。
- REST 分布选择端点现在会拟合候选分布，并报告 deviance、AIC、BIC、GAIC、参数数量和迭代次数，不再只返回第一个候选项作为占位实现。
- Pro 侧 gRPC 契约镜像现在包含 list/delete model RPC，并为预测加入高效的 repeated-double 数组载荷，同时保留 JSON 兼容性。
- Pro 客户端现在提供 `list_models()` 与 `delete_model()`，并在预测请求中同时发送 JSON 与数组列载荷。
- Pro AutoML 现在仅通过 Core `BatchFit` 客户端调用按 deviance、AIC、BIC、GAIC 对候选分布排序，并通过 batch-fit 重采样提供 bootstrap deviance 置信区间。

## 边界说明

- `omnilss-pro` 仍然不导入 `omnilss`；所有 Pro 自动化都通过注入的 `OmniLSSCoreClient` 兼容边界运行。
- REST 与 gRPC 模型生命周期状态共享 Core 注册表实现，因此两个服务表面的生命周期行为保持一致。
