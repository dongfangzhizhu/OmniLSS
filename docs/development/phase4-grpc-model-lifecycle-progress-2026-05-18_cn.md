# 第四阶段进展：gRPC 模型生命周期 RPC

English version: [phase4-grpc-model-lifecycle-progress-2026-05-18.md](phase4-grpc-model-lifecycle-progress-2026-05-18.md)

## 本步骤已完成

- 扩展 `fit.proto`，新增模型生命周期消息：
  - `ListModelsRequest` / `ListModelsResponse`
  - `DeleteModelRequest` / `DeleteModelResponse`
- 扩展 `FitService`，新增两个 RPC：
  - `ListModels`
  - `DeleteModel`
- 使用项目工具的 fallback 路径重新生成 gRPC stubs。
- 在 `omnilss.api.grpc.server` 中实现服务处理逻辑，并委托给共享注册表 helper。
- 新增直接服务测试，验证 list/delete 行为。

## 第四阶段剩余工作

- 为 fit/predict 增加高效数组载荷 schema，并保证严格 JSON 兼容。
- 增加跨进程边界的重启安全预测集成测试。
