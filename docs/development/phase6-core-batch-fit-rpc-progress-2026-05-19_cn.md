# 第六阶段进展：面向 Pro AutoML 的 Core Batch-Fit RPC — 2026-05-19

English version: [phase6-core-batch-fit-rpc-progress-2026-05-19.md](phase6-core-batch-fit-rpc-progress-2026-05-19.md)

## 本步骤已完成

- 在 Core 与 Pro 的 `fit.proto` 契约中新增 `BatchFitRequest`、`BatchFitResponse` 以及 `FitService.BatchFit` RPC，同时保留现有单模型 `Fit` RPC。
- 重新生成 Core 与 Pro protobuf 模块和 fallback gRPC service wrapper，使不具备 `grpcio-tools` 的环境也一致暴露新的批量方法。
- 增加 Core server `BatchFit` 实现：逐个执行 `FitRequest`，将每个成功模型保存到现有 registry，并返回逐模型 `FitResponse` 记录。
- 在 Pro 侧新增 `OmniLSSCoreClient.batch_fit()`，并让 Pro AutoML 排名与 bootstrap helper 使用 Core batch-fit 调用，而不是逐候选或逐重采样调用 `fit()`。
- 增加测试，证明 Core service 可处理多请求 batch fitting，且 Pro AutoML 使用 batch 边界。

## 边界说明

`omnilss-pro` 仍只导入自身镜像的 protobuf stub 与 gRPC runtime，不导入 `omnilss`；所有 AutoML 与 bootstrap 工作都通过 `OmniLSSCoreClient.batch_fit()` 跨越 Core 边界。
