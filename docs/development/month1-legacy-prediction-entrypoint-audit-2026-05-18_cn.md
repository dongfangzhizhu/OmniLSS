# 第 1 月 Legacy Prediction 入口审计进展（2026-05-18）

[English version](month1-legacy-prediction-entrypoint-audit-2026-05-18.md)

本文推进 [六个月每周实现清单](six-month-weekly-implementation-checklist-2026-05-17_cn.md) 的第 2 周任务，以及 [六个月执行计划](six-month-execution-plan-2026-05-17_cn.md) 中第 1 月可信核心工作流。

## 范围

本次审计覆盖此前尚未完全纳入 schema-safe 生产路径的 R 对齐 legacy prediction 入口：

- `omnilss.predict_gamlss_23_12_21.predict()`；
- `omnilss.predictAll_22_08_22.predict_all()`，通过共享的单参数预测 helper 进入同一路径。

## 已完成变更

- legacy `newdata` prediction 现在委托给 `GAMLSSModel.predict_params()` 使用的同一个 schema 检查设计矩阵构建器。
- 对于具备权威 schema 的模型 artifact，legacy `predict()` 和 `predict_all()` 现在会传播结构化 `PredictionSchemaError`，而不是回退到基于 term label 的重建设计矩阵。
- 仅对缺少完整 schema contract 的手工构造 legacy partial object 保留有界 fallback；当保存的参数 schema 已声明预期列数时禁止 fallback。
- schema 物化遇到 dot formula 时，如果 expanded term metadata 可用，会优先使用该 metadata，因为 literal `.` 在序列化后缺少原始 model frame 展开信息，不能安全复现。

## 新增证据

- legacy `predict()` 回归测试验证未知 factor level 会从共享 prediction builder 抛出结构化 schema error。
- legacy `predict_all()` 回归测试验证从 JSON artifact 加载后缺失 smooth metadata 会抛出结构化 schema error。

## 第 2 周剩余跟进

- 将审计扩展到间接调用 prediction 且接受 `newdata` 的高级便捷 wrapper，尤其是绘图和报告 helper。
- 已在 [模型 Artifact Schema 与验证](../api/model-artifact-schema_cn.md#artifact-与预测错误示例) 中补充公开示例，展示 legacy 入口 error envelope 与 validator CLI 输出。
