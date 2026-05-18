# 第 1 月 Legacy Prediction 入口审计进展（2026-05-18）

[English version](month1-legacy-prediction-entrypoint-audit-2026-05-18.md)

本文推进 [六个月每周实现清单](six-month-weekly-implementation-checklist-2026-05-17_cn.md) 的第 2 周任务，以及 [六个月执行计划](six-month-execution-plan-2026-05-17_cn.md) 中第 1 月可信核心工作流。

## 范围

本次审计覆盖此前尚未完全纳入 schema-safe 生产路径的 R 对齐 legacy prediction 入口：

- `omnilss.predict_gamlss_23_12_21.predict()`；
- `omnilss.predictAll_22_08_22.predict_all()`，通过共享的单参数预测 helper 进入同一路径；
- 接受 `newdata` 或自行构造 prediction grid 的间接 prediction/reporting wrapper，包括 `prodist_data()`、`get_pef_data()`、scoring helper、`get_tgd_data()` 和 `extract_tgd_data()`。

## 已完成变更

- legacy `newdata` prediction 现在委托给 `GAMLSSModel.predict_params()` 使用的同一个 schema 检查设计矩阵构建器。
- 对于具备权威 schema 的模型 artifact，legacy `predict()` 和 `predict_all()` 现在会传播结构化 `PredictionSchemaError`，而不是回退到基于 term label 的重建设计矩阵。
- 仅对缺少完整 schema contract 的手工构造 legacy partial object 保留有界 fallback；当保存的参数 schema 已声明预期列数时禁止 fallback。
- schema 物化遇到 dot formula 时，如果 expanded term metadata 可用，会优先使用该 metadata，因为 literal `.` 在序列化后缺少原始 model frame 展开信息，不能安全复现。
- JSON artifact 现在会保留轻量 term metadata，包括 response name 和 term label，使 validation/report wrapper 能在加载后的 artifact 上运行，而不需要嵌入训练数据。

## 新增证据

- legacy `predict()` 回归测试验证未知 factor level 会从共享 prediction builder 抛出结构化 schema error。
- legacy `predict_all()` 回归测试验证从 JSON artifact 加载后缺失 smooth metadata 会抛出结构化 schema error。
- 间接 wrapper 回归测试验证 `prodist_data()` 和 `get_pef_data()` 会继续传播相同的 structured schema error，而不是吞掉或重写 prediction failure。
- reporting 与 validation wrapper 回归测试验证 `scoring.log_score()`、`get_tgd_data()` 和 `extract_tgd_data()` 在其委托的 prediction 路径遇到不可移植 smooth metadata 时，会保留同一个 `PredictionSchemaError` envelope。

## 第 2 周剩余跟进

- 当前 staged helper 中，接受 `newdata` 或通过共享 prediction 路径构造 prediction grid 的 prediction/reporting wrapper 审计已具备测试覆盖。
- 后续 plot/report helper 从 staged API 提升到 production API 时仍需持续监控；新增 helper 必须调用共享 schema-safe prediction builder，或明确说明其不执行 prediction。
- 已在 [模型 Artifact Schema 与验证](../api/model-artifact-schema_cn.md#artifact-与预测错误示例) 中补充公开示例，展示 legacy 入口 error envelope 与 validator CLI 输出。
