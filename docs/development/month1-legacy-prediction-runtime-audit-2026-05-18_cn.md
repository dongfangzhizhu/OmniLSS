# 第 1 月 Legacy Prediction Runtime 审计（2026-05-18）

> English version: [month1-legacy-prediction-runtime-audit-2026-05-18.md](month1-legacy-prediction-runtime-audit-2026-05-18.md)
>
> 父计划：[six-month-execution-plan-2026-05-17_cn.md](six-month-execution-plan-2026-05-17_cn.md)
>
> 每周 checklist：[six-month-weekly-implementation-checklist-2026-05-17_cn.md](six-month-weekly-implementation-checklist-2026-05-17_cn.md)

## 范围

本记录关闭当前 runtime 表面的第 2 周 legacy prediction 入口审计。目标是确保旧 helper 不会绕过第 1 周引入的已保存 design-matrix schema、结构化预测错误和 family capability 门禁。

## 发现与变更

- `omnilss.prediction_runtime` 不再从 `omnilss.prediction` 导入不存在的 `predictAll` 符号。
- `predict_mean()`、`predict_distribution()`、`predict_quantile()` 和 `predict_interval()` 现在委托给 schema-safe 的 `predict_params()`。
- Runtime wrapper 在执行预测前要求 family 具备 `prediction` capability。为保持开发兼容性，experimental prediction route 仍允许使用；unsupported 或 unknown family 会通过 capability registry 失败。
- Quantile 和 interval helper 现在使用所有预测出的分布参数，而不是构造仅包含 `mu` 的参数字典。
- 回归测试覆盖 legacy wrapper，并将其结果与规范的 `GAMLSSModel.predict_params()` 和 `GAMLSSModel.predict_quantiles()` 路径对齐。

## 第 2 周状态

当前生产路径下，第 2 周可视为已完成：

1. 公开 artifact-schema 示例和 validator CLI 已在此前加入。
2. Legacy prediction runtime wrapper 已完成审计，并路由到 schema-safe prediction。
3. 剩余事项属于未来扩展而非第 2 周 blocker：每个新发现的 legacy prediction alias 都必须在 release 前迁移到同一 schema-safe 路径。

## 与第 3 周的衔接

本审计也为第 3 周做准备：prediction runtime 行为现在会查询 capability registry。Method-route capability 映射已暴露到机器可读矩阵中，因此文档、测试和 strict routing 共享同一份 route-feature 映射。
