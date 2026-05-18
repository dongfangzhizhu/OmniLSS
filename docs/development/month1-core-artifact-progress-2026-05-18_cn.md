# 第 1 月核心模型工件进展（2026-05-18）

> English version: [month1-core-artifact-progress-2026-05-18.md](month1-core-artifact-progress-2026-05-18.md)
>
> 父计划：[six-month-execution-plan-2026-05-17_cn.md](six-month-execution-plan-2026-05-17_cn.md)

## 范围

本文记录第 1 月 / 工作流 D1：**模型工件与设计矩阵 Schema v2** 的下一步实现进展。

## 已实现进展

- JSON 模型工件默认不再保存完整训练数据。
- 只有在调用 `save_model_json(..., include_training_data=True)` 时，才写入训练响应数组。
- 当训练数据被移除时，序列化后的 call 元数据会标记 `training_data_omitted=true`。
- `df_fit` 会写入 JSON 元数据，并在加载时恢复，不再根据名义系数数量重新计算。
- 稳定的标量诊断信息会写入工件元数据，供后续审计和报告路径使用。
- Artifact validation report 现在使用 versioned `artifact_validation_report` envelope，并包含 typed `artifact_validation_issue` 条目、明确 severity 与 error/warning 计数。
- Artifact validation 现在会检查 schema-safe prediction 所需的 categorical factor-level metadata 和 numeric-transform AST metadata。
- `artifact_schema_policy()` 现在记录受支持的 schema version，以及 legacy 或 future artifact 的 validation/load 行为。
- 平滑项相关的拟合自由度通过共享算法 helper 计算，保证 RS 与 CG 的自由度统计一致。

## D1 剩余工作

- 为支持的平滑项持久化足够的 smooth-basis 元数据，实现 schema-safe 的平滑预测 roundtrip。
- 继续把结构化预测/工件错误覆盖扩展到未来的 service fit/sample 边界。
- 扩展 schema 校验，更完整覆盖剩余 interaction 和不支持 smoother 边界。
- 在未来 schema revision 引入时持续同步 migration policy。
