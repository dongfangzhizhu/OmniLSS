# 第 1 月预测 Schema 与 Parser Checklist 进展（2026-05-17）

[English version](month1-prediction-schema-parser-checklist-2026-05-17.md)

## 计划 Checklist 对照

本文推进[六个月执行计划](six-month-execution-plan-2026-05-17_cn.md)中第 1 月 D1 与 D2 工作流：artifact schema v2、预测 schema 硬错误、公式 parser 加固，以及恶意/边界 parser 测试。

## Checklist 状态

| 计划项 | 状态 | 证据 |
|---|---|---|
| 缺少预测变量时返回结构化 schema 错误 | 进行中 / 已加强 | `PredictionSchemaError` 现在携带 `code`、`parameter`、`term` 和 `reason` 字段，便于 schema 消费方处理。 |
| 未见过的 factor level 显式失败 | 已新增 | 回归测试验证 `unseen_factor_levels` metadata。 |
| 缺少 smooth metadata 显式失败 | 已新增 | 回归测试验证 `missing_smooth_metadata` metadata。 |
| 列数不匹配显式失败 | 已新增 | 回归测试验证 `schema_column_mismatch` metadata。 |
| smooth/tensor 参数解析避免朴素逗号切分 | 已新增 | 公式解析现在只按顶层分隔符切分，并支持嵌套括号和带引号字符串。 |
| Parser 边界场景有测试覆盖 | 已新增 | 测试覆盖嵌套 `k_list=[5, 8]`、带逗号的引号参数，以及括号不平衡拒绝。 |
| 生产预测路径不执行任意 Python 代码 | 保持 | 数值表达式评估仍统一走 AST allowlist evaluator。 |

## 第 1 月剩余跟进

1. 将结构化 schema 错误扩展到所有绕过 `omnilss.prediction` 的 legacy prediction 入口。
2. 增加公开 artifact-schema 示例，展示结构化错误形状和推荐客户端处理方式。
3. 随着非核心 helper 模块进入生产预测路径，继续替换其中的 legacy 字符串切分逻辑。
