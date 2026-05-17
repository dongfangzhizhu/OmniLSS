# 模型 Artifact Capability Snapshot（2026-05-17）

[English version](model-artifact-capability-snapshots-2026-05-17.md)

本文推进 [六个月执行计划](six-month-execution-plan-2026-05-17_cn.md) 中的 artifact 工作，并完成 [family capability registry 说明](family-capability-registry-2026-05-17_cn.md) 的第二个后续动作：JSON 模型 artifact 现在会持久化拟合 family 的 capability metadata snapshot。

## 运行时行为

`save_model_json()` 现在会在 `meta.json` 中写入 `family_capability` 对象。该对象由运行时 capability registry 生成，包含：

- family 名称；
- `rs_fit`、`rs_jax_fit`、`prediction`、`r_consistency`、`production_safe` 等 feature 状态；
- 人类可读的 registry notes。

`load_model_json()` 会将同一个对象恢复到 `model.additional_slots["family_capability"]`，使后续 prediction、reporting、service API 和 audit tool 能检查 artifact 写入时对应的证据等级。

## 为什么重要

- 模型 artifact 更可审计：使用者可以看到该 family 在关键 feature 上是 validated、experimental 还是 unsupported。
- 服务运行时可以在 model report 中直接暴露 capability snapshot，而不需要重新计算。
- 未来 artifact migration 可以检测模型训练后 runtime capability policy 是否发生变化。

## 后续工作

1. 增加 artifact compatibility report，对比已保存 capability snapshot 与当前 runtime registry。
2. 增加服务端 endpoint，用于暴露模型 artifact capability snapshot。
3. 将 capability snapshot 纳入 calibration 和 governance report。
4. 只有通过文档化 validation report 的 feature 才能从 `experimental` 提升为 `validated`。
