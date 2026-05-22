# 中文版本：Changelog

[English version](CHANGELOG.md)

> 当前中文页提供 v0.3.0 关键发布说明的可审计摘要；如与英文条目有差异，以英文原文为准并在后续同步修订。

## [0.3.0] - 2026-05-17

### 新增（Added）
- 新增联合 Fisher 组装工具与 family 级交叉信息块构建能力，用于多参数 GAMLSS 的一致性验证。
- 新增 CG 预热与路由/性能/夹具回放相关的稳定化测试与基线夹具（R golden + performance snapshot）。

### 变更（Changed）
- `gamlss(..., method="CG")` 接入可配置 RS warm-start，并暴露 warm-start 诊断槽位。
- `fit_penalized_wls` 在线性无平滑项场景增加 NumPy 快路径，减少不必要的 JAX 转换开销。

### 修复（Fixed）
- 改进 CG 信息块提取后的 roundtrip 校验，避免 Fisher 形状不一致时静默通过。

### 验证（Validation）
- 覆盖联合 Fisher、cross-information、CG warm-start、HTTP `/fit` 并发、R golden 回放和性能快照冻结测试。

### 已知限制（Known limitations）
- 受环境影响，JAX x64 相关 warning 在部分运行环境中仍会出现；发布门禁以测试通过与约束校验为准。

## 翻译状态

- [ ] 完成全文中文翻译（当前仅维护 v0.3.0 稳定化所需关键条目）
- [x] v0.3.0 关键发布条目与英文版本对齐
