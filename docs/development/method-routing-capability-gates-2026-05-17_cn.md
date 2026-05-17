# Method Routing Capability Gates（2026-05-17）

[English version](method-routing-capability-gates-2026-05-17.md)

本文推进 [六个月执行计划](six-month-execution-plan-2026-05-17_cn.md) 的 D3 工作流，并完成 [family capability registry 说明](family-capability-registry-2026-05-17_cn.md) 中的第一个后续动作：`gamlss()` 现在会在进入具体拟合 backend 前检查 family capability registry。

## 运行时行为

`gamlss()` 会在开始昂贵拟合工作前校验解析后的 method 和 family：

| Method | 检查的 capability feature | 是否允许 experimental | 不支持时行为 |
|---|---|---:|---|
| `RS` | `rs_fit` | 是 | 抛出 `FamilyCapabilityError` |
| `RS_JAX` | `rs_jax_fit` | 是 | 在调用 JAX backend 前抛出 `FamilyCapabilityError` |
| `CG` | `cg_fit` | 是 | 抛出 `FamilyCapabilityError` |
| `MIXED` | `cg_fit` | 是 | 抛出 `FamilyCapabilityError` |
| `joint` / `JOINT` | `cg_fit` | 是 | 在初始化/精修前抛出 `FamilyCapabilityError` |
| `lbfgs` / `LBFGS` | `cg_fit` | 是 | 在初始化/精修前抛出 `FamilyCapabilityError` |
| `auto` | 先解析为 `RS` 或 `RS_JAX`，再检查解析后的 route | 是 | 不支持的 `RS_JAX` route 无法继续 |

当前策略仍允许 experimental feature，因为项目现有默认定位偏研究/开发。关键变化是：`unsupported` route 现在会在 backend 执行前通过 capability registry 快速失败。

## 为什么重要

- 不支持的 method/family 组合会快速失败。
- routing 层现在与文档、测试使用同一个 capability source of truth。
- `RS_JAX` 不再只依赖 backend 内部的支持检查；不支持的 family 会在公开 `gamlss()` 边界被拒绝。
- 未来服务 API 可以在调度异步 job 前复用同一个 gate。

## 后续工作

1. 增加可选 strict production mode：除非显式启用，否则拒绝 `experimental` feature。
2. 将 capability snapshot 写入序列化模型 metadata。
3. 通过 HTTP/gRPC service endpoint 暴露 capability matrix。
4. 从运行时 registry 生成机器可读 capability matrix artifact。
