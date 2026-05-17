# 第 1 月能力矩阵进展（2026-05-18）

> English version: [month1-capability-matrix-progress-2026-05-18.md](month1-capability-matrix-progress-2026-05-18.md)
>
> 父计划：[six-month-execution-plan-2026-05-17_cn.md](six-month-execution-plan-2026-05-17_cn.md)

## 范围

本文记录第 1 月 / 工作流 D3：**RS/CG/JAX 能力矩阵** 的进展。

## 已实现进展

- 运行时能力注册表现在提供机器可读的 `capability_matrix()` 快照，包含 feature 列表和每个分布族的证据状态。
- production-safe 分布族可以将核心路径标记为 validated；当前基线将 `NO` 的 `rs_fit`、`prediction`、`r_consistency` 和 `production_safe` 标记为 validated。
- `gamlss(..., strict_capabilities=True)` 现在会拒绝实验性 method/family 路径，只允许 validated capability feature。
- 默认开发行为保持不变：除非显式请求 strict capability mode，否则实验性路径仍允许运行。
- 方法路由测试覆盖了 strict 模式下 validated 的 `NO` RS 路径可运行，以及实验性的 GA RS 路径被拒绝。

## D3 剩余工作

- 通过服务端点暴露能力矩阵，供 API 客户端使用。
- 在发布包和文档构建中增加生成的 JSON 工件。
- 扩展验证证据，让更多核心分布族可以针对特定路径从 experimental 升级为 validated。
