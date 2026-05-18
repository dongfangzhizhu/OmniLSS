# 第 1 月能力矩阵进展（2026-05-18）

> English version: [month1-capability-matrix-progress-2026-05-18.md](month1-capability-matrix-progress-2026-05-18.md)
>
> 父计划：[six-month-execution-plan-2026-05-17_cn.md](six-month-execution-plan-2026-05-17_cn.md)

## 范围

本文记录第 1 月 / 工作流 D3：**RS/CG/JAX 能力矩阵** 的进展。

## 已实现进展

- 运行时能力注册表现在提供机器可读的 `capability_matrix()` 快照，包含 feature 列表、method-routing map、strict-mode policy flag 和每个分布族的证据状态。
- 新增 `tools/generate_capability_matrix.py`，并生成 `family-capability-matrix-2026-05-18.json`，供文档、发布包和 API 客户端复用。
- gRPC 服务新增 `CapabilityService.CapabilityMatrix`，使服务客户端可以通过 API 获取同一份运行时能力矩阵。
- production-safe 分布族可以将核心路径标记为 validated；当前基线将 `NO` 的 `rs_fit`、`prediction`、`r_consistency` 和 `production_safe` 标记为 validated。
- `gamlss(..., strict_capabilities=True)` 现在会拒绝实验性 method/family 路径，只允许 validated capability feature。拟合代码使用与矩阵输出相同的 `require_method_route()` 映射。
- 默认开发行为保持不变：除非显式请求 strict capability mode，否则实验性路径仍允许运行。
- 方法路由测试覆盖了 strict 模式下 validated 的 `NO` RS 路径可运行，以及实验性的 GA RS 路径被拒绝。
- 生成的 JSON artifact、HTTP metadata response 和 gRPC capability response 现在携带与 `gamlss()` 相同的 method-to-feature routing contract。

## D3 剩余工作

- 扩展验证证据，让更多核心分布族可以针对特定路径从 experimental 升级为 validated。
- 增加专用 service-side route-admission helper，在未来 async fit job 被调度前复用矩阵进行检查。
