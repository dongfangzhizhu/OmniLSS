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
- gRPC capability service 现在也提供 `RouteCapability`，使服务客户端可以用与 HTTP metadata 和 `gamlss()` strict routing 相同的 report 预检 method/family admission。
- production-safe 分布族可以将核心路径标记为 validated；当前基线将 `NO` 的 `rs_fit`、`prediction`、`r_consistency` 和 `production_safe` 标记为 validated。
- `gamlss(..., strict_capabilities=True)` 现在会拒绝实验性 method/family 路径，只允许 validated capability feature。拟合代码使用与矩阵输出相同的 `require_method_route()` 映射。
- 默认开发行为保持不变：除非显式请求 strict capability mode，否则实验性路径仍允许运行。
- 方法路由测试覆盖了 strict 模式下 validated 的 `NO` RS 路径可运行，以及实验性的 GA RS 路径被拒绝。
- 生成的 JSON artifact、HTTP metadata response、gRPC capability matrix response 和 gRPC route-capability report 现在携带与 `gamlss()` 相同的 method-to-feature routing contract。
- 公开的 `method_route_feature()` 和 `require_method_route()` helper 已恢复，`capability_matrix()` 也重新包含向后兼容的 `method_routes` alias，使生成 artifact、测试、service metadata 和文档化 strict routing 共享同一个可导入 contract。
- `method_route_capability_report()` 现在为 service boundary 提供 JSON 友好的 route-admission report，可在未来 async fit job 调度前使用；`gamlss()` 也使用同一个 helper 进行 runtime gate，并且 HTTP metadata boundary 已暴露该 report 供 client preflight check 使用。

## D3 剩余工作

- 扩展验证证据，让更多核心分布族可以针对特定路径从 experimental 升级为 validated。
- 当 service job runtime 引入后，将 `method_route_capability_report()` 接入未来 async job admission。
