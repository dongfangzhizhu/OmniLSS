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
- Capability matrix schema version 现在通过 `CAPABILITY_MATRIX_VERSION = 3` 暴露，因此新增 `method_routes` 兼容 key 会对 client 可见，而不是静默改变 version-2 payload。
- 公开的 `method_route_feature()` 和 `require_method_route()` helper 已恢复，`capability_matrix()` 也重新包含向后兼容的 `method_routes` alias，使生成 artifact、测试、service metadata 和文档化 strict routing 共享同一个可导入 contract。
- 新增 `tools/validate_capability_matrix.py` 和 `validate_capability_matrix_payload()`，使生成的 matrix artifact 在发布包或 service metadata 复用前可检查 schema version、method-route alias drift、policy drift、family 覆盖率和 feature-status 有效性。
- Matrix validator 现在会针对不可读文件和 malformed JSON 返回结构化 report，并继续报告 schema drift，因此 release check 会收到统一的机器可读失败 envelope，而不是未捕获的 file/parse exception。
- `method_route_capability_report()` 现在为 service boundary 提供 JSON 友好的 route-admission report，可在未来 async fit job 调度前使用；`gamlss()` 也使用同一个 helper 进行 runtime gate，并且 HTTP metadata boundary 已暴露该 report 供 client preflight check 使用。

## D3 收尾与延期工作

当前第 1 月范围内的第 3 周 capability gate 实现已完成：runtime gate、生成 artifact、HTTP/gRPC metadata、schema versioning 和 validator check 都共享同一个 method-routing contract。

延期工作已明确分配到后续 roadmap：

- 扩展验证证据，让更多核心 family 可以从 experimental 升级到 validated，属于第 2 月 validation 工作。
- 将 `method_route_capability_report()` 接入 async job admission，属于第 3 月 service job-runtime 工作。
