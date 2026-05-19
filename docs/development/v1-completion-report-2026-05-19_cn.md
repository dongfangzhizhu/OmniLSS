# OmniLSS v1.0 开发完成报告 — 2026-05-19

English version: [v1-completion-report-2026-05-19.md](v1-completion-report-2026-05-19.md)

## 范围

本报告用于关闭 [v1-development-plan-2026-05-18_cn.md](v1-development-plan-2026-05-18_cn.md) 中列出的实现任务。项目默认语言仍为英文；本文档提供带 `_cn` 后缀的中文版本，并与英文版互相链接。

## 完成摘要

- **第一阶段 — JAX RS 性能架构：** cold-start JAX RS 已使用数据感知初始化，IRLS 公式已修正，同分布族批量拟合已支持 `jax.vmap`，`method="RS"` 已接入设备感知路由，显式 `RS_JAX` 保留为弃用兼容路径，TF score/Hessian kernel 已降低 retracing 开销。
- **第二阶段 — 算法命名与 Cole-Green 完善：** 历史 L-BFGS 路径已使用 L-BFGS 命名暴露，包级 `cg_fit` 已路由到真实 Cole-Green 后端，`algorithm="auto"` 已使用实际选择规则，奇异 Hessian 保护已有验证覆盖。
- **第三阶段 — 统一分布注册表与验证：** `distribution_registry.py` 已成为权威查找入口，内置分布通过单一字典表注册，`resolve_family()` 委托给注册表，旧的 legacy if-chain resolver 已移除，数值契约验证覆盖已就位。
- **第四阶段 — 生产级服务层：** proto3 数组载荷保持 JSON 兼容，模型 artifact 使用 SQLite 持久化存储，gRPC 与 REST 均暴露 list/delete 模型生命周期操作，重启后预测测试验证了持久模型复用。
- **第五阶段 — JOSS 发表准备：** benchmark 已拆分为 no-R、optional-R、optional-GPU 套件；cold 与 warm 计时声明已分离并包含置信区间；CI 已加入无 R/无 GPU benchmark smoke job；JOSS 论文已更新需求、方法、验证、基准、限制与参考文献内容。
- **第六阶段 — 商业 Pro 差异化：** Pro AutoML 通过 Core client 调用按 deviance、AIC、BIC、GAIC 排名候选分布族；Pro bootstrap 置信区间使用重复 Core 拟合；Pro 实现和测试继续保持 gRPC client 边界且不导入 `omnilss`。

## 验证重点

本次完成检查重点复核了计划中的近期执行顺序：

1. Cold-start JAX RS 仍不依赖 NumPy RS warm-start。
2. CG/L-BFGS 命名不再让包级 Cole-Green 调用走历史 L-BFGS 别名。
3. Batched JAX RS 已具备真实的同分布族 `jax.vmap` 路径，并保留确定性的 fallback 行为。
4. 分布族查找现在以注册表为优先，`distributions.py` 中不再保留 legacy resolver chain。
