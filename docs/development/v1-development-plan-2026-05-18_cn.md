# OmniLSS v1.0 开发计划

English version: [v1-development-plan-2026-05-18.md](v1-development-plan-2026-05-18.md)

## 范围

本计划描述当前 v0.3 分支迈向 v1.0.0 的路线。项目默认使用英文；本计划中的每份开发文档都必须提供中文版本，并在文件名中使用 `_cn` 后缀。

## 不可变工程约束

- 不得在 JAX RS 集成层重新引入 NumPy RS warm-start。
- 保持稳定的 RS 节奏：JAX RS 与 NumPy RS 的默认 `max_inner=1`。
- JAX 数组必须保持 float64 统计精度。
- 保持公共 API 向后兼容；只能添加带默认值的新参数。
- 保持 GPL 隔离：`omnilss-pro` 只能通过 gRPC 客户端与 Core 通信，不得直接导入 `omnilss`。
- 不得通过削弱参考测试或数学容差来掩盖失败。

## 第一阶段 — JAX RS 性能架构

1. 从 `jax_rs_integration.py` 移除 NumPy RS warm-start，并让 cold-start JAX RS 保持稳定。
2. 通过 `FamilyJAXSpec.init_etas` 和可选的 `jax_rs_fit_core(..., init_etas=None)` 默认值加入数据感知 cold-start 初始化。
3. 修正 JAX IRLS 工作权重和工作响应公式，使其与 NumPy RS 数学契约一致。
4. 增加基于 `jax.vmap` 的批量多模型拟合，作为真正的 GPU 加速路径。
5. 让 `method="RS"` 通过设备感知规则自动选择后端，并保留 `method="RS_JAX"` 作为弃用兼容路径。
6. 通过 JIT 缓存 score/Hessian 函数以及必要时手写导数来降低 TF 的 AD retracing 开销。

## 第二阶段 — 算法命名与 Cole-Green 完善

1. 将历史 L-BFGS 模块从误导性的 CG 命名中迁移出来。
2. 让 `cg_fit` 指向真正的 Cole-Green 实现，同时保留弃用别名以兼容旧代码。
3. 在 `mixed_algorithm.py` 中实现真正的 `algorithm="auto"` 选择规则。
4. 为 `fitting_cg.py` 增加奇异 Hessian 保护和数值验证测试。

## 第三阶段 — 统一分布注册表与验证

1. 让 `distribution_registry.py` 成为权威注册中心。
2. 通过单一字典机制注册所有分布族。
3. 在保留公共函数名的同时，让 `resolve_family()` 委托给注册表。
4. 增加独立的数值契约验证，覆盖 dpqr 一致性、期望值、Fisher 恒等式、Hessian 符号以及分位数单调性。

## 第四阶段 — 生产级服务层

1. 扩展 proto3 schema，支持高效数组载荷，同时保留 JSON 兼容路径。
2. 使用 SQLite 持久化存储替代临时模型存储，并通过环境变量控制 artifact 目录。
3. 增加模型列表和删除 RPC，并加入服务重启后预测可用的测试。
4. 增加 FastAPI REST 端点，并共享同一个持久化模型注册表。

## 第五阶段 — JOSS 发表准备

1. 将基准测试拆分为无需 R、可选 R、可选 GPU 三类套件。
2. 分别报告冷启动和 warm runtime 性能，并包含置信区间。
3. 增加不依赖 R 或 GPU 的 CI benchmark smoke job。
4. 更新 JOSS 论文，补充需求陈述、算法描述、保守性能结论、限制以及完整参考文献。

## 第六阶段 — 商业 Pro 差异化

1. 通过 Core batch-fit RPC 实现 Pro AutoML 分布选择。
2. 按 deviance、AIC、BIC 和 GAIC 对候选分布族排序。
3. 通过 batch-fit 重采样增加 Pro bootstrap 置信区间。
4. 持续保证 Pro 测试和实现只能通过 gRPC 客户端边界访问 Core。

## 近期执行顺序

1. 完成第一阶段任务 1.1：在不使用 NumPy RS warm-start 的前提下，让 cold-start JAX RS 通过现有核心与一致性测试。
2. 随后执行任务 2.1 的模块重命名，消除 CG/L-BFGS 命名歧义。
3. 随后实现任务 1.2 的批量 JAX RS 拟合，因为后续 Pro AutoML 与 bootstrap 都依赖它。
4. 随后统一分布注册表，以降低后续新增分布族的维护成本。
