# 第一阶段进展：TF JAX Score/Hessian 性能

English version: [phase1-tf-jax-performance-progress-2026-05-18.md](phase1-tf-jax-performance-progress-2026-05-18.md)

## 本步骤已完成

- 为通用 AD score/Hessian vmap helper 增加 JIT 包装，使基于 AD 构建的 family 能缓存向量化导数 kernel。
- 将 TF `FamilyJAXSpec` 的 `mu`、`sigma` 与 `nu` score 函数替换为手写 Student-t 导数。
- 将 TF 对角 Hessian 替换为负的 Fisher-information 风格近似，以稳定 IRLS 更新。
- 增加回归测试，将手写 TF score 与 JAX autodiff 对照，并检查 TF Hessian 保持有限且为负。

## 说明

TF JAX 路径不再在 TF spec 内构造 vmapped gradient/Hessian 闭包。这降低了重复 RS 调用中的 tracing 压力，同时保持 score 公式与 autodiff 参考结果数值一致。
