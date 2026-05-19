# 第一阶段进展：批量 JAX RS API

English version: [phase1-batched-jax-rs-progress-2026-05-18.md](phase1-batched-jax-rs-progress-2026-05-18.md)

## 本步骤已完成

- 新增 `omnilss.algorithms.jax_rs_batch.batch_jax_rs_fit`，作为独立 JAX RS 拟合的批量 API。
- 支持广播设计矩阵，也支持每个模型提供一个设计矩阵元组。
- 支持所有模型共享一个 family spec，也支持每个模型提供一个 `FamilyJAXSpec`，并按 family 名称对混合 family 输入进行确定性分组。
- 支持观测权重为 `None`、`[n]` 或 `[K, n]`。
- 在公式集成层新增 `gamlss_rs_jax_batch`，调用方可以批量拟合多个数据集或 family，而无需手动循环调用 `gamlss_rs_jax`。
- 从 `omnilss.algorithms` 导出新的批量 API。
- 增加测试，确认批量结果与重复调用 `jax_rs_fit_core` 一致，并确认公式层批量 helper 返回拟合模型。

## 说明

该 API 现在具备两种执行模式：同 family 且形状一致的批量输入使用 [phase1-jax-vmap-batch-progress-2026-05-18_cn.md](phase1-jax-vmap-batch-progress-2026-05-18_cn.md) 中描述的向量化 `jax.vmap` kernel；混合 family 或形状不一致的输入继续使用确定性的按 family fallback。
