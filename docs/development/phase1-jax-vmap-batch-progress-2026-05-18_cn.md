# 第一阶段进展：同 Family JAX `vmap` 批处理 Kernel

English version: [phase1-jax-vmap-batch-progress-2026-05-18.md](phase1-jax-vmap-batch-progress-2026-05-18.md)

## 本步骤已完成

- 在 `batch_jax_rs_fit` 中为同 family 且各参数设计矩阵形状一致的批次增加真正的 `jax.vmap` 执行路径。
- 对 mixed-family 批次或形状不一致的模型保留确定性的逐模型 fallback，从而保持现有公共 API 与结果顺序。
- 与之前一样返回 `JaxRSResult` 对象，同时加速路径会在一个 vmapped XLA 程序中计算参数、eta、系数、deviance、迭代次数与收敛标志。
- 增加回归测试，通过 monkeypatch 逐模型 fallback，验证同 family/同形状批次确实通过向量化路径完成。

## 说明

本步骤完成了第一阶段关于使用 `jax.vmap` 作为 GPU 加速路径进行批量多模型拟合的要求。后续加速器性能工作应集中在由基准测试支持的 crossover 阈值，以及降低大型生产批次的编译时间开销。
