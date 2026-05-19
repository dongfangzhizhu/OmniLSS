# GPU 交叉点分析（Week 3 草案）

> 日期：2026-05-19  
> English version: [gpu_crossover_analysis.md](./gpu_crossover_analysis.md)

## 目标

基于冷/热启动分离后的诚实基准结果，沉淀 OmniLSS 的实际后端路由阈值建议。

## 当前证据基础

- 已实现冷/热分离辅助函数：`benchmark_jax(...)`。
- `honest_benchmark(...)` 已输出 `cold_s` 与热启动中位数 `median_s`。
- 单元测试已验证辅助函数行为与计时不变量。

## 初步建议（待完整 sweep 后修订）

- 对低维小/中规模问题优先使用 **NumPy RS**。
- 仅当重复调用可摊销编译开销且矩阵规模足够大时考虑 **JAX RS**。
- 当前实验阶段的暂定阈值规则：

```text
当 n > 50k 且 p > 10（或多参数 CG block 较大）时，优先考虑 JAX 后端
```

## 后续必须补充的测量

1. 在固定 `n` 下做 `p`-sweep（`p in {2, 5, 10, 20, 50}`）。
2. 在固定 `p` 下做 `n`-sweep（`n in {1k, 5k, 10k, 50k, 100k, 500k}`）。
3. 每次运行都记录硬件信息：
   - CPU 型号
   - GPU 型号
   - JAX 版本
4. 冷/热启动分别对比以下基线：
   - NumPy RS（CPU）
   - R GAMLSS

## 状态

Week 3 基准修复进行中；本文档作为持续更新的分析台账。
