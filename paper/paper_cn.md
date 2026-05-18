[English version](paper.md)
---
title: 'OmniLSS：使用 JAX 的高性能 Python GAMLSS'
tags:
  - Python
  - JAX
  - statistical modeling
  - GAMLSS
  - distributional regression
authors:
  - name: OmniLSS contributors
    orcid: TBD
    affiliation: 1
affiliations:
  - name: Independent
    index: 1
date: 2026-05-18
bibliography: paper.bib
---

# 摘要

OmniLSS 是一个基于 NumPy 与 JAX [@jax2018] 的 Python 广义加性位置、尺度与形状模型（GAMLSS）[@rigby2005] 实现。它提供丰富的分布族目录、按参数指定的公式、稳定的 Rigby-Stasinopoulos（RS）拟合路径、面向加速器工作负载的 JAX 原生 cold-start RS 路径，以及可选的 HTTP 和 gRPC 服务接口。

# 需求说明

R `gamlss` 生态仍然是分布回归的参考实现，但许多科研、机器学习和生产服务工作流都以 Python 为主。这些环境中的实践者通常同时需要三类能力：熟悉的 GAMLSS 分布族、Python 原生的模型产物与预测 schema，以及能够参与现代自动微分和加速器工作流的数值内核。OmniLSS 通过保留核心 GAMLSS 概念，并引入明确的验证门禁与适合部署的服务边界，填补了这一缺口。

# 算法方法

OmniLSS 将 RS 算法作为保守的默认拟合路径。NumPy 实现优先保证参考稳定性和与 R 的一致性检查。JAX RS 实现保持稳定的 `max_inner=1` 节奏，使用数据感知 cold-start 初始化而不是 NumPy RS warm-start，保留 float64 计算，并为独立的同分布族模型提供批量拟合入口。这些批量入口面向 bootstrap、交叉验证和候选分布族评估等场景，在这些场景中，加速器吞吐量比单个小模型拟合更有意义。

该包还包含 Cole-Green full-Hessian 拟合路径，并为历史 L-BFGS 入口保留弃用兼容别名。分布族元数据通过注册表集中管理，使公式拟合、验证、服务 API 和下游自动化使用同一套 family resolution 契约。

# 验证与基准测试

基准测试结论由 `benchmarks/` 下的脚本生成。v1.0 基准测试工作流区分仅 Python 的 smoke 检查、可选 R 比较和可选 GPU 运行。报告必须区分 JAX cold 编译时间与重复 warm 计时，为重复 warm 计时提供置信区间，并说明 R 或 GPU 资源是否可用。CI smoke 作业不需要 R 或 GPU，只作为健康检查；可用于发表的比较必须使用 optional-R 套件，并引用生成的产物、硬件、JAX 后端、dtype、数据规模、公式和重复次数。

# 局限性

OmniLSS 并不是 R 生态每个功能的完全替代品。不同分布族和平滑器的验证覆盖程度不同，GPU 收益取决于工作负载和形状，JAX 编译开销可能主导小型一次性拟合。因此，性能结论应保持保守，并区分 cold 与 warm 行为。项目也有意将商业附加功能保留在 GPL core 进程边界之外；Pro 侧自动化通过 gRPC 与 Core 通信，而不是直接导入 `omnilss`。

# 致谢

OmniLSS 受到 R GAMLSS 项目，以及 Rigby、Stasinopoulos、Cole、Green 和更广泛分布回归社区统计工作的启发。

# 参考文献
