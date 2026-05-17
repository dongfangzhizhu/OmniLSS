# OmniLSS Phase 1 执行计划（2026H2）

[中文版本](phase1-execution-plan-2026H2_cn.md)

## 目标

Phase 1 从 Phase 0 的“止血与冻结”进入“技术可信度建设”：

- 从“Python GAMLSS”叙事升级为 **Differentiable Distributional Runtime**
- 建立稳定的数值系统、可验证的 autodiff 管线、可复现实验资产
- 形成 GPU/并行与诊断能力的工程闭环

> 周期建议：3–6 个月（上限 9 个月）

---

## 战略叙事（必须）

- ❌ 不再以 “Python version of GAMLSS” 作为主叙事
- ✅ 统一对外叙事：
  - Differentiable Probabilistic Runtime
  - GPU Distributional Regression
  - Uncertainty Infrastructure

---

## 模块化任务

## 模块 1：Numerical Stability System（P0）

- [x] 完整 safe math runtime：`safe_exp/log/log1p/softplus/sigmoid/divide/sqrt`
- [x] 统一稳定参数化：positive/bounded/ordered transforms
- [x] Hessian stabilization：adaptive damping + Fisher fallback + singular detection
- [x] Trust region / line search：step halving + trust region + adaptive scale

## 模块 2：Differentiable Runtime Architecture（P0）

- [x] JAX-native graph 能力矩阵（jit/grad/vmap/pmap）
- [x] 逐步纯函数化（减少 mutable state）
- [x] Distribution registry 系统（消除 if/else family 分支）
- [x] autodiff validation pipeline（finite difference vs autodiff）

## 模块 3：Core Statistical Runtime（P1）

- [x] MVP 扩展 families（GG/SHASH/NB/BCCG），总数≤10
- [x] `pb()` spline runtime 稳定化（penalty/knot/conditioning）
- [x] 统一预测接口：mean/quantile/interval/distribution
- [x] calibration runtime：PIT/CRPS/coverage/calibration curve

## 模块 4：GPU & Parallel Runtime（P1）

- [x] vmap-native fitting
- [x] jit compilation cache
- [x] GPU benchmark suite（small/medium/large/pathological）
- [x] 指标统一：runtime/memory/compile time/convergence

## 模块 5：Diagnostics Runtime（P0）

- [x] convergence diagnostics：gradient_norm/condition_number/step_size/hessian_stability
- [x] automatic warning system（梯度爆炸/奇异 Hessian/坏条件数/不稳定平滑）
- [x] numerical logging（可复现实验资产）

---

## 学术交付

- [x] 第一篇正式论文初稿（Differentiable Distributional Regression with JAX）
- [x] benchmark paper assets（correctness/stability/convergence/scaling/GPU）
- [x] family 数学文档体系（likelihood/score/hessian/constraints/parameterization）
- [x] technical blog series（3 篇）

---

## 商业边界交付（不做商业功能扩张）

- [x] OmniLSS-Pro skeleton（core -> grpc -> pro）
- [x] serving runtime prototype（quantile/interval/uncertainty API）
- [x] monitoring prototype（calibration drift / uncertainty shift / CRPS degradation）
- [x] license firewall（clear IPC boundary）
- [x] branding 切换完成

---

## Done Definition

### 开发
- [x] numerical stability layer 完整
- [x] autodiff validation pipeline
- [x] GPU benchmark suite
- [x] calibration runtime
- [x] diagnostics runtime
- [x] stable JAX runtime

### 学术
- [x] 论文初稿
- [x] benchmark assets
- [x] 数学文档体系
- [x] technical blogs

### 商业
- [x] grpc runtime boundary
- [x] serving prototype
- [x] monitoring prototype
- [x] branding 完成
