# Cole-Green (1992) CG 推导笔记（Week 1 草案）

> 日期：2026-05-19  
> English version: [cg_derivation.md](./cg_derivation.md)

## 目标

本文档作为 Week 1 的数学推导工作区，目标是支撑 CG 完整实现，重点覆盖联合打分矩阵所需的跨参数二阶导数。

## 符号约定

对参数 \(\theta = (\theta_1, \dots, \theta_K)\)，线性预测子 \(\eta_k = X_k \beta_k\)：

- 一阶 score 分量：\(u_k = \partial \ell / \partial \eta_k\)
- 同参数二阶导：\(h_{kk} = \partial^2 \ell / \partial \eta_k^2\)
- 跨参数二阶导：\(h_{jk} = \partial^2 \ell / (\partial \eta_j \partial \eta_k)\), \(j \neq k\)

CG 外循环使用联合矩阵进行 Newton 更新：

\[
H =
\begin{bmatrix}
H_{11} & H_{12} & \cdots & H_{1K} \\
H_{21} & H_{22} & \cdots & H_{2K} \\
\vdots & \vdots & \ddots & \vdots \\
H_{K1} & H_{K2} & \cdots & H_{KK}
\end{bmatrix},
\quad
s =
\begin{bmatrix}
s_1 \\
s_2 \\
\vdots \\
s_K
\end{bmatrix}
\]

其中每个块 \(H_{jk} = X_j^\top W_{jk} X_k\)，\(W_{jk}\) 来自 \(-h_{jk}\) 的构造。

## 与实现的映射（计划）

- `d_ll / d_eta_k`：复用现有 RS score。
- `d²_ll / d_eta_k²`：复用现有 RS Hessian 对角支持。
- `d²_ll / (d_eta_j d_eta_k)`：通过 JAX 自动微分实现（`jax.hessian` 或 `jax.jacfwd(jax.jacrev(...))`）。
- Fisher 期望信息：作为收敛诊断可选项。

## 验证计划

1. 使用 `check_grads` 做 AD 与有限差分对照（NO/GA/WEI）。
2. 与 R 内部导数参考做 mixed-partial 对照（例如 NO 的 \(\mu,\sigma\)）。
3. 验收目标：相对 R 的 mixed partial 误差 < 1e-6。

## 进度清单

- [x] 完成初始符号与 block-matrix 框架。
- [ ] 补全含 Schur 补策略的更新公式完整推导。
- [ ] 补全 NO/GA/WEI/NBI 分布族导数展开。
- [ ] 整理为可直接进入报告的证明叙述。
