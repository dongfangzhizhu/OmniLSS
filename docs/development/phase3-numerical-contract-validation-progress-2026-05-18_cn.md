# 第三阶段进展：数值契约验证基线

English version: [phase3-numerical-contract-validation-progress-2026-05-18.md](phase3-numerical-contract-validation-progress-2026-05-18.md)

## 本步骤已完成

- 新增独立的数值契约测试模块，用于核心分布行为验证。
- 增加 `NO` 的 dpqr 契约检查，验证分位数单调性与 `p(q(p))` 往返一致性。
- 增加 `NO` 的 score/Hessian 契约检查，包括期望 score 接近 0、Hessian 保持负号约束，以及经验 Fisher 恒等式 `E[s^2] ≈ -E[H]`。

## 说明

这为第三阶段任务 3.4 建立了可复现基线。后续可将相同契约矩阵扩展到更多 family（`GA`、`PO`、`WEI`、`TF`），并根据 family 特征设置专门容差与边界网格。
