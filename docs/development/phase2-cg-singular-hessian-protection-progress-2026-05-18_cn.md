# 第二阶段进展：CG 奇异 Hessian 防护验证

English version: [phase2-cg-singular-hessian-protection-progress-2026-05-18.md](phase2-cg-singular-hessian-protection-progress-2026-05-18.md)

## 本步骤已完成

- 为 `fit_cg` 增加回归测试：在 `mu` 设计矩阵中使用近奇异（完全共线）场景。
- 验证在显式 regularization 打开时，`fit_cg` 会返回有限 deviance 与有限 Fisher 信息，而不是在线性求解阶段崩溃。
- 修复 `fitting_cg.py` 非收敛分支中重复的 `warnings.warn(...)` 调用，确保 warning 行为干净且确定。

## 说明

本步骤增强了第二阶段任务 2.4 的覆盖，验证了一个数值契约：当提供 regularization 时，CG 在病态设计矩阵下仍可稳定返回结果。
