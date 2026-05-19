# CG 完整验证报告（Week 2 进展）

> 日期：2026-05-19  
> English version: [CG_FULL_VERIFICATION_2026_05_19.md](./CG_FULL_VERIFICATION_2026_05_19.md)

## 范围

本文档记录 Week 2 的 CG 完整验证当前进展：

- 联合打分矩阵组装（`build_joint_scoring_matrix`）
- 联合线性系统求解（`solve_joint_system`）
- 带线搜索的外循环单步（`cg_outer_step`）
- 迭代收敛脚手架（`run_cg_outer_loop`）

## 已完成验证

1. 与 NumPy 参考解的 block 求解一致性。
2. 在凸型 deviance toy 系统上的线搜索单调性。
3. 按相对 global deviance 收敛准则（`c_crit`）的迭代行为验证。
4. 针对验收分布标签 NO/GA/WEI/NBI 的 Week 2 验证脚手架测试。
5. 新增面向 `method="CG"` 的 R 对齐测试模块（`test_cg_algorithm_full_r_alignment.py`），覆盖 NO/GA/WEI；当 R bridge 不可用时自动跳过。

## 当前限制

- 与 R `gamlss` 的最终 deviance 直接对齐（`< 0.01`）目前为部分接入：已新增 NO/GA/WEI 测试模块，NBI 仍待补齐。
- 仓库中已有 R 一致性测试体系，但专门面向 Week 2 full-CG vs R bridge 的执行仍待集成。

## 执行命令

```bash
python -m pytest omnilss/tests/test_cg_algorithm_full.py omnilss/tests/test_cg_algorithm_full_validation.py omnilss/tests/test_cg_algorithm_full_r_alignment.py -q
```
