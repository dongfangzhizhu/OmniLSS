# 第二阶段进展：CG/L-BFGS 命名清理

English version: [phase2-cg-lbfgs-renaming-progress-2026-05-18.md](phase2-cg-lbfgs-renaming-progress-2026-05-18.md)

## 本步骤已完成

- 将历史上的 `cg_algorithm.py` 实现迁移到 `lbfgs_algorithm.py`，使优化器身份更加明确。
- 在 `cg_algorithm.py` 保留兼容 shim，以支持旧导入路径。
- 新增 `lbfgs_fit`，作为 `joint_lbfgs_fit` 的清晰别名。
- 保留基于公式的 `cg_fit` 与 `cg_fit_lbfgs` 兼容包装，并发出 `DeprecationWarning`，让现有调用方可以继续运行，同时引导新代码迁移到更清晰的命名。
- 更新 mixed-algorithm 内部导入，使其从 `lbfgs_algorithm.py` 使用 L-BFGS 后端，而不再依赖历史 CG 模块名。
- 为 mixed fitting 实现初始的 `algorithm="auto"` 选择辅助函数，覆盖平滑项、不稳定分布族以及小数据集优先 RS 的规则。
- 扩展 `compare_algorithms`，加入 deviance、迭代次数、收敛状态与运行时间指标，同时保留现有 `{"rs": model, "cg": model}` 访问方式。

## 第二阶段剩余工作

- 在迁移直接测试覆盖和辅助函数使用后，完全退役或合并 `cg_algorithm_v2.py`。
- 扩展 `fitting_cg.py` 的奇异 Hessian fallback 覆盖。
- 继续结合基准测试数据调优 `algorithm="auto"` 阈值。
