# 第二阶段进展：`cg_fit` 使用真正的 Cole-Green 后端

English version: [phase2-cg-fit-real-cg-progress-2026-05-18.md](phase2-cg-fit-real-cg-progress-2026-05-18.md)

## 本步骤已完成

- 更新包级 `omnilss.algorithms.cg_fit` wrapper，使公式接口调用路由到 `gamlss(method="CG")`，也就是 full-Hessian Cole-Green 后端。
- 通过 `cg_fit_lbfgs`、`joint_lbfgs_fit` 与 `lbfgs_fit` 保留历史 L-BFGS 兼容入口。
- `cg_fit` 继续接受旧的 step-size 关键字参数以保持调用签名兼容；full-Hessian CG 后端会忽略这些值，并使用自己的阻尼 line search。
- 增加回归测试，确认包级 `cg_fit` 不再发出旧 L-BFGS deprecation warning。

## 说明

较低层的 `omnilss.algorithms.cg_algorithm.cg_fit` 模块 shim 仍是历史 CG 命名 L-BFGS 模块旧导入的弃用兼容别名。新的包级代码应使用 `omnilss.algorithms.cg_fit` 表示 Cole-Green，使用 `joint_lbfgs_fit`/`lbfgs_fit` 表示 L-BFGS。
