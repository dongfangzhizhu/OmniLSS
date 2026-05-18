# 第 2 月 Family Validation Schema 种子（2026-05-18）

[English version](month2-family-validation-schema-2026-05-18.md)

> 父计划：[六个月执行计划](six-month-execution-plan-2026-05-17_cn.md)
>
> 每周 checklist：[六个月每周实施 checklist](six-month-weekly-implementation-checklist-2026-05-17_cn.md)
>
> 机器可读种子：[core-family-validation-plan-2026-05-18.json](core-family-validation-plan-2026-05-18.json)

## 范围

本文启动第 1 月 release-gate preflight 之后的下一个计划工作流：**D4 validation matrix**。

当前输出是 schema seed 和 prioritized-family JSON plan。它尚未将统计检查标记为 passed；它定义了第 5 周实现应填充的证据形态。

## 优先 Family

初始 validation plan 遵循六个月执行计划，优先覆盖：

`NO`、`GA`、`PO`、`BI`、`NBI`、`BE`、`WEI`、`TF`、`LOGNO` 和 `ZAGA`。

## 必需检查类型

JSON plan 中每个 family row 都包含以下占位项：

- density 或 PMF reference values；
- CDF monotonicity；
- quantile/CDF inverse consistency；
- 适用时的 random sample moments；
- score versus finite-difference checks；
- Hessian versus finite-difference 或 AD checks；
- edge cases 与 invalid parameter handling。

## 失败分类

未来 validation output 应将失败分类为以下之一：

- `implementation_bug`；
- `numerical_tolerance_issue`；
- `unsupported_domain`；
- `reference_mismatch`；
- `environment_unavailable`。

## 下一步实现

添加 generator/runner，先为 `NO` 填充具体检查结果，然后按优先级扩展到剩余 family。
