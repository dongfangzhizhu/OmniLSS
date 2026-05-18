# 第 1 月 Release Gate 预检（2026-05-18）

[English version](month1-release-gate-preflight-2026-05-18.md)

> 父 checklist：[六个月每周实施 checklist](six-month-weekly-implementation-checklist-2026-05-17_cn.md)
>
> 相关 capability 收尾：[第 1 月 Capability Matrix 进展](month1-capability-matrix-progress-2026-05-18_cn.md)

## 范围

本文启动第 4 周：**第 1 月 release gate / Core trust checkpoint with reproducible test evidence**。

该预检 gate 特意保持 offline-friendly。它会在尝试 optional packaging tools、networked checks、R consistency job 或 GPU validation 之前，先验证必须通过的发布关键 metadata。

## 已实现的预检 Gate

`omnilss/tools/release_check.py` 现在支持：

```bash
PYTHONPATH=omnilss/src python omnilss/tools/release_check.py --preflight-only
```

当前 preflight path 会检查：

1. 双语文档 localization 与 cross-link policy；
2. 通过 `tools/validate_capability_matrix.py` 检查生成的 capability matrix schema/version/route-alias/family coverage；
3. 通过 `tools/release_gate_smoke.py` 运行确定性的核心 smoke checks，包括 linear fit/predict/serialize/load/predict，以及 schema-safe missing-variable / unseen-factor errors。

如果未提供 `--preflight-only`，完整 release check 仍会在 preflight 之后继续运行 packaging checks。

## 当前证据

| Check | Command | 当前结果 | 失败时是否阻塞发布 |
|---|---|---|---|
| Capability matrix validator tests | `PYTHONPATH=omnilss/src pytest -q omnilss/tests/test_generate_capability_matrix_tool.py` | Pass | 是 |
| Release preflight tests | `PYTHONPATH=omnilss/src pytest -q omnilss/tests/test_release_check.py` | Pass | 是 |
| Release-gate smoke tests | `PYTHONPATH=omnilss/src pytest -q omnilss/tests/test_release_gate_smoke.py` | Pass | 是 |
| Release-gate smoke CLI | `PYTHONPATH=omnilss/src python omnilss/tools/release_gate_smoke.py` | Pass；linear prediction roundtrip 最大误差 `0.0` | 是 |
| Offline release preflight | `PYTHONPATH=omnilss/src python omnilss/tools/release_check.py --preflight-only` | Pass | 是 |

## 第 4 周收尾与剩余环境相关工作

当前环境中的 offline 第 4 周 release gate 已完成：documentation localization、capability matrix validation、artifact validation、linear JSON roundtrip prediction，以及结构化 prediction-error smoke checks 均已覆盖。

标记发布前剩余的环境相关工作：

- 在 `build` 和 `twine` 可用的环境中运行更完整的 package/build checks。
- 决定第 1 月是否可以在将 JAX float64 environment warning 记录为 non-blocking 的情况下关闭，或是否需要在 release 前调整 warning policy。
