# GAMLSS 分布族 placeholder 迁移队列清单

> 生成日期：2026-08-23
> 生成方式：`tools/generate_placeholder_migration_queue.py`
> 状态来源：`omnilss.family_capabilities` 的代码派生 `implementation_status`（placeholder = 生产路由被拒，需重写）
> 批次划分：`docs/development/full-gamlss-migration-audit-and-roadmap-2026-08-22.md` 附录 A

勾选规则：`- [ ]` 表示待迁移；`- [x]` 表示已完成（状态已离开 placeholder）。

## A1 偏态 t 与 SEP

| 勾选 | R family | registry | 当前状态 | 备注 |
|---|---|---|---|---|
| [x] | `ST1` | `ST1` | compat_alias | 状态已离开 placeholder（compat_alias） |
| [x] | `ST2` | `ST2` | compat_alias | 状态已离开 placeholder（compat_alias） |
| [x] | `ST3` | `ST3` | compat_alias | 状态已离开 placeholder（compat_alias） |
| [ ] | `ST3C` | `ST3C` | 占位 |  |
| [x] | `ST4` | `ST4` | compat_alias | 状态已离开 placeholder（compat_alias） |
| [ ] | `ST5` | `ST5` | 占位 |  |
| [ ] | `SST` | `SST` | 占位 |  |
| [ ] | `SEP` | `SEP` | 占位 |  |
| [ ] | `SEP1` | `SEP1` | 占位 |  |
| [ ] | `SEP2` | `SEP2` | 占位 |  |
| [ ] | `SEP3` | `SEP3` | 占位 |  |
| [ ] | `SEP4` | `SEP4` | 占位 |  |

## A2 Box-Cox 与有界混合

| 勾选 | R family | registry | 当前状态 | 备注 |
|---|---|---|---|---|
| [x] | `BCCGo` | `BCCGO` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `BCPEo` | `BCPEO` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `BCTo` | `BCTO` | experimental | 状态已离开 placeholder（experimental） |
| [ ] | `BEINF0` | `BEINF0` | 占位 |  |
| [ ] | `BEINF1` | `BEINF1` | 占位 |  |
| [ ] | `BEOI` | `BEOI` | 占位 |  |
| [ ] | `BEZI` | `BEZI` | 占位 |  |
| [ ] | `LG` | `LG` | 占位 |  |
| [ ] | `LNO` | `LNO` | 占位 |  |
| [ ] | `LOGITNO` | `LOGITNO` | 占位 |  |
| [ ] | `NET` | `NET` | 占位 |  |

## A3 离散与零膨胀计数

| 勾选 | R family | registry | 当前状态 | 备注 |
|---|---|---|---|---|
| [ ] | `DEL` | `DEL` | 占位 |  |
| [ ] | `DPO` | `DPO` | 占位 |  |
| [ ] | `SI` | `SI` | 占位 |  |
| [ ] | `WARING` | `WARING` | 占位 |  |
| [ ] | `YULE` | `YULE` | 占位 |  |
| [ ] | `NBF` | `NBF` | 占位 |  |
| [ ] | `GPO` | `GPO` | 占位 |  |
| [ ] | `NBII` | `NBII` | 占位 |  |
| [ ] | `ZALG` | `ZALG` | 占位 |  |
| [ ] | `ZIP` | `ZIP` | 占位 |  |
| [ ] | `ZIP2` | `ZIP2` | 占位 |  |
| [ ] | `ZIPF` | `ZIPF` | 占位 |  |

## A4 正实数与其余连续

| 勾选 | R family | registry | 当前状态 | 备注 |
|---|---|---|---|---|
| [ ] | `GB1` | `GB1` | 占位 |  |
| [ ] | `GB2` | `GB2` | 占位 |  |
| [ ] | `GIG` | `GIG` | 占位 |  |
| [ ] | `NO2` | `NO2` | 占位 |  |
| [ ] | `NOF` | `NOF` | 占位 |  |
| [ ] | `PARETO` | `PARETO` | 占位 |  |
| [ ] | `PARETO1` | `PARETO1` | 占位 |  |
| [ ] | `PE2` | `PE2` | 占位 |  |
| [ ] | `PIG2` | `PIG2` | 占位 |  |
| [ ] | `RGE` | `RGE` | 占位 |  |
| [ ] | `WEI2` | `WEI2` | 占位 |  |
| [ ] | `exGAUS` | `EXGAUS` | 占位 |  |
| [ ] | `GAF` | `GAF` | 占位 |  |
| [ ] | `EGB2` | `EGB2` | 占位 |  |
| [ ] | `LOGSHASH` | `LOGSHASH` | 占位 |  |
| [ ] | `LQNO` | `LQNO` | 占位 |  |
| [ ] | `SN1` | `SN1` | 占位 |  |
| [ ] | `SN2` | `SN2` | 占位 |  |
| [ ] | `MN3` | `MN3` | 占位 |  |
| [ ] | `MN4` | `MN4` | 占位 |  |
| [ ] | `MN5` | `MN5` | 占位 |  |

## 附录：尚未在任何批次中的状态

当前 placeholder 家族总数（批次内）：49
