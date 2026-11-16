# GAMLSS 分布族 placeholder 迁移队列清单

> **Scope (P0-6):** `gamlss2` is **OUT OF SCOPE** for OmniLSS 1.0 (separate spec / version / evidence chain; it does not block 1.0). Out-of-scope exports are still enumerated in the per-symbol matrix for completeness and ARE included in raw totals (e.g. r_exports_total / surface_missing); they are NOT OmniLSS 1.0 delivery targets and do not count against 1.0 completeness. Filter on in_scope / in_scope_exports for the 1.0 figure.

> 生成日期：2026-11-10
> 生成方式：`tools/generate_placeholder_migration_queue.py`
> 状态来源：`omnilss.family_capabilities` 的代码派生 `implementation_status`（placeholder = 生产路由被拒，需重写）
> 批次划分：`docs/development/full-gamlss-migration-audit-and-roadmap-2026-09-11.md` 第 5 节 P1 批次

勾选规则：`- [ ]` 表示待迁移；`- [x]` 表示已完成（状态已离开 placeholder）。

## A1 偏态 t 与 SEP

| 勾选 | R family | registry | 当前状态 | 备注 |
|---|---|---|---|---|
| [x] | `ST1` | `ST1` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `ST2` | `ST2` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `ST3` | `ST3` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `ST3C` | `ST3C` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `ST4` | `ST4` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `ST5` | `ST5` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `SST` | `SST` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `SEP` | `SEP` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `SEP1` | `SEP1` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `SEP2` | `SEP2` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `SEP3` | `SEP3` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `SEP4` | `SEP4` | experimental | 状态已离开 placeholder（experimental） |

## A2 Box-Cox 与有界混合

| 勾选 | R family | registry | 当前状态 | 备注 |
|---|---|---|---|---|
| [x] | `BCCGo` | `BCCGO` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `BCPEo` | `BCPEO` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `BCTo` | `BCTO` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `BEINF0` | `BEINF0` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `BEINF1` | `BEINF1` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `BEOI` | `BEOI` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `BEZI` | `BEZI` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `LG` | `LG` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `LNO` | `LNO` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `LOGITNO` | `LOGITNO` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `NET` | `NET` | implemented | 状态已离开 placeholder（implemented） |

## A3 离散与零膨胀计数

| 勾选 | R family | registry | 当前状态 | 备注 |
|---|---|---|---|---|
| [x] | `DEL` | `DEL` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `DPO` | `DPO` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `SI` | `SI` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `WARING` | `WARING` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `YULE` | `YULE` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `NBF` | `NBF` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `GPO` | `GPO` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `NBII` | `NBII` | implemented | 状态已离开 placeholder（implemented） |
| [x] | `ZALG` | `ZALG` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `ZIP` | `ZIP` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `ZIP2` | `ZIP2` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `ZIPF` | `ZIPF` | experimental | 状态已离开 placeholder（experimental） |

## A4 正实数与其余连续

| 勾选 | R family | registry | 当前状态 | 备注 |
|---|---|---|---|---|
| [x] | `GB1` | `GB1` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `GB2` | `GB2` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `GIG` | `GIG` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `NO2` | `NO2` | implemented | 状态已离开 placeholder（implemented） |
| [x] | `NOF` | `NOF` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `PARETO` | `PARETO` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `PARETO1` | `PARETO1` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `PE2` | `PE2` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `PIG2` | `PIG2` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `RGE` | `RGE` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `WEI2` | `WEI2` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `exGAUS` | `EXGAUS` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `GAF` | `GAF` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `EGB2` | `EGB2` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `LOGSHASH` | `LOGSHASH` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `LQNO` | `LQNO` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `SN1` | `SN1` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `SN2` | `SN2` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `MN3` | `MN3` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `MN4` | `MN4` | experimental | 状态已离开 placeholder（experimental） |
| [x] | `MN5` | `MN5` | experimental | 状态已离开 placeholder（experimental） |

## 附录：尚未在任何批次中的状态

当前 placeholder 家族总数（批次内）：0
