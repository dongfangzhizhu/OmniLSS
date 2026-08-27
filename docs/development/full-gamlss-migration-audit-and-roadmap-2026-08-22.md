# OmniLSS R `gamlss` 全量迁移：现状审计与开发路线图

> 文档日期：2026-08-22
> 审计基线：仓库 commit `23b7aca`（`main` 分支），工作树仅 4 个教程文件有未提交修改。
> 文档性质：本文件在《DEVELOPMENT_PLAN.md》（Release 1 执行总纲）基础上，回答一个更宽的问题——**如何尽快把 R 语言 `gamlss` 生态的全部功能迁移到 Python（`omnilss`），先保证 `gamlss` + `gamlss.dist` 的功能一致性，再考虑其他相关包**。
> 后续开发会话应先读《DEVELOPMENT_PLAN.md》定位当前阶段，再读本文件定位"全量迁移"的差距与批次。

## 0. 当前开发进度（2026-08-28 更新）

> 本节是审计基线之后的滚动进度入口。下文第 1–11 节及附录 A 保留
> commit `23b7aca` 时的历史审计口径，不应再被当作当前能力矩阵；当前注册表
> 状态以 `docs/development/r-export-coverage.json` 和运行时
> `omnilss.family_capabilities` 为准。

### 0.1 已完成交付

| 任务 | 状态 | 提交与证据 |
|---|---|---|
| A0.1 环境与版本基线 | ⚠️ 部分完成 | commit `26acd6a` 已修复源码版本报告并补 WSL 基线；源码导入报告 `0.4.0`，但 2026-08-28 实测根 `.venv` 仍缺少 `pip`，因此“恢复 pip”验收尚未闭环 |
| A0.2 覆盖数据刷新 | ✅ 已完成 | commits `e9e7688`、`7504a60`；本次再次从代码刷新 `r-export-coverage.json`，基线见下节 |
| A0.3 注册缺失家族 | ✅ 已完成 | commit `cd15321`；`BEo/GEOMo/PARETO2o` 已注册并有 R d/p/q 一致性测试 |
| A0.4 placeholder 队列 | ✅ 已完成 | commit `59de2b0`；清单为 `docs/development/placeholder-migration-queue-2026-08-23.md` |
| A1 偏态 t 与 SEP | ✅ 已完成 | commits `55c4a6f`–`bd566bb`；ST1–5/ST3C、SEP/SEP1–4/SST 已补真实 d/p/q/r，批内 R 对照通过 |
| A2.1 有界混合与变换族 | ✅ 已完成 | commit `a43f5bc`；BEINF0/BEINF1/BEZI/BEOI/LNO/LOGITNO/NET/LG 已补真实 p/q/r，R 对照通过 |
| A2.2 Box-Cox `o` 变体 | ✅ 已完成 | commit `18e01e8`；BCCGo/BCTo/BCPEo 已按 R 源码重写并补 p/q/r，R 对照通过 |

### 0.2 当前能力与下一步

- **已提交代码基线（HEAD `18e01e8`）**：145 个注册家族，其中 1 implemented、
  83 experimental、25 compat_alias、36 placeholder；相对 R 的 117 个家族为
  1 implemented、81 experimental、2 compat_alias、33 placeholder、0 missing。
- **Tier 0**：A0.2–A0.4 已闭环；A0.1 仅剩根 `.venv` 恢复 `pip`。该环境问题
  不阻塞当前源码运行与 R 一致性测试，但不得继续标记为完全验收。
- **Tier 1**：A1 已完成，A2 路线图所列家族已完成；下一执行批次为 **A3 离散与
  零膨胀计数**。当前工作树已有 YULE p/q/r 的在途实现，尚未计入上述已提交基线。
- **逐函数覆盖限制**：`docs/migration-progress.md` 仍是 2026-05-23 的人工快照，
  尚未形成可重复生成的 171 符号逐函数表；在生成器落地前不可据此宣称 Tier 3 完成率。

### 0.3 本轮执行顺序

1. 完成并提交 YULE 的 d/p/q/r R 数值一致性与边界测试（A3.1）。
2. 继续按依赖拆分 A3：`WARING/DEL/DPO/SI`，再处理
   `NBF/GPO/NBII/ZALG/ZIP/ZIP2/ZIPF`；每个可独立验收的家族或紧耦合小组单独提交。
3. 每次家族状态变化后重跑 `tools/generate_r_export_coverage.py`，同步能力统计和
   placeholder 队列；不得把“有构造器”替代 R d/p/q/r 对照证据。

---

## 1. 执行摘要（结论先行）

1. **Release 1 已经冻结并基本闭环**：`omnilss` 0.4.0（2026-08-12）交付了最小可信核心——`NO/GA/PO/BI/NBI/BE/ZAGA/GU` 八个核心分布 E3 晋级、`RS/CG/MIXED` 三算法 R 对照、`pb/ps/cs` 三种平滑器的单一状态与 `newdata` 重建、`--require-r` fail-closed 门禁。这是继续全量迁移的可靠地基。
2. **但距离"gamlss 功能全部迁移成功"仍有很大缺口**：
   - `gamlss.dist`：R 侧 117 个分布家族（不含 `lot*` 绘图辅助）中，Python 注册表只有 **1 个生产级（`NO`）**、58 个 experimental、49 个 placeholder、6 个 compat_alias、**3 个未注册（`BEo/GEOMo/PARETO2o`，虽有构造器但不在 registry）**。
   - `gamlss` 核心：R 侧 `export()` 共 171 个符号。高层拟合/预测/诊断/模型选择 API 已大面积存在，但**平滑器只准入 `pb/ps/cs`**（`lo/random/re/ri/pcat/fp/bfp/cy/pvc/pbm/pbz/pbo/pbc/pbp/scs` 等未完成端到端验证），且若干函数仍是"部分实现"（`prof.dev`、`find.hyper`、`histSmoO/C/P`、`flexDist`、`calibration`、`rqres.plot`、`gamlss.family` 系统等）。
   - 周边 11 个 R 包（`gamlss.add`、`gamlss.cens`、`gamlss.tr`、`gamlss.mx`、`gamlss.spatial`、`gamlss.foreach`、`gamlss.prepdata`、`gamlss.ggplots`、`gamlss.data`、`gamlss2`、`WeatherGermany`）**基本未迁移**，只有零散对应物。
3. **推荐路线**：先完成 **Tier 1 = `gamlss.dist` 全部 117 家族的 d/p/q/r 数值等价**，再完成 **Tier 2 = 每个家族在 RS/CG/MIXED 下的拟合 R 对照**，然后 **Tier 3 = `gamlss` 核心 171 符号逐一闭环**，最后才进入周边包。每个 Tier 都以"固定 R golden artifact + 中文提交 + 门禁"为完成标准。
4. **环境可用**：Windows `.venv`（Python 3.10.16 + JAX 0.6.2 CPU）与 WSL `debianjaxgpu`（Python 3.11.15 + JAX 0.10.0 CUDA 12 + RTX 3060）均可运行；Windows 与 WSL 均安装了 R + `gamlss 5.5-0` + `gamlss.dist 6.1-1`，R 一致性测试可真实执行（本次抽样 `EXP/GA` 6 项全部通过）。
5. **最需要先做的事**（本周）：刷新过时的覆盖数据（`r-export-coverage.json`、`docs/migration-progress.md`）、修复 3 个未注册家族、把 `.venv` 里过期的 editable 安装升级到 0.4.0、为 49 个 placeholder 家族建立"占位→真实内核"的迁移队列。

---

## 2. 审计范围与方法

### 2.1 范围

| 对象 | 路径 | 说明 |
|---|---|---|
| R 原始项目 | `R/` | 12 个 R 包源码 + `WeatherGermany` 数据包 |
| Python 主实现 | `omnilss/src/omnilss/` | L1 统计引擎（JAX） |
| Python 参考实现 | `gamlss-python/` | 外部项目 `fzhao70/gamlss-python`，已被 `.gitignore` 忽略，纯 NumPy/SciPy 转录，仅作交叉参考 |
| 在线学习层 | `ondil/src/ondil/` | L2 在线 GAMLSS |
| 商业服务层 | `omnilss-pro/`、`omnilss-server/` | L3 gRPC/HTTP 服务与 SDK |
| 验证资产 | `omnilss/tests/`、`benchmarks/`、`tools/`、`docs/` | 测试、基准、门禁工具、文档 |

### 2.2 方法

- **R 侧**：解析各包 `NAMESPACE` 的 `export()` 与 `R/*.R` 的函数定义，统计导出符号与家族名。
- **Python 侧**：以 `omnilss.distribution_registry`（142 个注册家族）与 `omnilss.family_capabilities`（运行时能力矩阵，AST 派生的 `implemented/experimental/compat_alias/placeholder`）为权威状态源；辅以 `omnilss/__init__.py` 顶层导出面与测试文件扫描。
- **环境侧**：直接探测 Windows `.venv`、`omnilss/.venv`、WSL `debianjaxgpu` 与 Windows/WSL 的 R 环境。
- **一致性抽样**：在 Windows `.venv` 下真实运行 R 一致性测试 `test_r_consistency_exp.py` 与 `test_r_consistency_ga.py`，6 项全部通过。

### 2.3 状态口径（重要）

`family_capabilities` 的 `implementation_status` 是**代码派生的保守口径**，不等于"是否可 import"：

- `implemented`：生产级，当前只有 `NO`。
- `experimental`：有真实数值内核与部分测试，但证据不足以上生产路由（默认开发/实验可用）。
- `compat_alias`：公开构造器复用另一个数值内核，**不计入已迁移**（如 `ST1-4` 目前克隆 `ST3C`，`DBI/DBURR12` 等 skeleton 别名）。
- `placeholder`：公开表面含占位/近似数值行为，生产路由被拒绝，**需要重写**。
- `missing`：R 有而 Python 注册表没有。

本文所有"完成率"都按此口径，避免把"能 import"当成"迁移完成"。

---

## 3. 环境审计

### 3.1 Python 环境

| 环境 | 路径 | Python | 关键包 | 用途/状态 |
|---|---|---|---|---|
| Windows 主环境 | `.venv/` | 3.10.16 | JAX 0.6.2（CPU）、NumPy 2.2.6、SciPy 1.15.3、pandas 2.3.3、hypothesis 6.152.1、pytest 可用 | 主开发/测试环境；`import omnilss` 自动开启 x64（实测 `jax_enable_x64=True`） |
| omnilss 二级环境 | `omnilss/.venv/` | 3.12.9 | 未逐一核验 | 疑似历史遗留环境，需确认是否仍使用 |
| WSL Debian GPU 环境 | `debianjaxgpu/` | 3.11.15 | JAX 0.10.0 + `jax_cuda12_plugin/pjrt`（CUDA 12）、GPU 可用 | WSL2 内 GPU 训练/推理；`nvidia-smi` 可见 RTX 3060 12GB |

**发现（需处理）**：

- E1. `.venv` 中安装的 `omnilss` 是 **0.3.0**（editable），而 `omnilss/pyproject.toml` 已是 **0.4.0** → editable 安装过期，可能让本地测试跑在旧代码/旧元数据上。
- E2. `.venv` 无 `pip` 模块（`python -m pip` 不可用），只能用已装好的 pytest/python 或重建环境。
- E3. Windows JAX 0.6.2 与 WSL JAX 0.10.0 版本不一致；数值一致性与性能基准必须标注后端版本，避免跨环境结论漂移。
- E4. `omnilss/.venv`（3.12.9）与 `.venv`（3.10.16）并存，仓库文档未说明其用途，建议确认后清理或写入环境基线文档。

### 3.2 R 环境

| 环境 | R 版本 | 已安装 gamlss 生态包 |
|---|---|---|
| Windows 系统 R | 4.6.0 | `gamlss 5.5-0`、`gamlss.dist 6.1-1`、`gamlss.data 6.0-9` |
| WSL Debian R | 4.5.0 | `gamlss 5.5-0`、`gamlss.dist 6.1-1`、`gamlss.data 6.0-7` |

**发现（需处理）**：

- E5. 两个 R 环境都**只装了核心三件套**；`gamlss.add/cens/tr/mx/spatial/foreach/ggplots/prepdata/gamlss2` 未安装。迁移周边包时需先从 `R/` 源码安装，或至少建立固定的 R library。
- E6. `docs/development/environment-baseline-2026-07-23.md` 记录的版本（`gamlss 5.5-3`、`gamlss.dist 6.1-8`）与当前实际安装版本（5.5-0/6.1-1）不一致，基线文档需更新；golden artifact 以 `R/` 内 vendored 源码为准。

---

## 4. R 原始项目盘点（`R/`）

### 4.1 各包规模

| 包 | `export()` 符号 | R 文件数 | 函数定义数 | 一句话定位 |
|---|---:|---:|---:|---|
| `gamlss` | 171 | 79 | 326 | 核心：拟合/预测/诊断/模型选择/平滑/绘图 |
| `gamlss.dist` | 661 | 124 | 640 | 分布家族 d/p/q/r 与辅助函数 |
| `gamlss2` | 44 | 24 | 221 | 新一代 `gamlss2()` API（formula/list、stepwise、GDF） |
| `gamlss.ggplots` | 89 | 67 | 253 | 基于 ggplot2 的诊断/拟合/预测绘图 |
| `gamlss.prepdata` | 59 | 22 | 115 | 数据预处理/诊断/切分工具 |
| `gamlss.foreach` | 13 | 8 | 40 | 并行 centiles.boot/PCR/QR 等 |
| `gamlss.mx` | 17 | 2 | 40 | 有限混合与 NP 分布（`gamlssMX`/`gamlssNP`） |
| `gamlss.add` | 22 | 8 | 33 | 附加项：GA/BA/NN/FK/tr/centilesTwo |
| `gamlss.spatial` | 9 | 5 | 23 | MRF/GMRF 空间平滑 |
| `gamlss.cens` | 10 | 5 | 6 | 删失响应（`gen.cens`） |
| `gamlss.tr` | 13 | 7 | 8 | 截断分布与尾部拟合（`gen.trun`、`fitTail`） |
| `gamlss.data` | 7 | 1 | 0 | 数据集（`abdom`、`usair`、`aids` 等） |
| `WeatherGermany` | 0 | 0 | 0 | 数据包（inst/data） |
| **合计** | **1115** | **352** | **1705** | （另有 4 个 .R 在 tests/inst 等位置，全库 356 个 .R） |

### 4.2 `gamlss.dist` 家族

- 从 `NAMESPACE` 提取 d/p/q/r 家族名：**117 个**（已排除 `lotBEINF` 等绘图辅助）。这 117 个家族与 8 个 `lot*` 绘图辅助、其他辅助函数一起构成 `gamlss.dist` 的 661 个 `export()` 符号。
- 家族覆盖：连续实数域、正实数域、(0,1) 有界、Box-Cox 变换族、离散计数、零膨胀/零调整、混合、ST/SEP 偏态 t 族等。

### 4.3 `gamlss` 核心 171 符号分类（基于 `docs/migration-progress.md` 的 2026-05-23 快照口径）

| 类别 | 符号数（快照） | 备注 |
|---|---:|---|
| 拟合引擎 | 14 | `gamlss/gamlss.control/glim.control/gamlssML/gamlssVGD/gamlssCV/update/refit` 等 |
| 预测与线性预测器 | 18 | `predict/predictAll/lpred/lp/fv/prodist/Formulae2data` 等 |
| 模型诊断与残差 | 38 | `rqres/wp/Q.stats/LR.test/VC.test/devianceIncr/hatvalues/logLik/vcov` 等 |
| 绘图与可视化 | 32 | `term.plot/fittedPlot/pdf.plot/plot2way/par.plot/vis.lo` 等 |
| 模型选择与逐步回归 | 28 | `GAIC/stepGAIC/stepGAICAll.A/B/stepTGD/drop1All/add1All` 等 |
| 平滑项 | 62 | `pb/ps/cs` 已准入；`lo/random/re/ri/pcat/fp/bfp/cy/pvc/pbm/pbz/pbo/pbc/pbp/scs/pp` 等未完成 |
| 分位数图表与 LMS | 18 | `centiles/centiles.pred/centiles.fan/centiles.split/lms/quantSheets` 等 |
| 其他工具 | 6 | `gamlssNews` 等 |

---

## 5. Python 实现盘点

### 5.1 `omnilss`（L1 主实现）

- 包位置：`omnilss/src/omnilss/`，共 253 个 `.py` 文件（含 `algorithms/`、`core/`、`smooth/`、`api/`、`deep/` 等子包）。
- 顶层 `__init__.py`（1112 行）导出面很大：拟合（`gamlss`、`gamlss_ml`、`gamlssML`、`gamlssCV/VGD`）、预测（`predict`、`predict_all`、`lpred`）、诊断（`wp`、`q_stats`、`likelihood_ratio_test`、`vuong_clarke_test`、`devianceIncr`、`hatvalues`）、模型选择（`step_gaic`、`step_gaic_all`、`step_tgd`、`addterm/dropterm`）、平滑（`getSmo`、`plot_smoothers`、`getZmatrix`）、LMS/centiles（`lms`、`z_scores`、`centileSK`、`centileKurt`、`quantSheets`）、操作符（`coef`、`fitted`、`residuals`、`deviance`、`gaic`、`vcov`、`refit`）等。
- **分布注册表**：`distribution_registry._BUILTIN_FAMILY_FACTORIES` 注册 **142 个家族**。
- **能力矩阵**（`family_capabilities`，2026-08-22 实测）：

| 状态 | 数量 | 含义 |
|---|---:|---|
| `implemented` | 1 | `NO`（生产级基线） |
| `experimental` | 60 | 有真实内核，证据不足以上生产 |
| `compat_alias` | 29 | 复用其他内核的别名/骨架（不计入迁移） |
| `placeholder` | 52 | 含占位/近似行为，生产路由拒绝 |
| `missing` | 0（注册表内部口径） | 注册表内无 missing；但相对 R 有 3 个家族未注册 |

- 检查到的事实：142 个注册家族的 `d/p/q/r` 属性均非 `None`（所以"属性存在"不能说明问题）；真正的数值等价必须靠 R 对照测试与 golden artifact 证明。

### 5.2 `gamlss-python`（参考项目，gitignored）

- 外部参考项目（`fzhao70/gamlss-python`），纯 NumPy/SciPy/Pandas/Patsy，无 JAX；版本 0.2.0。
- 声称对 R `gamlss 5.5-0`/`gamlss.dist 6.1-1` 做数值级转录（系数与 R 差异 ~1e-9），实现 35 个分布模块与 `pb/pbz` 平滑。
- 定位：**仅作交叉参考与公式核对**，不是本项目目标实现。目标实现是 `omnilss`（JAX 可微、可 GPU、可服务化）。

### 5.3 其他 Python 组件

| 组件 | 路径 | 状态 |
|---|---|---|
| `ondil` | `ondil/src/ondil/` | L2 在线学习，有独立分布/估计器；其测试在根 `.venv` 下有 43 个收集错误（依赖未装或路径问题），Phase 6 待修 |
| `omnilss-pro` | `omnilss-pro/src/omnilss_pro/` | L3 SDK，Phase 8.1 已补 capability/version/health 端点 |
| `omnilss-server` | `omnilss-server/` | gRPC/HTTP 服务，Phase 5 安全加固已完成 |

### 5.4 测试与验证资产

- `omnilss/tests/`：**346 个测试文件**；其中 **105 个 `test_r_consistency_*.py`**（R 一致性）。
- 根目录 `.venv` 下 `pytest --collect-only`：**2752 个用例收集成功 + 43 个收集错误（集中在 `ondil/tests`）**。
- R 一致性在 Windows 可真实执行：本次抽样 `test_r_consistency_exp.py`（3 项）与 `test_r_consistency_ga.py`（3 项）全部通过。
- Golden 体系：`omnilss/tests/fixtures/r_golden/generate_golden.R`、`regenerated/`、manifest（R 4.5.0 / gamlss 5.5.0 / gamlss.dist 6.1.1 / seed / djb2 hash）；`tools/stage0_gate.py`、`tools/stage_completion_checker.py` 提供 fail-closed 门禁与中文提交校验。
- CI：`.github/workflows/ci.yml` Group B 用 `r-lib/actions/setup-r@v2` 安装 R 与 gamlss，fail-closed 重跑 R 一致性（出现 skip 即失败）。

---

## 6. 覆盖与差距分析

### 6.1 `gamlss.dist` 家族覆盖（重点）

把 R 的 117 个家族与 Python 注册表（大小写不敏感）对照：

| 分类 | 数量 | 明细 |
|---|---:|---|
| 已注册且生产级 | 1 | `NO` |
| 已注册 experimental | 58 | 含 8 核心（`GA/PO/BI/NBI/BE/ZAGA/GU`）与 `EXP/IG/LO/WEI/TF/JSU/BCCG/BCT/BCPE/GG` 等；另有 49 个 placeholder（含 `GB2` 等，需重写） |
| 已注册 placeholder | 49 | 见附录全表，需重写数值内核 |
| 已注册 compat_alias | 6 | `DBI/DBURR12/ST1/ST2/ST3/ST4` |
| **未注册（missing）** | **3** | **`BEo`、`GEOMo`、`PARETO2o`**——`omnilss/__init__.py` 已 import 三者的构造器（`omnilss.BEo` 等存在），但 `distribution_registry` 未注册，按名字 `resolve()` 会失败，拟合路由无法使用 |
| 有 R 一致性测试文件提及 | 106/117 | 见附录；测试文件"提及"是覆盖上界，不等于全部通过 |

**八核心 E3 已晋级**（Release 1）：`NO/GA/PO/BI/NBI/BE/ZAGA/GU`，d/p/q/r + hypothesis 属性测试 + R 一致性（rtol ≤ 1e-3）+ golden replay（2e-4）。

**已知数值遗留**（来自 `CHANGELOG 0.4.0` Notes，仍是后续工作输入）：
- `CG-ZAGA`、`CG-GU`、`Mixed-GU` 收敛缺陷；14 个非核心族拟合差异已列入例外清单。
- `ZASICHEL/ZISICHEL` 的 d/p/q 存在 NaN；Windows 下 vendored R 源包安装是既有阻塞项。

### 6.2 `gamlss` 核心 API 覆盖

上一份逐函数审计是 `docs/development/r-gamlss-coverage-and-plan-2026-05-22.md` 与 `docs/migration-progress.md`（2026-05-23 快照：`gamlss` 核心 51%、`gamlss.dist` 76%、综合 63%）。该快照**已过时**（Phase 1–4 修复了 `pb/ps/cs`、qTF/pIG/qIG/WEI3/BCCG-BCT-BCPE、MIXED 等），但仍是目前唯一的逐函数清单，必须刷新（见路线图 A0.2）。

按当前代码核对的关键差距：

| 缺口 | 现状 | 优先级 |
|---|---|---|
| 平滑器准入 | 只有 `pb/ps/cs` 通过端到端验证；`lo` 只是轻量诊断实现；`random/re/ri/pcat/fp/bfp/cy/pvc/pbm/pbz/pbo/pbc/pbp/scs` 等未完成 | 高 |
| `gamlss.family` 系统 | `as.family/checklink/show.link/make.link.gamlss` 等未完成（`links.py` 只有基础 link 函数） | 高 |
| `stepGAIC.VR/CH`、`stepTGDAll.A`、`gamlss.scope` | 未实现（`stepGAICAll_A/B` 已有） | 中 |
| `prof.dev/prof.term` | 只结构化数据，不执行重拟合 | 中 |
| `find.hyper` | 只汇总预计算结果，不做网格搜索重拟合 | 中 |
| `histSmoO/C/P` | 四变体实现相同，未区分平滑方法 | 中 |
| `flexDist` | 只算经验统计量，未拟合参数分布 | 中 |
| `calibration`、`predict.lms` 等 LMS 完整链路 | 部分实现 | 中 |
| `rqres.plot`、`edf`（单项）、`predictAll` 顶层导出 | 未实现/未导出 | 低 |
| 分布辅助 `plotBEINF/meanZAGA/tofySICHEL/zetaP/GetBI_C/get_C` 等 | 未实现 | 低（随家族批次补齐） |

### 6.3 周边包覆盖

| R 包 | Python 现状 | 建议次序 |
|---|---|---|
| `gamlss.tr` | 部分：`FitTail.py`/`logSurv`/`loglogSurv` 已有；`gen.trun/trun.d/p/q/r/fitTailAll` 未完成 | 第一批（与分布/fitTail 强相关） |
| `gamlss.cens` | 未迁移（`gen.cens/cens.d/p/q`） | 第一批（分布回归常见需求） |
| `gamlss.mx` | 雏形：`omnilss/core/mixture.py` 存在，`gamlssMX/gamlssNP` 未完成 | 第二批 |
| `gamlss.add` | 未迁移（`ga/ba/nn/fk/tr/centilesTwo`） | 第二批 |
| `gamlss.spatial` | 未迁移（MRF/GMRF） | 第三批 |
| `gamlss.foreach` | 未迁移（并行 centiles.boot/PCR/QR） | 第三批 |
| `gamlss.prepdata` | 未迁移（59 个数据工具） | 第三批 |
| `gamlss.ggplots` | 未迁移（89 个 ggplot2 绘图） | 第四批（依赖成熟绘图方案） |
| `gamlss.data` | 数据集未打包（R 数据在 `R/gamlss.data/data/`） | 随需 |
| `gamlss2` | 仅 `distributions3.py` 接口对齐；`gamlss2()` 主 API 未迁移 | 长期（新一代 API，单独规划） |
| `WeatherGermany` | 未迁移 | 随需 |

### 6.4 一致性验证体系现状

- 已具备：固定 R golden artifact 管线（生成脚本、manifest、hash）、`--require-r` fail-closed、阶段完成检查器（中文提交校验）、能力矩阵生成器（`tools/generate_r_export_coverage.py`）。
- 待补：`docs/development/r-export-coverage.json` 生成于 2026-07-23（状态分布 1/55/29/57），与当前实测（1/60/29/52）不一致，需重跑生成器；`docs/migration-progress.md` 停在 2026-05-23。

---

## 7. 关键风险清单

| # | 风险 | 影响 | 处置 |
|---|---|---|---|
| R1 | 3 个 R 家族（`BEo/GEOMo/PARETO2o`）未注册 | 按名字拟合直接失败 | A0.3 立即注册并补测试 |
| R2 | 49 个 placeholder 家族生产路由全拒 | "全量迁移"完成率被卡住 | A1–A4 分批重写内核 |
| R3 | `ST1-4` 是 compat_alias（克隆 `ST3C`） | 与 R 的 ST1/ST2/ST4 数值不等价 | 随 ST 批次重写 |
| R4 | 覆盖数据陈旧（`r-export-coverage.json`、`migration-progress.md`） | 决策依据漂移 | A0.2 刷新 |
| R5 | `.venv` editable 安装 0.3.0 vs pyproject 0.4.0 | 本地测试可能跑旧代码 | A0.1 修复 |
| R6 | Windows JAX 0.6.2 vs WSL JAX 0.10.0 | 跨环境数值/性能结论不可比 | 版本冻结或统一升级 |
| R7 | 周边 11 个 R 包基本未迁移 | 范围蔓延风险 | 严格按 Tier 3 再启动 |
| R8 | `ondil` 43 个收集错误 | L2 层测试不可信 | Phase 6 修复 |
| R9 | 已知拟合缺陷（CG-ZAGA/CG-GU/Mixed-GU、14 非核心族差异、ZASICHEL/ZISICHEL NaN） | 一致性承诺破口 | 纳入 Tier 2 拟合批次 |

> 说明：`DEVELOPMENT_PLAN.md` 中的 C1–C7/H1–H9 审计缺陷，Phase 1–5 已按计划关闭大部分（C1–C7 由 Phase 1 修复、H7/H8 由 Phase 5 加固、部分性能项在 Phase 7）。本表列的是**当前仍存在**的迁移范围风险，不是重复旧审计。

---

## 8. 全量迁移路线图（接下来的开发工作）

> 执行口径：每个任务完成 = 实现 + 测试 + 固定 R golden artifact + **中文提交（含任务 ID）**，经 `tools/stage_completion_checker.py` 校验后才算数。R 对照优先于自证；无 R 环境只能标 smoke-only，不得计入完成率。

### Tier 0 —— 刷新真相基座（第 1 周，阻塞后续批次）

| 任务 | 内容 | 验收 |
|---|---|---|
| A0.1 | 修复 `.venv`：重建或升级 editable 安装到 0.4.0，恢复 `pip`；在环境基线文档记录 `.venv`/`omnilss/.venv`/`debianjaxgpu` 三者的职责与版本 | `python -c "import omnilss; print(omnilss.__version__)"` 输出 0.4.0；三环境基线文档更新 |
| A0.2 | 重跑 `tools/generate_r_export_coverage.py` 刷新 `r-export-coverage.json/.md`；用脚本重新生成 `docs/migration-progress.md` 的逐函数表（或新建 `r-export-coverage` 系列） | 覆盖 JSON 状态分布与 `family_capabilities` 实测一致；逐函数表标注生成日期与 commit |
| A0.3 | 注册 `BEo/GEOMo/PARETO2o` 到 `distribution_registry`，补 R 一致性测试 | `resolve("BEo")` 等成功；三个家族 d/p/q/r 与 R 对照通过（至少解析层面） |
| A0.4 | 为 49 个 placeholder 家族建立迁移队列清单（本文件附录 A 即为初版），并在 `.kiro` 或 `docs/development` 落一份可勾选清单 | 清单文件存在，与能力矩阵一致 |

### Tier 1 —— `gamlss.dist` 全家族 d/p/q/r 数值等价（核心，预计 6–10 周）

**目标**：117 个 R 家族全部达到 `experimental` 以上，d/p/q/r 与 R 在固定点集上 rtol ≤ 1e-3（特殊函数按其数值条件），support/boundary/monotonicity 属性测试通过。

**批次划分建议**（按依赖与风险排序，每批一个中文提交系列）：

| 批次 | 家族 | 说明 |
|---|---|---|
| A1 偏态 t 与 SEP | `ST1/ST2/ST3/ST3C/ST4/ST5/SST/SEP/SEP1-4` | 当前 ST1-4 是克隆；需按 R 源码重写，依赖 `dpqr_functions` 的数值积分/级数 |
| A2 Box-Cox 与有界混合 | `BCCG/BCT/BCPE` 的 `o` 变体、`BEINF0/BEINF1/BEOI/BEZI`、`LG/LNO/LOGITNO/NET` | Box-Cox 主族已 experimental，补 `o` 参数化；有界混合需重写混合似然 |
| A3 离散与零膨胀计数 | `DEL/DPO/SI/WARING/YULE/NBF/GPO/NBII/ZALG/ZIP/ZIP2/ZIPF` | 重点核对递推（PIG/SICHEL 类）、零膨胀/零调整参数化 |
| A4 正实数与其余连续 | `GB1/GB2/GIG/NO2/NOF/PARETO/PARETO1/PE2/PIG2/RGE/WEI2/exGAUS/GAF/EGB2/LOGSHASH/LQNO/SN1/SN2/MN3/MN4/MN5` | `GB2` 等虽有 dpqr 但被标 placeholder，需按 R 公式/截断归一化核对 |

**每批验收**：`--require-r` 批内 suite 全绿；`family_capabilities` 中对应状态由 placeholder/compat_alias → experimental；golden manifest 覆盖批内每家族 d/p/q/r。

### Tier 2 —— 全家族拟合一致性（RS/CG/MIXED）

**目标**：Tier 1 完成后，对每个非 placeholder 家族至少跑通 RS 拟合 R 对照（记录收敛、deviance、系数、fitted values）；八核心之外的家族逐批晋升 E3。

- B1：为每个家族生成固定 R 拟合 golden（`generate_golden.R` 扩展），覆盖零响应/边界/极端均值等计数场景。
- B2：修复已知收敛缺陷（`CG-ZAGA`、`CG-GU`、`Mixed-GU`、14 个非核心族差异）。
- B3：把"每族 RS 一致性"纳入 `--require-r` 全量门禁，替换当前的抽样门禁。

### Tier 3 —— `gamlss` 核心 171 符号闭环

- C1：刷新逐函数覆盖表（A0.2 产出）后，逐项关闭未实现/部分实现符号，优先级：`gamlss.family` 系统 → `stepGAIC.VR/CH`、`gamlss.scope` → `prof.dev/prof.term`、`find.hyper`、`histSmoO/C/P`、`flexDist`、`calibration` → `rqres.plot`、`edf`、顶层 `predictAll`。
- C2：平滑器扩展：`lo`（从轻量实现升级为端到端拟合）、`random/re/ri`、`pcat`、`fp/bfp`、`pvc`、`pbm/pbz/pbo/pbc/pbp`、`scs`；每个都走 `smoother_admission` 的七项能力链（parsing/basis/penalty/lambda_edf/fitting/prediction/r_golden）后才准入。
- C3：绘图函数（`term.plot`、`fittedPlot`、`pdf.plot` 等）做数值数据层对齐；图形外观层允许差异，但数据必须一致。

### Tier 4 —— 周边包（`gamlss`/`gamlss.dist` 达标后再启动）

| 批次 | 包 | 目标 |
|---|---|---|
| D1 | `gamlss.tr`、`gamlss.cens` | `gen.trun/trun.d/p/q/r/fitTailAll`、`gen.cens/cens.d/p/q` 与核心拟合打通 |
| D2 | `gamlss.mx`、`gamlss.add` | `gamlssMX/gamlssNP`、`ga/ba/nn/fk/tr/centilesTwo` |
| D3 | `gamlss.spatial`、`gamlss.foreach`、`gamlss.prepdata` | MRF/GMRF；并行工具；数据工具 |
| D4 | `gamlss.ggplots`、`gamlss.data`、`WeatherGermany` | 绘图数据层对齐 + 数据集打包 |
| D5 | `gamlss2` | 单独规划：新一代 API（formula/list/stepwise/GDF），建议作为 1.0 之后的长期路线 |

### 里程碑建议

| 里程碑 | 内容 | 版本建议 |
|---|---|---|
| M1 | Tier 0 + Tier 1 完成（117 家族 d/p/q/r 全 experimental+） | 0.5.0 |
| M2 | Tier 2 完成（全家族 RS 拟合 R 对照 + 已知缺陷清零） | 0.6.0 |
| M3 | Tier 3 完成（`gamlss` 核心 171 符号闭环） | 0.9.0 |
| M4 | Tier 4 D1–D4 完成（周边包迁移） | 1.0.0 |

---

## 9. 开发工作流与命令

### 9.1 Windows（主开发，CPU）

```powershell
# 使用仓库根 .venv
.\.venv\Scripts\python.exe --version        # 3.10.16

# x64 已由 omnilss/__init__ 自动启用（导入即生效），无需手动设置
.\.venv\Scripts\python.exe -c "import omnilss, jax; print(jax.config.jax_enable_x64)"

# 运行指定测试（R 一致性可真实执行，Windows 已装 R 4.6.0 + gamlss）
.\.venv\Scripts\python.exe -m pytest omnilss/tests/test_r_consistency_exp.py -q

# 收集统计
.\.venv\Scripts\python.exe -m pytest --collect-only -q
```

### 9.2 WSL Debian（GPU，性能与大规模拟合）

```bash
cd /mnt/d/AI/my/githubprojs/OmniLSS
./debianjaxgpu/bin/python --version       # 3.11.15
./debianjaxgpu/bin/python -c "import jax; print(jax.__version__); print(jax.devices())"
# 预期可见 cuda:0（RTX 3060）
```

### 9.3 R golden artifact 生成（一致性证据）

```bash
# Windows
Rscript omnilss/tests/fixtures/r_golden/generate_golden.R

# WSL（R 4.5.0 + gamlss 5.5.0 + gamlss.dist 6.1.1）
wsl -e bash -lc "cd /mnt/d/AI/my/githubprojs/OmniLSS && Rscript omnilss/tests/fixtures/r_golden/generate_golden.R"
```

### 9.4 门禁与提交纪律

```bash
# 阶段门禁（需要 R；缺 R 或 skip 即失败）
.\.venv\Scripts\python.exe tools/stage0_gate.py check

# 任务完成记录校验（中文提交、任务 ID、证据文件）
.\.venv\Scripts\python.exe tools/stage_completion_checker.py check --task 0.5
```

- 每个任务单独提交，提交主题用中文并标识任务 ID（例如 `完成A1.1：ST1-4 按 R 源码重写并补 R 一致性`）。
- 涉及数值行为的 PR 必须附 R 对照结果或 golden artifact 路径；无 R 环境只能标 smoke-only。

---

## 10. 待决策项（建议尽快拍板）

| # | 决策 | 建议默认 | 影响 |
|---|---|---|---|
| Q1 | 全量迁移的第一个里程碑是否定为"117 家族 d/p/q/r 全 experimental+"（M1） | 是 | 决定 Tier 1 范围 |
| Q2 | `gamlss2`（新一代 API）是否纳入当前全量迁移，还是放到 1.0 后单独规划 | 放到 1.0 后 | 决定 D5 是否提前 |
| Q3 | Windows JAX 是否升级到与 WSL 一致的 0.10.0 | 升级并冻结 | 消除跨环境版本漂移 |
| Q4 | `omnilss/.venv`（3.12.9）是否保留 | 确认用途后清理或归档 | 减少环境混乱 |
| Q5 | 周边包优先级是否按 D1→D4（tr/cens → mx/add → spatial/foreach/prepdata → ggplots/data） | 是 | 决定 Tier 4 顺序 |
| Q6 | `gamlss.ggplots`（ggplot2 依赖）是否只对齐数据层、不追求图形外观一致 | 是 | 避免绘图外观返工 |

---

## 11. 附录

### 附录 A：R `gamlss.dist` 117 家族 ↔ Python 状态全表

> 生成方式：解析 `R/gamlss.dist/NAMESPACE` 得到 117 个 d/p/q/r 家族（排除 `lot*` 绘图辅助），与 `omnilss.distribution_registry`（大小写不敏感）及 `family_capabilities.implementation_status` 对照；"R-test" = 是否在 `omnilss/tests/test_r_consistency_*.py` 中被提及（覆盖上界，非通过证明）。
> `missing` = 有顶层构造器但未注册进 registry（`BEo/GEOMo/PARETO2o`）。

| R family | Python registry | status | R-test |
|---|---|---|---|
| `BB` | `BB` | experimental | Y |
| `BCCG` | `BCCG` | experimental | Y |
| `BCCGo` | `BCCGO` | experimental | . |
| `BCPE` | `BCPE` | experimental | Y |
| `BCPEo` | `BCPEO` | experimental | . |
| `BCT` | `BCT` | experimental | Y |
| `BCTo` | `BCTO` | experimental | . |
| `BE` | `BE` | experimental | Y |
| `BEINF` | `BEINF` | experimental | Y |
| `BEINF0` | `BEINF0` | placeholder | Y |
| `BEINF1` | `BEINF1` | placeholder | Y |
| `BEo` | — | missing | . |
| `BEOI` | `BEOI` | placeholder | Y |
| `BEZI` | `BEZI` | placeholder | Y |
| `BI` | `BI` | experimental | Y |
| `BNB` | `BNB` | experimental | Y |
| `DBI` | `DBI` | compat_alias | Y |
| `DBURR12` | `DBURR12` | compat_alias | Y |
| `DEL` | `DEL` | placeholder | Y |
| `DPO` | `DPO` | placeholder | Y |
| `EGB2` | `EGB2` | placeholder | Y |
| `exGAUS` | `EXGAUS` | placeholder | . |
| `EXP` | `EXP` | experimental | Y |
| `GA` | `GA` | experimental | Y |
| `GAF` | `GAF` | placeholder | Y |
| `GB1` | `GB1` | placeholder | Y |
| `GB2` | `GB2` | placeholder | Y |
| `GEOM` | `GEOM` | experimental | Y |
| `GEOMo` | — | missing | . |
| `GG` | `GG` | experimental | Y |
| `GIG` | `GIG` | placeholder | Y |
| `GP` | `GP` | experimental | Y |
| `GPO` | `GPO` | placeholder | Y |
| `GT` | `GT` | experimental | Y |
| `GU` | `GU` | experimental | Y |
| `IG` | `IG` | experimental | Y |
| `IGAMMA` | `IGAMMA` | experimental | Y |
| `JSU` | `JSU` | experimental | Y |
| `JSUo` | `JSUO` | experimental | Y |
| `LG` | `LG` | placeholder | Y |
| `LNO` | `LNO` | placeholder | Y |
| `LO` | `LO` | experimental | Y |
| `LOGITNO` | `LOGITNO` | placeholder | Y |
| `LOGNO` | `LOGNO` | experimental | Y |
| `LOGNO2` | `LOGNO2` | experimental | Y |
| `LOGSHASH` | `LOGSHASH` | placeholder | Y |
| `LQNO` | `LQNO` | placeholder | Y |
| `MN3` | `MN3` | placeholder | Y |
| `MN4` | `MN4` | placeholder | Y |
| `MN5` | `MN5` | placeholder | Y |
| `NBF` | `NBF` | placeholder | . |
| `NBI` | `NBI` | experimental | Y |
| `NBII` | `NBII` | placeholder | Y |
| `NET` | `NET` | placeholder | Y |
| `NO` | `NO` | implemented | Y |
| `NO2` | `NO2` | placeholder | Y |
| `NOF` | `NOF` | placeholder | Y |
| `PARETO` | `PARETO` | placeholder | Y |
| `PARETO1` | `PARETO1` | placeholder | Y |
| `PARETO1o` | `PARETO1O` | experimental | Y |
| `PARETO2` | `PARETO2` | experimental | Y |
| `PARETO2o` | — | missing | . |
| `PE` | `PE` | experimental | Y |
| `PE2` | `PE2` | placeholder | Y |
| `PIG` | `PIG` | experimental | Y |
| `PIG2` | `PIG2` | placeholder | Y |
| `PO` | `PO` | experimental | Y |
| `RG` | `RG` | experimental | Y |
| `RGE` | `RGE` | placeholder | Y |
| `SEP` | `SEP` | placeholder | Y |
| `SEP1` | `SEP1` | placeholder | Y |
| `SEP2` | `SEP2` | placeholder | Y |
| `SEP3` | `SEP3` | placeholder | Y |
| `SEP4` | `SEP4` | placeholder | Y |
| `SHASH` | `SHASH` | experimental | Y |
| `SHASHo` | `SHASHO` | experimental | Y |
| `SHASHo2` | `SHASHO2` | experimental | Y |
| `SI` | `SI` | placeholder | Y |
| `SICHEL` | `SICHEL` | experimental | Y |
| `SIMPLEX` | `SIMPLEX` | experimental | Y |
| `SN1` | `SN1` | placeholder | Y |
| `SN2` | `SN2` | placeholder | Y |
| `SST` | `SST` | placeholder | Y |
| `ST1` | `ST1` | compat_alias | Y |
| `ST2` | `ST2` | compat_alias | Y |
| `ST3` | `ST3` | compat_alias | Y |
| `ST3C` | `ST3C` | placeholder | Y |
| `ST4` | `ST4` | compat_alias | Y |
| `ST5` | `ST5` | placeholder | Y |
| `TF` | `TF` | experimental | Y |
| `TF2` | `TF2` | experimental | Y |
| `WARING` | `WARING` | placeholder | Y |
| `WEI` | `WEI` | experimental | Y |
| `WEI2` | `WEI2` | placeholder | Y |
| `WEI3` | `WEI3` | experimental | Y |
| `YULE` | `YULE` | placeholder | Y |
| `ZABB` | `ZABB` | experimental | Y |
| `ZABI` | `ZABI` | experimental | . |
| `ZABNB` | `ZABNB` | experimental | Y |
| `ZAGA` | `ZAGA` | experimental | Y |
| `ZAIG` | `ZAIG` | experimental | Y |
| `ZALG` | `ZALG` | placeholder | Y |
| `ZANBI` | `ZANBI` | experimental | . |
| `ZAP` | `ZAP` | experimental | Y |
| `ZAPIG` | `ZAPIG` | experimental | Y |
| `ZASICHEL` | `ZASICHEL` | experimental | Y |
| `ZAZIPF` | `ZAZIPF` | experimental | Y |
| `ZIBB` | `ZIBB` | experimental | Y |
| `ZIBI` | `ZIBI` | experimental | . |
| `ZIBNB` | `ZIBNB` | experimental | Y |
| `ZINBF` | `ZINBF` | experimental | Y |
| `ZINBI` | `ZINBI` | experimental | Y |
| `ZIP` | `ZIP` | placeholder | Y |
| `ZIP2` | `ZIP2` | placeholder | Y |
| `ZIPF` | `ZIPF` | placeholder | Y |
| `ZIPIG` | `ZIPIG` | experimental | Y |
| `ZISICHEL` | `ZISICHEL` | experimental | Y |
