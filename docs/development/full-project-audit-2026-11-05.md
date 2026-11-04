# OmniLSS 全面项目审核报告（2026-11-05）

> 审核对象：`main` 分支 HEAD（`6ba4610` 之后的本次会话工作树）、`R/` vendored R 源码、
> `omnilss/`、`omnilss-pro/`、`omnilss-server/`、`benchmarks/`、`tools/`、`docs/`、
> `ondil/`、`gamlss-python/`。
> 审核方法：以**可从源码重生成的权威工件**为准（而非历史文档的口头结论），逐层核对
> Surface / Kernel / Parity / Admission 四层证据，并用活 R、运行时注册表、门禁脚本复算。
> 本报告是**结论与证据台账**；后续执行计划见根目录 `DEVELOPMENT_PLAN.md` 第二十二节。

---

## 0. 结论摘要（直接回答审核问题）

**问题：是否所有 R 版本的 gamlss 功能都已经实现？—— 答案：否，远未全部实现。**

必须区分三个不同层次，历史文档经常把它们混为一谈：

| 层次 | 现状 | 判据 |
|---|---|---|
| **分布族数值内核（gamlss.dist）** | 覆盖广、parity 扎实，但形态是「实验级」 | 146 个注册族、d/p/q/r 全接线、placeholder 0；三闸扫描 0 fail；但 `implemented` 仅 1 族、`experimental` 125 族、production_safe 仅 1 族（NO） |
| **gamlss 核心 API / 拟合 / 诊断 / 预测** | 主路径基本可用，细节大面积残缺 | `gamlss()` 主参数与 R 对齐（含 D2.131–D2.134 的 `*.start/*.fix/start.from/contrasts/n.cyc/trace`）；但 152 个核心导出仅 82 个有 Python surface，平滑项 62 个符号仅约 8 个实现 |
| **R 周边包 / 生态 / 框架系统** | 基本空白 | 8 个周边 R 包（add/cens/foreach/mx/prepdata/spatial/tr/ggplots）surface 命中 0–1；`gamlss.family`/link 构造体系、gen.Family、在线/流式估计、多元分布全部缺席 |

一句话定性（与 `docs/development/overall-development-plan-2026-09-24.md` 第 13 节一致）：
**问题不是「功能太少」，而是「公开表面、历史文档与真实 validated 证据之间存在明显距离」**。
OmniLSS 目前是一个「统计内核覆盖面很宽、但可声明为生产可用/与 R 等价的范围极窄」的 Beta 实现。

### 0.1 三个必须钉住的量化结论

1. **R 导出符号覆盖**：12 个 R 包共 1080 个导出符号，范围内 1036 个；Python 有 surface 689 个，**missing 391 个**（范围内 349 + `gamlss2` 42）。`docs/development/r-symbol-coverage.json`，provenance 经 `--check` 验证为最新。
2. **证据成熟度**：四层里 `parity=unverified` 425 个、`admission=blocked` 526 个；`golden:validated` 仅 34 个、`admission=production_safe` 仅 5 个。
3. **运行时注册表**：146 族中 `implemented` 1、`experimental` 125、`compat_alias` 20、`placeholder` 0；feature 层面 `rs_fit`/`prediction` 仅 2 族 validated，`r_consistency` 39 族 validated，`rs_jax_fit`/`ad_hessian` 各 140 族 unsupported。

> 口径警示：`146 个注册族` ≠ `146 个 R family 已迁移`。compat_alias 是复用别的数值核的兼容别名，
> 不计独立 R 行为；`experimental` 明确「不足以进入生产路由」；`implemented` 只表示达到 production-safe。

---

## 1. 审核范围、环境与证据链

### 1.1 环境（本次实测，非引用历史文档）

| 项 | 实测值 | 说明 |
|---|---|---|
| `.venv` | Python 3.10.16 / JAX 0.6.2 (CPU) / NumPy 2.2.6 / SciPy 1.15.3 | 项目运行环境，`import omnilss` 成功，版本 `0.4.0` |
| `debianjaxgpu` | WSL Debian 的 JAX GPU 环境 | gitignored，**非本项目源码，本次未清理未修改** |
| R | R 4.6.0 (2026-04-24) + gamlss 5.5.0 + gamlss.dist 6.1.1 + jsonlite | 可跑 live-R parity |
| `R/` vendored | gamlss 5.5-3 / gamlss.dist 6.1-8 / 12 个包目录 / 1354 个文件 | 权威 R 来源；覆盖生成器解析 12 个 NAMESPACE |
| `ondil` | 0.4.8 | 参考项目（在线估计），gitignored |
| `gamlss-python` | 0.2.0（pyproject）/ 0.1.0（`__init__.__version__`，版本自相矛盾） | 参考项目（NumPy/SciPy 转录），gitignored |

受保护环境 `.venv`、`debianjaxgpu`、`R/`、`ondil/`、`gamlss-python/` 全部保持完整（清理后逐一核对文件数）。

### 1.2 证据来源与可信度分级

本次审核**只采用可复算的一手证据**，并在括号标注复算命令：

- **A 级（机器生成、可重跑、带 provenance）**
  - `docs/development/r-symbol-coverage.json` / `.md`：`tools/generate_symbol_coverage.py --check` → `symbol coverage artifacts are up to date`。
  - `docs/development/r-export-coverage.json` / `.md`：`tools/generate_r_export_coverage.py`（registry inventory，生成时间 2026-11-04）。
  - `docs/development/symbol-coverage-diff.json`：`tools/coverage_diff_report.py` → `result: OK`，`ratchet_ok`/`ceiling_ok` 均 true，无 regressions。
  - `omnilss/src/omnilss/family_capabilities.py` 运行时 `capability_matrix()`：本次直接导入复算 146 族状态。
- **B 级（活 R / 门禁 / 测试）**
  - 三闸全族扫描结论见 `HANDOFF.md` 第 1 节（D2.129，146 族）：`deriv 145/146 pass (skip 1 GT)`、`dpqr 121/146 pass (skip 25)`、`rsdev 82/146 pass, skip 28, warn 36, 0 fail`；`rqres 146/146`、`has_dqr 146/146`、`placeholder 0`。
  - 全库测试：`DEVELOPMENT_PLAN.md` 第二十/二十一节（2026-11-04）：75 条原始失败 → 0；8 分片 495 文件跑通；14 文件串行回归 `exit=0`。本次 `find omnilss/tests -name 'test_*.py'` = **497 个测试文件**。
  - golden manifest：9 个 case、26 个 golden JSON；`response_domain._RESPONSE_DOMAINS` = 82 条（活 R dump 生成）。
- **C 级（历史文档，仅作背景，结论一律以 A/B 为准）**
  - `docs/migration-progress.md`（2026-05-23，基于 omnilss 0.3.0）声称「总体 63%」——**已过时**，其逐符号状态与当前 146 族、0 placeholder 不符，不可作为当前完成率。
  - `docs/development/overall-development-plan-2026-09-24.md` 的 P0 清单：本次抽查过半已解决（见 §6.3）。

> 关键纪律（来自项目长期记忆）：「活 R 对照是唯一可信 parity 证据」；「`skip` ≠ `pass`」；
> 「`gate 绿` ≠ 族健康」；「注册数量 ≠ 功能完成」。本报告所有完成率均指**证据层级**，非注册数量。

---

## 2. 权威基线数据（本次复算）

### 2.1 R → Python 逐符号覆盖（`r-symbol-coverage.json`）

| 指标 | 数值 |
|---|---|
| R 包解析数 | 12（范围内 11，`gamlss2` 明确 out-of-scope） |
| R 导出符号总数 | 1080（范围内 1036） |
| Python surface matched | 689 |
| **surface missing** | **391**（范围内 349 + gamlss2 42） |
| family-linked 符号 | 531（其中 dpqr 424） |
| Python 独有公开导出 | 300 |

**四层证据分布（1080 个符号）**

| 层 | 分布 |
|---|---|
| surface | missing 391 / module_def 226 / module_export 96 / public_api 191 / registry 176 |
| kernel | implemented 163 / experimental 525 / compat_alias 1 / missing 391 |
| parity | capability:unsupported 249 / capability:validated 107 / golden:validated 34 / r_consistency_test 172 / tested 93 / **unverified 425** |
| admission | **blocked 526** / exported_api 85 / internal 73 / not_available 391 / **production_safe 5** |

### 2.2 逐 R 包 surface 覆盖（`per_package`）

| R 包 | 版本 | 导出数 | surface matched | 缺口性质 |
|---|---|---:|---:|---|
| `gamlss.dist` | 6.1-8 | 654 | **604** | 主体已覆盖；50 缺（绘图/均值辅助、gen.Family、内部 C 辅助、骨架族） |
| `gamlss` | 5.5-3 | 152 | **82** | 核心 API；70 缺（见 §3.3），主要是平滑项与 S3 方法 |
| `gamlss.ggplots` | 2.1-34 | 109 | 1 | **几乎空白**（ggplot2 可视化层，优先级最低） |
| `gamlss.prepdata` | 0.1.19 | 59 | 0 | **空白**（数据预处理层） |
| `gamlss.add` | 5.1-14 | 15 | 0 | **空白**（加性项：GA/BA/NN/FK/TR） |
| `gamlss.foreach` | 1.1-10 | 13 | 0 | **空白**（并行/批量拟合） |
| `gamlss.mx` | 6.0-4 | 12 | 0 | **空白**（有限混合） |
| `gamlss.spatial` | 3.0-3 | 9 | 0 | **空白** |
| `gamlss.tr` | 5.1-10 | 8 | 0 | **空白**（截断变换） |
| `gamlss.cens` | 5.0-8 | 5 | 0 | **空白**（删失） |
| `gamlss.data` | 6.0-9 | 0 | 0 | 纯数据包，无导出函数 |
| `gamlss2` | 0.1-0 | 44 | 2 | **明确 out-of-scope**（R 新一代重写，见 §3.6） |

范围内 missing 349 的构成：ggplots 108 + gamlss 70 + prepdata 59 + dist 50 + add 15 + foreach 13 + mx 12 + spatial 9 + tr 8 + cens 5。

### 2.3 运行时能力矩阵（直接 `import` 复算）

| 状态 | 族数 | 含义 |
|---|---:|---|
| `implemented` | 1 | 仅 `NO` |
| `experimental` | 125 | 可注册可拟合，但证据不足以进生产路由 |
| `compat_alias` | 20 | 兼容别名，复用别的数值核，不计独立迁移 |
| `placeholder` | 0 | 占位核已清零（这是真实进展） |
| production_safe | 1 | 仅 `NO` |

feature validated 计数：`rs_fit` 2、`prediction` 2、`r_consistency` 39、`sampling`/`smooth_terms`/`cg_fit` 0；
`rs_jax_fit` 与 `ad_hessian` 各 140 族 `unsupported`（仅 NO/GA/PO/BI/WEI/TF 六族挂 experimental）。

### 2.4 覆盖 ratchet（`symbol-coverage-diff.json`）

`r_exports_total 1080`、`missing 391`、`placeholder 0`、`compat_alias 1`、`evidence_gap 34`、`extra 300`；
`ok / ratchet_ok / ceiling_ok = true`，`regressions = []`、`improvements = []`——即相对基线**无符号级变化**。
`tools/coverage_diff_report.py` 输出 `result: OK`。

---

## 3. 与 R 的功能差距（逐子系统）

### 3.1 分布族数值内核：覆盖面是真资产，但「实验级」不等于「完成」

**已实现且证据较强**：146 个注册族全部有 d/p/q/r 接线（`has_dqr 146/146`、`rqres 146/146`），
`build_ad_family` 的 AD 分数/Hessian 体系、zero-variant p/q/r 通道、response-domain 硬闸
（82 条 `y.valid` 表 + 活 R golden）、三闸 parity 扫描（deriv/dpqr/rsdev）145/146、121/146、82/146 pass
且 0 fail。这是全项目最扎实的部分，D2.18–D2.166 的长链条工作已把「对齐 R」这条线收口到 0 fail。

**但必须诚实标注的边界**：
- 0 fail ≠ 0 缺口：`dpqr` 有 25 个 skip（R 侧无同名函数：BINOM_/CONTR_/COUNT_ 骨架族、BCCGUNTR/BCTUNTR、MULTIN、ZILG 等）；`rsdev` 有 36 个 warn（R 未收敛/发散/sigma 地板/早停，逐条有归属）；`deriv` 1 skip（GT，探针 >180s）。
- `gate 绿` ≠ 族健康：记忆中已登记「全族扫描样本 y≤7 掩盖初值缺陷，只有 n=100/y≤15 的 consistency 测试暴露」。
- 绝大多数族是 `experimental`：`strict_capabilities` 下生产路由 fail-closed 会拒绝它们。
- 已知挂账缺陷：BCTO deriv、BCT tau 交叉块 AD 空白（`betainc` 对形状参数无梯度）、SHASH（非 o）密度积分缺口、NET 的 `tau=max(tau,nu)` clamp 与 R `dNET` 的 stop 未对齐、PARETO1o 幽灵项、JSU 拟合轨迹、LNO intercept-only 既有失败、ZISICHEL 在 R 侧自身失败导致无参照。

### 3.2 `gamlss()` 用户面：主路径已对齐，边角仍缺

D2.130 审计曾列 4 个缺口，D2.131–D2.134 已交付（本次在 `fitting.py:1114` 的签名中逐项核实）：
`mu_start/sigma_start/nu_start/tau_start`、`mu_fix/.../tau_fix`、`start_from`、`contrasts`、`n_cyc`、`trace` 均已落地，
且有 `omnilss/tests/fixtures/r_golden/*_2026-11-01.json` 活 R 对照。

仍缺（来自 `gamlss-user-surface-gap-audit` 与 `HANDOFF.md` 1.1）：
- `*.fix` / `contrasts` **只接 `method='RS'`**；分类协变量的预测重建、RS 以外算法（CG/MIXED/joint）路径待做（D2.136）。
- `predict.gamlss` 的 `type="terms"` / `se.fit` 细节（Phase 3 契约）。
- `gamlssML` / `gamlssCG` 独立入口名、R `...` 中的部分控制项。

### 3.3 R 核心包 70 个缺失符号：主要是「平滑项 + 框架系统」

`gamlss` 包 152 导出仅 82 命中；61 个缺失函数样本：
`bfp, bp, calibration, cs.control, cy, cy.control, edf, fp, gamlss.cs/cy/fp/lo/pb/pbc/pbm/pbo/pbp/pbz/pcat/pp/ps/pvc/random/re, ...`
归类：
- **平滑项（最大缺口）**：R 约 62 个 smoother 符号，OmniLSS 仅 `pb`/`ps`/`cs` 有数值实现，
  另有 tps/tensor/gcv/reml/ubre 等 Python 扩展；`pbm/pbz/pbo/pbp/pbc/pvc/cy/fp/bfp/re/ri/pp/polyS/pcat/scs/lo(真 LOESS)/random(真 nlme 后端)` 全部缺席。
  `docs/migration-progress.md` 给平滑项完成率 13%（该文档过时，但方向一致）。
- **gamlss.family / link 框架体系**：`as.family` / `gamlss.family` / `as.gamlss.family` / `checklink` /
  `show.link` / `make.link.gamlss` / `gen.Family` / `Family.*` 全无。这直接影响「用户自定义族」能力。
- **S3 方法与绘图辅助**：`plot.*`/`coef.*`/`print.*`/`fitted.*` 等、`rqres.plot`、`bp` 等（优先级低）。
- **模型选择变体**：`stepGAIC.VR` / `stepGAIC.CH` / `gamlss.scope` / `stepTGDAll.A`。

### 3.4 周边 R 包：8 个包几乎空白（P4 整块未启动）

`gamlss.add`（有限混合/神经网络加性项）、`gamlss.cens`（删失）、`gamlss.tr`（截断）、
`gamlss.mx`（混合模型）、`gamlss.spatial`、`gamlss.foreach`（并行）、`gamlss.prepdata`、
`gamlss.ggplots` 的 surface 命中为 0–1。删失/截断/混合是 GAMLSS 的重要建模能力，属于功能完整性缺口，
不是「可不做的可选项」。

### 3.5 `gamlss2`：明确 out-of-scope（设计决策，非缺陷）

`r-symbol-coverage.json` 的 `scope.out_of_scope_packages = ["gamlss2"]`，
`full-gamlss-migration-audit-and-roadmap-2026-09-11.md` 第 5 节 P4-D5 明确「gamlss2 单独规格/版本/证据，
不阻塞 1.x/1.0」。44 个导出、42 个函数、77 个 S3 方法。这是**有意的范围裁剪**，报告与计划须持续标注，
不能被误读为「漏做」。

### 3.6 不在 R 范围内、但 OmniLSS 已扩展的能力（正向差异）

JAX RS/AD Hessian、RS_JAX 批量、CG IRLS-cross、joint/Adam/L-BFGS 优化器、deep GAMLSS（flax/optax）、
sklearn `GAMLSSRegressor`、CRPS/log_score/DSS/interval_score/PIT 概率评分、模型序列化（JSON/pickle）、
非参/参数/残差 bootstrap、ridge/lasso/elastic-net 正则化、tps/tensor/GCV/REML/UBRE 平滑扩展、
REST(FastAPI)/gRPC 服务层、`omnilss-pro` / `omnilss-server` 商业原型、automl 等。
这些是 `python_only_public_exports = 300` 的主要来源，属产品化增量，不计入 R 迁移完成率。

---

## 4. 与参考项目对照

> 边界（沿用 `overall-development-plan` 与 `full-gamlss-migration-audit` 的裁定）：
> `gamlss-python` 是 NumPy/SciPy 转录 oracle，`ondil` 是在线估计设计 oracle；
> 二者都**不能替代 R parity 或 JAX kernel**，也**不改变 R 优先级**。

### 4.1 `gamlss-python` 0.2.0

- 规模：34 个 `dist/*.py` 分布族 + `engine.py`（RS/CG/mixed）+ `formula.py`（R 风格公式）+
  `family.py` + `links.py` + `methods.py` + `model.py` + `rqres.py` + `smooth.py`（pb/pbz，含 ML 平滑参数循环）。
  `data/` 带 6 个数据集。
- **OmniLSS 明显更宽**：146 族 vs 34 族；且带 JAX 可微/GPU/服务/评分/bootstrap。
- **但 OmniLSS 有 gamlss-python 已具备、自己却缺失的能力**：
  - `as_gamlss_family` / `checklink` / `make_link_gamlss` —— 即 §3.3 的 family/link 构造体系；
  - 顶层 `predictAll`（OmniLSS 的 `predict_all` 存在但未在顶层导出，见 `migration-progress` 1.2）；
  - patsy 风格的完整 R 公式契约（OmniLSS 公式解析器自行实现，部分函数如 `factor` 才在 D2.134 补齐）。
- 结论：gamlss-python 可作「公式/结果对象槽位/参考转录」的 oracle，但功能面是 OmniLSS 的子集。

### 4.2 `ondil` 0.4.8

- 规模：26 个分布（含 **6 个多元**：`MultivariateNormalInverse{Cholesky,LowRank,ModifiedCholesky}` 与
  `MultivariateStudentTInverse{...}`）+ 4 个**在线估计器**（`online_gamlss`、`online_linear_model`、
  `online_lasso`、`online_mvdistreg`）+ coordinate descent / elastic-net / lasso / ridge 路径 +
  sklearn estimator 契约、`sample_weight`、scaler、pipeline、diagnostics、information_criteria、Gram。
- **OmniLSS 的结构性空白（本次全库 grep 核实）**：
  - 无 `partial_fit` / 在线 `update` / forget factor / 流式状态 —— **整个在线/流式估计层缺席**；
  - 无多元分布（`multivariate` 关键字命中均为注释或期刊名，非实现）；
  - `coordinate_descent` 命中仅为 `fit_lasso_coordinate_descent`（离线），非 ondil 的在线路径。
- OmniLSS 已有的对应面：离线正则化、bootstrap、sklearn 包装、评分 —— 但语义不同，不覆盖在线契约。
- 结论：ondil 在「在线/流式 + 多元 + 正则路径」上是 OmniLSS 的**能力超集**，
  对应 `full-gamlss-migration-audit` 的 P5 独立轨道，应独立验收、不混入 R 迁移完成率。

### 4.3 对照矩阵（谁更宽）

| 能力面 | OmniLSS | gamlss-python | ondil | R |
|---|---|---|---|---|
| 分布族数量 | **146** | 34 | 26 | ~120 导出族 |
| 与 R 数值 parity 证据 | **强（三闸 + 活 R golden）** | 弱（转录） | 无（非目标） | 基准 |
| 拟合算法 | RS/CG/MIXED/joint/LBFGS/RS_JAX | RS/CG/mixed | 在线/CD | RS/CG/mixed |
| JAX 可微 / GPU | **有** | 无 | 无 | 无 |
| 平滑项 | pb/ps/cs + 扩展 | pb/pbz | 无 | ~15 类 |
| family/link 构造体系 | **缺** | 有 | 部分 | 有 |
| 在线/流式估计 | **缺** | 无 | **有** | 无 |
| 多元分布 | **缺** | 无 | **有** | MN3/4/5/MULTIN |
| 周边包（删失/截断/混合/空间/并行/绘图/预处理） | **缺** | 无 | 无 | 有 |

---

## 5. 已实现能力盘点（正面清单，避免只谈缺口）

- 拟合：`gamlss()`（RS/CG/MIXED/joint/LBFGS/RS_JAX/auto）、`gamlssML`、`gamlssVGD`、update/refit、
  `gamlss_control`/`glim_control`、逐观测初值（71 族共享 R 表达式）、绝对收敛判据、非有限 deviance 分类。
- 分布：146 族 d/p/q/r、AD 分数/Hessian、zero-variant 与 o-参数化别名族、response-domain 硬闸、
  Bessel JVP、`_lbeta` 等数值精度修正。
- 预测/诊断/选择：`predict`/`predictAll`/`lpred`/`fv`/`prodist`、rqres、wp、Q.stats、acfResid、
  hatvalues、vcov/rvcov、confint、logLik、summary/print、GAIC/stepGAIC/stepTGD/fitDist/chooseDist、
  centiles/quantSheets/lms、CRPS 等评分。
- 平滑/正则：`pb`/`ps`/`cs`、tps/tensor、GCV/REML/UBRE、ridge/lasso/elastic-net、bootstrap。
- 工程：capability fail-closed 门禁、能力矩阵 API、序列化、REST/gRPC、golden manifest gate、
  stage0/stage completion gate、tiered collection（4 tier）、coverage ratchet、symbol coverage 生成器、
  全族三闸扫描器、497 个测试文件、75→0 失败收口。

---

## 6. 缺陷与技术债登记（来自长期记忆 + 本次核实，均为当前未解决项）

### 6.1 数值 / 内核
1. `bessel.py` 的 `log_bessel_kv` 走 `jax.pure_callback` 调 scipy（主机往返、切断 jit 融合）；
   Sichel 三族拟合 ~172s/周期。需 JAX 原生实现 + 双验收（对数网格 1e-12 相对误差 + 周期耗时下降）。
2. 逐族独立播种 + 一次性重登记证据：D2.47 的 Bessel JVP 修复改变了拟合轨迹，
   `ZIPIG`/`ZINBI` 的 `_RSDEV_R_STOPPED_EARLY` 登记数字待全族流重立；改播种会使所有登记数字整体失效。
3. 具体族缺口：BCTO deriv、BCT tau 交叉块（JAX `betainc` 对形状参数无梯度，已登记为可解释 xfail）、
   SHASH 密度积分、NET clamp、PARETO1o 幽灵 score、JSU 拟合轨迹、LNO intercept-only 失败、
   ZISICHEL 无 R 参照、EXGAUS q 的 uniroot 容差残差。

### 6.2 功能 / API
4. `*.fix`/`contrasts` 仅 RS；分类协变量预测重建、RS 外算法待做（D2.136）。
5. family/link 构造体系、gen.Family、顶层 `predictAll`、约 50 个平滑项、8 个周边 R 包（§3.3/§3.4）。
6. 在线/流式估计、多元分布（ondil 对照，§4.2）。
7. `MIXED` 语义：`fitting.py:658` 记为「75% step (damped RS, experimental)」，
   与 R 的 `mixed(n1,n2)=RS(n1)→CG(n2)` **不同名同义**（`overall-development-plan` P0-03 仍开放）。

### 6.3 P0 阻断项复核（2026-09-24 清单的当前状态）

| 编号 | 原问题 | 本次抽查结论 |
|---|---|---|
| P0-01 | save/load 格式不配对 | **已解决**：`serialization.py:952-953` 均为 `*_json` |
| P0-02 | 工件 0.3.0 / 包 0.4.0 漂移 | **基本解决**：`_version.py` 0.4.0，schema v2 常量独立 |
| P0-03 | MIXED ≠ R mixed | **仍开放**（见 §6.2.7） |
| P0-06 | `httpx2` 疑似错误依赖 | **未复现**（全库 grep 无命中） |
| P0-07 | CI 引用不存在的 `test_smooth_outer_loop.py` | **未复现**（`.github` 无命中） |
| P0-08 | devcontainer 引用 `.codex/setup.sh` | **仍开放**：`.devcontainer/*.json` 仍引用，文件确实缺失 |
| P0-09 | capability 依赖运行时 AST | **部分解决**：已有可重生成工件，但状态仍由 AST 派生 |
| P0-10 | 历史文档宣称「八核心族/三算法与 R 等价」 | **仍开放**：`docs/migration-progress.md` 等旧文档仍含 63%/等价措辞，需校准 |

### 6.4 工程 / 证据债
- 全库 497 文件 >1.5h，必须 8 分片并行后台跑；4 分片会被回收（记忆明确登记）。
- 改源码/测试必须同 commit 再生覆盖工件（`generate_coverage_artifacts.py --date` + `coverage_diff_report.py → OK`），
  否则 `test_symbol_coverage_tool` 的 provenance 假失败。
- CI 目前只有测试，无 lint/type 步骤；Phase 8 的用户文档/教程、method 名一致性核对未完成。
- `docs/migration-progress.md`（2026-05-23）、`gamlss-core-migration-audit-2026-05-22.json` 等旧文档数字过时，
  与当前 146/125/20/1、0 placeholder 冲突，**不得再作为完成率引用**。

---

## 7. 本次会话的清理与修复（可提交）

| 动作 | 范围 | 影响 |
|---|---|---|
| 清理 `tmp/` | 删除 3536 个文件 / 20.4 MB（探针脚本、分片日志、扫描 JSON、一次性 patch 脚本） | gitignored，不入库 |
| **保留** `tmp/` 5 个承重 R 参照文件 | `bi_dbi_fixture_r.json`、`bi_dbi_ref.R`、`d223_c_ref.csv`、`d223_full_ref.csv`、`d223_ref.R` | 这些被 `test_bi_dbi_binomial_parity.py` / `test_dbi_double_binomial.py` 运行时读取；删除会把 live-R parity 测试静默变成 `skip` |
| 清理 `omnilss/tmp/` | 删除 483 个文件 / 2.2 MB | gitignored，不入库 |
| 清理缓存 | 1256 个 `.hypothesis`/`rbus/temp`/stray log 文件、449 个 `__pycache__` 目录 | gitignored，可再生成 |
| 删除 stray 产物 | `git rm Rplots.pdf`（被误跟踪的 R 绘图输出）；`tmp_batch.log`、`omnilss/tmp_*.log` 9 个 | 入库删除 |
| `.gitignore` | 新增 `Rplots.pdf` 规则，防止 R 测试再次生成并被误提交 | 入库 |
| 修复文档损坏 | `DEVELOPMENT_PLAN.md` 第 1 行 `[object Object]` → 正确标题 | 该损坏由 `e75f69d`（D2.150）引入，历史标题为「# OmniLSS 全项目开发计划（总纲）」；入库 |
| 新增本报告 | `docs/development/full-project-audit-2026-11-05.md` | 入库 |
| 新增后续计划 | `DEVELOPMENT_PLAN.md` 第二十二节 | 入库 |
| 覆盖工件 | **本次不改动源码/测试**，provenance 经 `--check`（up to date）与 `coverage_diff_report.py`（OK）复核，按纪律无需再生 | 避免无意义的时间戳 churn |

受保护环境 `.venv`/`debianjaxgpu`/`R`/`ondil`/`gamlss-python` 清理后逐一核对完好（13013 / 7134 / 1354 / 290 / 124 文件）。

---

## 8. 审核验证（本次实际运行的命令）

```
.venv/Scripts/python.exe --version                          # 3.10.16
.venv/Scripts/python.exe -c "import jax,numpy,scipy,omnilss"  # 0.6.2 / 2.2.6 / 1.15.3 / 0.4.0
Rscript --version                                           # R 4.6.0
.venv/Scripts/python.exe -c "<capability_matrix 复算>"        # 146 族状态
.venv/Scripts/python.exe tools/generate_symbol_coverage.py --check  # up to date
.venv/Scripts/python.exe tools/coverage_diff_report.py        # result: OK
```

未运行全库 497 文件测试（>1.5h，需 8 分片后台）；本次结论引用 2026-11-04 已归档的
8 分片全量复测 + 14 文件串行回归 `exit=0`（`DEVELOPMENT_PLAN.md` 第二十/二十一节），
并按项目纪律注明「未在本会话重新跑全量」。

---

## 9. 结论与优先级建议

1. **对外声明必须收敛**：不能说「已实现 R gamlss」。可说的是「146 个分布族数值内核 + RS/CG 拟合
   对活 R 达到 0 fail 的三闸 parity（实验级）；生产可用范围仅 NO 一族」。
2. **最高价值不是「加功能」而是「升证据」**：把高频族从 `experimental` 升到 `validated`/production_safe，
   补齐 425 个 `unverified` 与 526 个 `blocked` 的证据链；这比再扩 surface 更接近 1.0。
3. **结构性缺口按用户价值排序**：平滑项/随机效应 → family/link 框架 → 删失/截断/混合周边包 →
   在线/多元（ondil 轨道，独立发布）。ggplots 外观层优先级最低。
4. **性能项单独一轮**：JAX 原生 Bessel（Sichel 性能）+ GPU（`debianjaxgpu`）验证，必须在一致性 gate 之后。
5. **工程止血**：修 `.codex/setup.sh`（P0-08）、校准旧文档的 63%/等价措辞（P0-10）、
   给 CI 加 lint/type、明确 `gamlss2` out-of-scope 的持续标注。

完整的分阶段 WBS、验收标准、依赖与里程碑见 `DEVELOPMENT_PLAN.md` 第二十二节。
