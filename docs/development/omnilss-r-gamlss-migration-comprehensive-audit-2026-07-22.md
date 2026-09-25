# OmniLSS：R `gamlss` 生态到 JAX/Python 迁移全面审计

- **审计日期**：2026-07-22
- **审计对象**：`omnilss/`、`R/` 下所有附带项目、迁移文档、能力矩阵、测试与基准
- **Git 基线**：`ccad68e`（`2026-06-05`）
- **审计工作树**：存在与 `ps`/`cs` 平滑器有关的未提交改动；本报告记录的是**包含这些改动的当前工作树**，且没有改动它们。
- **结论等级**：`已证实` 表示由当前代码或本地执行结果直接支持；`高风险` 表示实现/证据不足而不能用于 R 等价或发布声明；`待验证` 表示尚无足够实验结论。

## 执行摘要

OmniLSS 已经形成了可观的 Python/JAX API、分布注册表、RS/CG/Mixed/JAX/联合优化入口和大量测试文件，但它**尚未达到“完成 R `gamlss` / `gamlss.dist` 迁移”或“广泛 R 数值等价”状态**。当前最关键的问题不是 API 数量，而是实现真相、数值证据和发布口径尚未一致。

本次审计的直接证据如下：

| 维度 | 当前结果 | 审计判断 |
|---|---:|---|
| 已注册分布 | 142 | 名称可解析，不等于统计实现或 R 等价 |
| `implementation_status=implemented` | 1（`NO`） | 能力矩阵本身仅承认一个具备最强证据的实现 |
| `experimental` / `compat_alias` / `placeholder` | 55 / 29 / 57 | 141/142（99.3%）的已注册名称不是可按“完成迁移”计数的实现 |
| 标记为有 R 一致性证据的族 | 31 | 是证据清单，不代表所有 d/p/q/r、拟合、预测和边界场景均通过 |
| 注册族解析/实例化 | 142/142 成功 | 仅证明导入与工厂可调用 |
| 强制 live-R 快速矩阵 | 14/42 通过（33.3%） | 当前最重要的数值发布门禁失败 |
| d/p/q 子集 | 0/27 通过 | 既有小数值误差，也有显著参数化/量化函数/测试夹具问题 |
| 拟合子集 | 11/12 通过 | `PO` + `RS` 偏差显著，不能据此宣称核心拟合已完成 |
| 平滑子集 | 0/3 通过 | `pb`、`ps`、`cs` 都未达到当前严格 R 对照阈值 |
| 关键 pytest 集合 | 84 通过、3 失败 | GPU/自动路由契约与实现策略不一致 |
| 附带 R 扩展项目 | 11 个无 Python/文档映射 | 全项目迁移范围尚未被立项、拆解或验收 |

**推荐的总体决策**：将“完整迁移”拆成三个明确定义的产品范围。当前只把 `gamlss` + `gamlss.dist` 的经过 R 验证的核心子集作为 Release 1 目标；将 `gamlss.add`、`gamlss.cens`、`gamlss.tr`、`gamlss2` 等纳入后续独立扩展路线；将数据包与可视化包标记为数据/展示互操作工作，而非静默计入核心统计迁移完成度。

---

## 1. 审计范围、环境与方法

### 1.1 源范围

| 分层 | R 项目 | 审计发现 | 建议定位 |
|---|---|---|---|
| 核心统计引擎 | `gamlss` 5.5-3 | 主拟合、控制、预测、诊断、选择、LMS、平滑项与 S3 方法 | **Release 1 主范围** |
| 分布与族协议 | `gamlss.dist` 6.1-8 | 大量族工厂、d/p/q/r、`gamlss.family`、`gen.Family`、`distributions3` | **Release 1 主范围** |
| 额外项 | `gamlss.add` 5.1-14 | `ga`/`ba`、神经网络、树、自由/固定节点、`tr` | Release 2，独立 capability |
| 区间删失 | `gamlss.cens` 5.0-8 | `gen.cens`、`cens.*`、`cens` | Release 2，独立族变换框架 |
| 截断分布 | `gamlss.tr` 5.1-10 | `gen.trun`、`trun.*`、`fitTail*` | Release 2，先实现通用截断核 |
| 混合/非参数 | `gamlss.mx` | `gamlssMX`、`gamlssNP`、密度/CDF 辅助 | Release 3，非核心 |
| 空间模型 | `gamlss.spatial` | MRF/GMRF、邻接与精度矩阵 | Release 3，依赖图/稀疏线性代数设计 |
| 批量与降维 | `gamlss.foreach` | Bootstrap、rolling、PCR、QR | Release 3；可复用现有 Python 生态但需 API 设计 |
| 数据准备 | `gamlss.prepdata` | 大量数据质量、切分、缩放、K-fold、相关性工具 | 作为独立 Python utilities 范围评估 |
| ggplot 展示 | `gamlss.ggplots` | 大量残差、分位数、密度与互动图 API | 可视化适配层，不应阻塞统计核心 |
| 新一代框架 | `gamlss2` 0.1-0 | 新公式/族/选模、`special_*`、MCMC、bamlss2 等 | **独立大项目**，不得混入 classic `gamlss` 完成率 |
| 数据包 | `gamlss.data`、`WeatherGermany` 1.0-0 | 数据集；`WeatherGermany` 没有 `NAMESPACE`，包含数据与空间数据依赖 | 数据分发/加载互操作，不是统计内核迁移 |

针对 `omnilss/` 与 `docs/` 的名称检索没有找到对上述 11 个扩展/数据项目的专门映射。该结果不表示 Python 中绝无相似功能，而是表示**没有可审计的“R 符号 → Python 入口 → 数值/行为证据”追踪链**。因此，它们不能计入当前迁移完成率。

### 1.2 实际验证环境

| 项目 | 结果 |
|---|---|
| Python | `.venv` 中 Python 3.10.16 |
| JAX | 0.6.2 |
| NumPy / SciPy | 2.2.6 / 1.15.3 |
| 默认 JAX x64 | **关闭** (`jax_enable_x64=False`) |
| live R | R 4.6.0 |
| 已安装 R 对照包 | `gamlss` 5.5.0、`gamlss.dist` 6.1.1、`jsonlite` 2.0.0 |
| 仓库 R 源声明 | `gamlss` 5.5-3、`gamlss.dist` 6.1-8 |

存在两类基线漂移风险：

1. `omnilss/pyproject.toml` 仅使用 `jax>=0.4.20`、`numpy>=1.24.0` 等开放下界，未锁定运行时解析结果；数值软件不能只依赖此类范围保证复现性。
2. 当前安装的 R 包版本与 `R/` 中 vendored 源的 `DESCRIPTION` 版本不相同。现有基准比较的是已安装 R 包，而不是明确从 `R/gamlss`、`R/gamlss.dist` 安装的固定源码。

`benchmarks/comprehensive_r_consistency_test.py` 会在导入 JAX 前设置 `JAX_ENABLE_X64=true`，故本报告的 R 对照基准以 x64 运行；普通库调用和部分测试不会自动获得该设置。对分布尾部、分位数、Hessian 和收敛门槛而言，这一差异不可忽略。

### 1.3 证据规则（后续报告必须采用）

不要再以“模块存在”“名称可注册”或“Python-only 测试通过”代表迁移完成。每个 R 导出符号、分布或平滑项都必须同时记录：

| 层级 | 定义 | 可用于的声明 |
|---|---|---|
| E0：缺失/占位 | 无入口，或返回 `NaN`/正态近似/其他族的核 | 不可用；仅可列为迁移债务 |
| E1：接口可用 | 能导入、解析、实例化，基础契约测试通过 | API/smoke 可用，不可称 R 等价 |
| E2：独立实现 | 有专属数学核、参数化、支持域、导数/优化路径 | 实验性功能，可由显式 opt-in 使用 |
| E3：R 数值验证 | 固定 R 基线下，d/p/q/r、拟合或数据型 API 的合约通过 | 该已验证功能与 R 对齐 |
| E4：发布验证 | E3 + 边界/随机性质/预测/序列化/性能回归 + 可追溯工件 | 可纳入生产/发布承诺 |

状态必须由生成器产出，并绑定：R 源版本、R 运行包版本、Python/JAX 版本、数据种子、公式、参数、容差、测试文件和结果工件路径。

---

## 2. 覆盖统计的真实含义

### 2.1 现有文档不能直接作为完成率

仓库中存在三组不完全一致的数字：

- [`docs/migration-progress.md`](../migration-progress.md) 将大量入口记为已实现，并给出约 63% 的综合完成率。
- [`docs/development/r-gamlss-coverage-and-plan-2026-05-22.md`](r-gamlss-coverage-and-plan-2026-05-22.md) 已正确指出兼容别名、泛化 p/q/r 占位和注册表不能当作统计迁移，并采用更保守规则。
- 运行时 [`family_capabilities.py`](../../omnilss/src/omnilss/family_capabilities.py) 当前只把 `NO` 标为 `implemented`；其他 141 个已注册族均为 `experimental`、`compat_alias` 或 `placeholder`。

因此，后续仪表盘至少应同时展示下列四个数，而不能只展示一个“完成率”：

1. R 导出总数与逐符号状态；
2. Python 可解析/可调用总数；
3. 专属数学实现总数；
4. E3/E4 验证通过总数。

### 2.2 分布注册表审计

实际执行 `list_families()`、`list_family_capabilities()` 与 `resolve()` 得到：

```text
registered = 142
implementation_status = {
  compat_alias: 29,
  experimental: 55,
  implemented: 1,
  placeholder: 57
}
r_consistency_validated = 31
resolve_failures = 0
```

这说明当前 registry 有良好的可解析性，但不能作为正确性结论。特别是：

- `ST1`–`ST4`、`DBI`、`DBURR12`、`BCCGuntr`/`BCPEuntr`/`BCTuntr`、`MULTIN` 与 `gen.Family` 骨架已明确是兼容层或别名，不得计入完成迁移。
- `dpqr_functions.py` 保留了标准正态近似、固定区间积分/量化近似及 `NaN` 返回的通用回退；这类函数必须保持 `placeholder`，不能通过“函数存在”提升状态。
- 有 **27 个** factory module 名称含 `compat`，但 capability status 为 `experimental` 而非 `compat_alias`，其中包括 `BCCGo`、`BCPEo`、`BCTO`、`GP`、`ZAGA`、`ZAIG` 与多种零修正族。这未必逐个都是数学别名，但说明“按少数已知集合/模块名推断状态”的策略不完整。应改为每个注册项都有显式 `implementation_origin`、`r_symbol`、`kernel_owner` 与 `evidence` 元数据，禁止文件名推断。

### 2.3 分布 d/p/q/r 的当前数值证据

在强制 live-R 的 9 分布快速矩阵中，27 个 d/p/q 项**全部未通过**默认 `abs=1e-5`、`rel=1e-5` 阈值：

| 分组 | 结果 | 解释与处置 |
|---|---|---|
| `NO` | d/p/q 最大绝对误差约 `4.8e-5` | 可能是精度、桥接或容差设计问题；先定位，不能直接放宽全局容差 |
| `GA` | d/p 约 `5e-5`，q 误差 `3.615` | q 参数化/实现为 P0 数学缺陷 |
| `PO` | d/p 约 `4e-5`，q 误差 `1` | 离散分位数定义、边界与 R 的最小整数规则需要逐点比对 |
| `BI` | d/p 为 `NaN`，q 误差 `1` | 基准未提供/编码 `bd` 等固定数据参数的可能性很高；须先修测试夹具，再裁决内核 |
| `NBI` | p 误差 `0.2602`，q 误差 `2` | 参数化/CDF/量化逻辑为 P0 缺陷 |
| `BE` | d/p/q 绝对误差约 `5e-5`，相对误差异常大 | 在接近零密度处应采用混合绝对/相对规则并单独测试边界 |
| `ZAGA` | q 误差 `1.392` | 零质量、连续部分缩放或逆 CDF 为 P0 |
| `GU` | p 误差 `0.2643`，q 误差 `3.073` | CDF/量化实现或参数化为 P0 |

**优先原则**：不要先把全局容差调到 `1e-4`。先对每个项记录支持域、R 输入参数、Python 输入参数、逐元素输出、dtype、是否有限和单调性。只有在数学实现正确、差异稳定且可解释时，才可设置逐族容差；绝不能让 `NaN`、全常量、标量错误广播或离散 q 的整数量化差异通过。

---

## 3. 核心拟合、算法与运行时审计

### 3.1 RS / CG / Mixed / JAX 路由

`gamlss()` 提供 RS、RS_JAX、CG、Mixed、joint 和 L-BFGS 路由，并在调用前通过 `method_route_capability_report()` 拒绝 `compat_alias` 与 `placeholder` 族。这是正确的防线；后续要确保全部 API、批处理、服务端和 benchmark 使用同一入口。

本次运行以下针对性测试：

```powershell
$env:JAX_ENABLE_X64 = 'true'
.\.venv\Scripts\python.exe -m pytest `
  omnilss/tests/test_audit_gamlss_migration.py `
  omnilss/tests/test_family_capabilities.py `
  omnilss/tests/test_distribution_registry_authoritative.py `
  omnilss/tests/test_method_routing_integration.py `
  omnilss/tests/test_formula_parser_s.py `
  omnilss/tests/test_smooth_fitting.py `
  omnilss/tests/test_r_consistency_smooth.py
```

结果为 **84 passed, 3 failed**。失败均在 `test_method_routing_integration.py`：

- `test_default_rs_routes_to_jax_above_threshold`
- `test_auto_routes_to_jax_above_threshold`
- `test_rs_jax_receives_routing_decision_payload`

根因已证实：`fitting.gamlss()` 在 GPU/TPU 阈值命中后，又对较小设计矩阵（`p_total <= 4`）强制回退到 NumPy RS；测试仍期待只要超越 crossover 就转入 RS_JAX。当前文档也同时把 `method="RS"` 描述成默认 CPU 路径和可参与自动路由的入口，造成语义不清。

**决策要求**：

1. 明确 `RS` 是否始终表示“显式 NumPy RS”，而 `auto` 才能选 JAX；或明确二者都可自动路由。
2. 若保留 small-p 回退，必须将其作为稳定的、可观测的策略写入配置、文档和测试；测试数据需覆盖 `p<=4` 与 `p>4` 两侧。
3. 每个模型的 `additional_slots` 必须保存 requested method、selected method、backend、threshold、`p_total`、reason 和决策版本。
4. 在真实 GPU 环境建立族别、n、p 的 crossover 工件前，不应将 RS_JAX 作为性能承诺或默认加速路径。

### 3.2 拟合一致性

同一快速矩阵用 `NO`、`GA`、`PO`、`BI` 以及 RS/CG/Mixed 运行 12 项，得到 **11/12 通过**；唯一失败是：

```text
PO_RS_fitting
R comparison exceeded tolerance
deviance diff = 69.215
max_abs = 69.21, max_rel = 1.072
```

这说明“部分核心拟合可对齐”是积极信号，但不足以证明 RS 全面正确。至少应补充以下矩阵：

- 每个发布族至少含截距模型、线性 mu、sigma/nu/tau 公式、权重、固定参数和缺失数据策略；
- RS、CG、Mixed 必须分别报告收敛、deviance、每个参数的系数与 fitted values；
- 计数族必须覆盖零、多零、离散响应边界与极端均值；
- CG 应增加 observed information 条件数、阻尼、线搜索与 Hessian 正定修复的工件；
- 任何 weight/Hessian floor、eta clip、step-halving、NaN replacement 或 fallback solver 都必须写入模型诊断，而非静默改变统计更新。

### 3.3 数值精度与复现风险

库的默认运行环境没有启用 JAX x64，但实现多处显式请求 `jnp.float64`，因此默认调用会产生 float32 截断告警。应在 Release 1 前完成以下一项明确策略：

- **推荐**：在受控的应用初始化层明确启用 x64，并让任何在 JAX 导入后才设置 x64 的调用快速失败；
- 或：正式支持 float32，但为每个 API 声明精度语义、调整验证阈值，并在所有 R 对照中使用一致精度策略。

同时引入 `constraints`/lock 文件，CI 覆盖至少一个固定 CPU 组合，并将 JAX、jaxlib、NumPy、SciPy、R、`gamlss` 与 `gamlss.dist` 版本写入结果 JSON。

---

## 4. 公式、平滑器、预测和诊断审计

### 4.1 当前 R 平滑器覆盖

核心 `gamlss` 的 `NAMESPACE` 导出大量平滑器。当前真正具有优先实现路径的是 `pb`、`ps`、`cs`；薄板样条与张量积是 OmniLSS 扩展。以下 R 功能仍缺少 R 等价实现或只存在分阶段兼容基础：

- `pbm`、`pbz`、`pbo`、`pbp`、`pbc`、`pvc`、`cy`；
- `fp` / `bfp`、`pp`、`bp`、`scs`；
- `lo` / `vis.lo`；
- `random`、`re`、`ri`、`pcat`；
- 每个 smoother 的 R 对象方法、EDF/抽取/预测重建与绘图方法。

当前 `formula_parser._build_smooth_basis()` 对 `lo`、`random`、`re`、`ri`、`pcat`、`fp`、`bfp`、`cy`、`pvc`、`pbm`、`pbz`、`pbo`、`pbc` 实际返回带 `staged_compat=True` 的趋势、多项式、虚拟变量或三角基。运行 `lo`、`random`、`pbm`、`fp`、`cy` 的最小公式均可构造 `(8, 3)` 设计矩阵。

这类行为会把未迁移的 R 术语静默变成另一种统计模型，风险高于直接报错。**在没有基函数、惩罚、lambda 选择、EDF、拟合、预测重建和 R 对照之前，解析器应 fail closed，抛出结构化 `UnsupportedSmootherError`。**

### 4.2 `pb` / `ps` / `cs` 的 live-R 结果

在固定 300 样本 `NO` 平滑数据上，强制 live-R 快速矩阵得到：

| 平滑器 | Python 与 R deviance 差 | 当前状态 |
|---|---:|---|
| `pb` | 5.9587 | 形状可能接近，但不满足当前 1e-3 门槛；需明确允许的 R 行为差 |
| `ps` | 152.22 | 严重偏差，P0 |
| `cs` | 3237.6 | 严重偏差，P0；与旧状态文档中的 55.4 不同，表明当前工作树改动已改变行为 |

当前报告的 `14/42` 与 `CURRENT_STATUS_2026-06-05.md`（历史快照，已随文档清理删除，内容见 Git 历史）的历史数字不能混用。新报告必须记录提交、dirty diff、R 基线和运行命令。

### 4.3 当前未提交 `ps` / `cs` 改动的已证实风险

本审计对未提交 diff 进行只读语义审查，发现下列高优先级问题：

1. **双列平滑（已证实）**：`formula_parser` 先为 `ps`/`cs` 加入单列 `ones` 占位，再由 `build_smooth_design()` 使用 `np.hstack()` 追加初次拟合值。`SmoothFitInfo.basis_columns` 指向原占位列；第二轮 RS 仅覆盖该列，初次拟合列残留在设计矩阵中。`y ~ ps(x)` 由单个 R 平滑项变成含额外未受管理列的模型。
2. **lambda/EDF 生命周期错误（已证实）**：工作响应重拟合将首次 lambda 固定下来，不区分用户固定 lambda 与由 `df`/选择方法导出的 lambda；结果 lambda、EDF、knots 不回写。公式中只要含 `ps`/`cs`，外层 lambda 更新会被整体跳过，从而可能同时停止 `pb` 的更新。
3. **控制参数漂移（已证实）**：`ps_intervals` 不存于 `SmoothFitInfo`，重拟合回退到 20；`cs` 用已有 EDF 作为新 df，并硬编码 GCV，不能保证保留原始 df/方法语义。
4. **预测回归（已证实）**：训练侧新 `ps` 表示为 3 列，预测侧按保存 knots 重建 23 个 B-spline 列，触发 `PredictionSchemaError`。当前 schema 检查会 fail closed，避免静默错误，但公开 `predict()` 已无法使用此类模型。
5. **诊断脚本不构成门禁（已证实）**：`investigate_ps_lambda.py` 在 R JSON 中 lambda 为 `null` 时仍会对 `None` 调用 `np.isnan()`；新增脚本没有纳入自动回归测试。

**立即修复顺序**：先恢复“一个平滑项只拥有一个明确训练表示”的不变量，再设计可预测的保存状态；不得先继续调 lambda。推荐保留可重建的完整 basis + penalty + coefficient 方案，或定义并序列化真正的单列 smoother 对象与其 `predict(newdata)`；两者不能混用。

### 4.4 预测、诊断、选择与辅助 API

现有迁移文档已映射了大量入口，但仍应按“数据契约”而非函数名验收：

- `predict.gamlss`、`predictAll`、`Formulae2data`、参数公式和新数据平滑重建；
- `rqres`/`rqres.plot`、worm plot、`Q.stats`、`vcov`/`rvcov`、`edf`/`edfAll`、profile 与 `find.hyper`；
- `stepGAIC` 的 scope、`stepGAIC.VR/CH`、TGD 和分布选择；
- LMS、quantile sheets、删失/截断和 survival 辅助。

对于图形 API，首版可以是 data-only 返回值，但需对列名、长度、排序、分组、缺失值和边界行为建立 R 对照；“返回了绘图数据”不能自动等同 R `plot` 方法已完成。

---

## 5. 测试与基准体系审计

### 5.1 当前优点

- 具有广泛的单元、契约、族注册、R 一致性、算法、平滑器和服务端测试文件；
- R 环境可用时，R bridge 与强制 R 基准能够工作；
- `family_capabilities` 已开始区分 `compat_alias`、`placeholder` 与 evidence tier；
- `compileall` 对 `omnilss/src/omnilss` 已通过。

### 5.2 当前不足

1. [`test_audit_gamlss_migration.py`](../../omnilss/tests/test_audit_gamlss_migration.py) 仅断言 audit payload 的行数和 shape，不能证明数学实现、R parity 或 API 完整性。
2. [`docs/testing/r-consistency-coverage.md`](../testing/r-consistency-coverage.md) 只列出少量 d/p/q 族，而 capability matrix 列出 31 个 R evidence 族；文档、测试和矩阵没有单一生成源。
3. `comprehensive_r_consistency_test.py` 使用手写 `DPQR_DISTRIBUTIONS`、`FIT_DISTRIBUTIONS`、`SMOOTHERS`，没有从 capability/evidence 生成；它还曾把无 R 的 Python-only 路径标为 `success=True`。这对 smoke 有用，但不可用于 R 一致性成功结论。
4. 基准仅比较 `d/p/q`，未对 `r` 做带种子分布性质、支持域、矩/分位数或统计检验验证。
5. 单一全局容差无法同时正确覆盖密度尾部、离散量化、零膨胀混合分布和拟合 devaice；但放宽全局容差会掩盖实际错误。
6. 未生成且验证 version-pinned 的 R golden artifacts。静态 null 或历史工件不得用作发布证据。

### 5.3 必须建立的发布门禁

每个 release candidate 至少运行：

```powershell
# 由确定的 .venv 与固定 R library 执行；命令应成为 CI job。
$env:JAX_ENABLE_X64 = 'true'
.\.venv\Scripts\python.exe benchmarks\run_local_validation.py --quick
```

门禁语义必须满足：

- `--require-r` 时 Rscript、`gamlss`、`gamlss.dist`、`jsonlite` 任一缺失、跳过、空结果或全 NaN 均失败；
- R 不可用时结果只能叫 `smoke-only`，退出/报告不能声称 R 通过；
- 只从 capability matrix 中选择 E2/E3 目标；`compat_alias`/`placeholder` 只测试“默认被拒绝”；
- 每一个 E3 分布有 d/p/q + r 分布性质 + 至少一个拟合例；
- 每一个 E3 smoother 有公式解析、basis/penalty、fit、EDF、newdata prediction 和 R 对照；
- 产出 JSON、Markdown 和版本清单，作为 CI artifact；
- 性能测量必须引用同一 case 的成功 correctness artifact，否则标记 smoke-only。

---

## 6. 优先级问题清单

| 优先级 | 问题 | 证据 | 完成条件 |
|---|---|---|---|
| P0 | 发布口径高估 | 142 可解析，但仅 1 `implemented`、57 placeholder、29 alias | 自动生成状态页；对外只展示 E3/E4 数字 |
| P0 | R 基线未固定 | vendored 与 installed R 版本不同 | CI 从 vendored 固定版本安装，或记录并锁定外部 R library |
| P0 | 默认精度不确定 | `.venv` 默认 x64 关闭 | 明确 x64/float32 策略并加入启动/CI 断言 |
| P0 | d/p/q 快速矩阵 0/27 | live-R 直接执行结果 | 按族修正内核/夹具；每项有支持域、单调性、边界与 R 工件 |
| P0 | `PO` RS 拟合失配 | deviance diff 69.215 | 确认 score/Hessian/link/初值与 R 一致并通过回归例 |
| P0 | `ps`/`cs` 双列和预测回归 | 当前 dirty diff 的语义审查 | 单一训练表示、状态持久化、newdata 预测通过 |
| P0 | `pb`/`ps`/`cs` 均未达 parity | live-R 0/3 | 分别通过 R 对照；不允许“平滑支持已完成”声明 |
| P0 | GPU/auto 语义冲突 | 3 个路由测试失败 | 写定 `RS`/`auto` 合约，更新实现与边界测试 |
| P1 | 未实现 smoother 被静默近似 | `staged_compat=True` 可建立设计矩阵 | 默认 reject；每项完整迁移后才开放 |
| P1 | compat 状态分类不完备 | 27 个 compat module 未标 alias | 显式注册元数据，审计所有 kernel owner |
| P1 | R 证据矩阵与文档漂移 | coverage doc、进度 doc、runtime matrix 不一致 | 单一生成器与 CI drift check |
| P1 | 通用 family/link/generator 体系不完整 | `gen.Family`、`gamlss.family` 等无完整等价 | 定义 Python family protocol，禁止 NO/PO 骨架伪装 |
| P1 | 预测/诊断/选择缺少数据契约 | 多入口只映射名称或 data-only | 每类典型/边界输入的 R shape/value 对照 |
| P2 | R 扩展生态无范围计划 | 11 个项目没有映射 | 逐包签署“迁移/互操作/明确不支持”决策 |
| P2 | 依赖未锁定 | 开放下界导致跨环境漂移 | lock/constraints、SBOM、CPU/GPU CI 组合 |
| P2 | 服务与序列化的安全/能力门禁 | 仅在旧计划中描述，未在本次复核全部验证 | 独立安全审计；仅安全 JSON 与 capability gate 允许的族 |

---

## 7. 建议实施路线与验收标准

### 阶段 0：建立真相门禁（1 周内，阻断新功能）

1. 生成 `r-export-coverage.json`：每个 R 包/版本/导出符号都必须有 `missing`、`placeholder`、`compat_alias`、`experimental`、`implemented` 或 `unsupported_with_rationale`。
2. 将 distribution registry、factory metadata、p/q/r 实现来源、R test、R result artifact 作为唯一数据源；删除手写总数。
3. 固定 R/Python/JAX 基线并在每个测试结果中写入版本/commit/dirty 状态。
4. 默认拒绝所有 placeholder、alias 和 staged smoothers；保留显式诊断 opt-in，且在模型对象中标注不可生产。
5. 修复 capability metadata 对所有 `compat` factory 的漏标，使用显式元数据而不是模块命名约定。

**验收**：生成矩阵覆盖 `R/` 下所有 13 个项目；每个公开 Python family 能追溯到数学核、R 符号和证据状态；无人工维护的“完成率”。

### 阶段 1：最小可信核心（2–4 周）

目标是发布一个小而可信的 core，而不是扩大名单。

1. 将 `NO`、`GA`、`PO`、`BI`、`NBI`、`BE`、`ZAGA`、`GU` 作为首批矩阵；先修复基准参数夹具，再逐项修核。
2. 逐族验证 d/p/q/r、支持域、`log`/tail 参数、极限值、单调性、随机样本分布和导数。
3. 修复 `PO` RS 失配；对 `NO`、`GA`、`PO`、`BI` 的 RS、CG、Mixed 分别保留小数据 R artifact。
4. 明确 JAX x64 启动策略与 dtype 测试。

**验收**：核心 8 族的 E3 结果全绿；任一 R 缺失、NaN、shape 错误或超阈值使门禁失败；不再把 Python-only success 叫 R consistency。

### 阶段 2：恢复可信平滑与预测（2–4 周）

1. 先锁定 `pb` 的 R 定义、basis、knots、penalty、df/lambda、EDF 和预测重建；不要同时改动多个平滑器。
2. 重设计 `ps`/`cs`：一个术语一个真实设计表示；保留所有预测所需状态；分清 fixed lambda、df target、ML/GCV/REML 选择。
3. 在每一轮 RS 中显式记录 smoother 更新：old/new lambda、EDF、working response、weight summary、solver 状态。
4. 默认拒绝其它 R smoother，直到各自完成端到端迁移。

**验收**：`pb`、`ps`、`cs` 在固定 R 基线上通过 parsing、fit、EDF、multiple smoother、mixed parameter、newdata predict 和至少三个 seed 的 R 对照；不再出现 schema 列数不匹配。

### 阶段 3：分布与经典核心逐波迁移（持续）

按业务优先级逐波处理：

1. 常用连续/计数/混合族；
2. Box-Cox 与重尾/偏态族；
3. 复杂离散、零修正和多参数族；
4. `gamlss.family`、`gen.Family`、hazard、LMS、model selection、诊断与 profile。

每波必须先完成数学规范（参数化、支持域、likelihood、score、Hessian、d/p/q/r），再注册 factory，最后添加 R 证据。兼容 alias 不能在没有专属核之前提升状态。

### 阶段 4：生态扩展（范围决策后）

对每个附带 R 项目作三选一决策：

- **迁移**：定义 Python public API、依赖、验收矩阵；
- **互操作**：保留 R bridge/文件格式/数据加载，并明确非 JAX 实现；
- **明确不支持**：说明原因和推荐替代方案。

建议优先顺序：`gamlss.tr`、`gamlss.cens`、`gamlss.add` → `gamlss.mx`/`gamlss.spatial` → `gamlss.foreach`/`gamlss.prepdata`/`gamlss.ggplots` → 独立评估 `gamlss2`。

---

## 8. 开发工作流

### 8.1 每个 PR 的最小要求

- 标明涉及的 R 包、R 源符号、Python API、数学参数化和 capability 变化；
- 新功能先添加失败的 R 对照/性质测试，再实现；
- 运行带固定 x64 与 R 基线的目标测试；
- 更新生成的 capability/coverage artifact，而非手写概览数字；
- 对 alias、placeholder、staged compatibility 改动必须说明为何不会被生产路径误用；
- 性能改动必须附带相同 case 的 correctness artifact。

### 8.2 推荐的日常命令

```powershell
# 在仓库根目录，使用指定虚拟环境。
$env:JAX_ENABLE_X64 = 'true'
$env:PYTHONPATH = "$PWD\omnilss\src"

# 静态编译检查
.\.venv\Scripts\python.exe -m compileall -q omnilss\src\omnilss

# 针对当前核心问题的测试
.\.venv\Scripts\python.exe -m pytest `
  omnilss/tests/test_family_capabilities.py `
  omnilss/tests/test_method_routing_integration.py `
  omnilss/tests/test_r_consistency_smooth.py

# 生成带 R 的可追溯一致性工件（正式 gate 使用 --require-r）
.\.venv\Scripts\python.exe benchmarks\comprehensive_r_consistency_test.py `
  --quick --require-r
```

不要以 `--no-r` 的结果作为 R parity 证据；不要在未固定 R library 的环境中替换 release golden artifacts。

---

## 9. 本次审计执行记录

| 验证 | 结果 |
|---|---|
| `.venv` / R 版本检查 | 完成；发现默认 JAX x64 关闭和 R 基线版本漂移 |
| 142 个注册族逐项 `resolve()` | 142/142 成功；仅接口健康检查 |
| 能力矩阵统计 | 1 implemented、55 experimental、29 alias、57 placeholder，31 R evidence |
| 目标 pytest（7 文件） | 84 passed、3 failed；失败为 routing 合约冲突 |
| 无落盘 quick live-R 矩阵 | 14/42；DPQR 0/27、fit 11/12、smooth 0/3 |
| `compileall` | 通过 |
| R 扩展项目映射检索 | 11 个附带项目无专门 Python/文档映射 |
| dirty diff 语义审查 | 已识别 ps/cs 双列、lambda/config 漂移与新数据预测回归 |

本报告不把上述局部通过项目外推为全库通过，也不因当前有未提交实验改动而覆盖、回滚或修改任何用户文件。
