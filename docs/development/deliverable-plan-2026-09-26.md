# OmniLSS 完整交付计划（2026-09-26）

> **文档定位**：本文件是 OmniLSS 当前阶段的**唯一交付执行计划（Delivery Plan）**，定义交付物目录、验收标准、里程碑、排期和项目结构治理规则。
> **阅读顺序**：`DEVELOPMENT_PLAN.md`（项目总纲与历史阶段状态）→ 本文件（交付什么、何时、如何验收）→ `full-gamlss-migration-audit-and-roadmap-2026-09-11.md`（全量迁移技术路线 P0–P5）→ `overall-development-plan-2026-09-24.md`（事实基线与差距审计）。
> **更新规则**：交付物状态只在证据满足验收标准后更新；缺证据不标完成。

## 1. 交付范围与非目标

### 1.1 范围内（1.0 必须交付）

1. **L1 统计引擎**：R `gamlss` + `gamlss.dist` 的分布 family d/p/q/r、RS/CG/MIXED 拟合、family/link/formula、预测/诊断/模型选择、准入 smoother。
2. **L4 验证交付**：逐符号 coverage 矩阵、R golden artifact、分层测试门禁、CI fail-closed、基准报告。
3. **L3 服务层**：已通过 admission 的 gRPC/HTTP 路由、序列化、TLS、SDK 文档。
4. **周边包（D1–D4）**：`gamlss.tr`、`gamlss.cens`、`gamlss.mx`、`gamlss.add`、`gamlss.spatial`、`gamlss.foreach`、`gamlss.prepdata`、`gamlss.ggplots` 数据层、`gamlss.data`。
5. **项目结构治理**：文档生命周期、生成物管理、临时文件清理（见第 7 节）。

### 1.2 范围外（1.0 不承诺）

- `gamlss2`（独立 API，1.0 后单独规划）；
- ggplots 像素级外观一致（只要求数据层一致）；
- `ondil` 在线层与离线 R parity 等价（L2 独立验收，见 D6）；
- 未通过 admission 的 family/method/smoother 的生产可用性。

## 2. 当前基线（2026-09-26 复核）

| 项目 | 基线 |
|---|---|
| OmniLSS 版本 | 0.4.0 |
| registry family | 146（1 implemented / 102 experimental / 25 compat_alias / 18 placeholder） |
| DPQR evidence VALIDATED | 35 |
| fitting evidence VALIDATED | 2 |
| R 源码 | 13 个包目录、12 个 NAMESPACE、363 个 `.R` 文件（gamlss 5.5-3、gamlss.dist 6.1-8） |
| 测试 | 419 个 `test_*.py`；四组分层收集已全绿（omnilss 3384 / omnilss-server 1 / omnilss-pro 25 / benchmarks 10，共 3420 项，0 error / 0 skip，见 D0.3）；单次全库运行仍不在时限内，不得宣称全库通过 |
| 环境 | Python 3.10.19 / JAX 0.6.2 CPU / R 4.6.0；x64 默认关闭，parity 前必须显式开启 |
| 参考项目 | gamlss-python 0.2.0（52 源文件）、ondil 0.4.8（119 源文件） |

## 3. 交付物目录（D0–D8）

每个交付物按四层证据验收：**Surface（可用）→ Kernel（真实实现）→ Parity（R 对照）→ Admission（生产门禁）**。

### D0 审计与验证基座（阻塞一切“完成”宣称）

| 交付项 | 内容 | 验收标准 |
|---|---|---|
| D0.1 真实 coverage | 解析 12 个 NAMESPACE、R 函数定义、Python registry/exports、测试与 golden manifest 的逐符号矩阵 | JSON/Markdown 可从源码重生成；每条含 r/python 符号、四层状态、证据路径、commit |
| D0.2 golden manifest | 固定 R/包/Python/JAX 版本、源码 hash、x64、seed、容差 | 任一 parity 测试可复现；无 R 时只能标 smoke-only |
| D0.3 分层测试门禁 | 本项目四组测试面：omnilss / omnilss-server / omnilss-pro / benchmarks 分别收集与运行（`ondil`、`gamlss-python` 为参考项目，只做对比、不入门禁） | 各组 collection 零错误；统计 collected/error/skipped；`--strict-skips` 拒绝隐性 skip |
| D0.4 自动差异报告 | missing/extra/alias/placeholder/evidence 缺失 | registry 或 R 源码变化时 CI 输出差异并阻断回归 |
| D0.5 环境门禁 | parity 入口强制检查 Rscript、x64、版本 | 未满足时 test fail-closed 而非 skip |

**D0.1 交付记录（2026-09-26）**

- 工具：`tools/generate_symbol_coverage.py` 解析 12 个 NAMESPACE 的 `export*()`、
  R 函数定义（`name <- function`）、Python `__all__`/定义/stub、测试引用与 golden manifest，
  输出逐符号四层矩阵（`surface`/`kernel`/`parity`/`admission`）。
- 生成物：`docs/development/r-symbol-coverage.json` 与 `docs/development/r-symbol-coverage.md`；
  `tools/generate_coverage_artifacts.py` 已接入；`--check` 供 CI 复核新鲜度
  （回归测试 `omnilss/tests/test_symbol_coverage_tool.py`）。
- 溯源：`R/` 为 vendored（`.gitignore` 忽略）源码，因此逐行证据使用「定义文件内容哈希 +
  上游 DESCRIPTION 版本 + Python 主模块最后提交」，不引用 R 文件 commit。

**D0.2 交付记录（2026-09-26）**

- 模板 `manifest_template.json`：固定 R/gamlss/gamlss.dist/Python/JAX 版本、`jax_enable_x64`、必填 provenance 字段（覆盖 `stage0_gate.PROVENANCE_FIELDS` 超集）、默认与分 kind 容差、期望 case 集合。
- 门禁 `tools/golden_manifest_gate.py`：`--check` 只做模板/manifest/fixture 内容哈希校验（快路径）；默认模式额外探针环境并给出 `pinned` / `smoke-only` 标签；`--require-r` fail-closed——R 缺失、任一版本漂移或 x64 未开启一律失败，非发布运行只能标 `smoke-only`。
- 可复现性：Python 侧按 `generate_golden.R` 的 djb2 复算 9 个 fixture 内容哈希，全部一致；本机 `--require-r` 实测 `pinned`（python 3.10.19 / jax 0.6.2 / x64=True / R 4.6.0 / gamlss 5.5.0 / gamlss.dist 6.1.1）。
- 读法：`parity` 依次取 `test_r_consistency*` 命中、golden fixture、capability 层级；
  `registry` 表示族已注册于 capability registry，kernel 深度以 `kernel` 列
  （implementation_status：implemented/experimental/compat_alias/placeholder）为准。

**D0.3 交付记录（2026-09-26）**

- 分组口径（按用户指示调整）：`ondil/`、`gamlss-python/` 是**参考项目**，只用于 README 对比，不进入本项目测试门禁；门禁覆盖本项目自有的四个测试面：`omnilss`、`omnilss-server`、`omnilss-pro`、`benchmarks`。
- 工具 `tools/tiered_collection.py`：逐组收集/运行并 fail-closed（零错误 + 至少 1 项）；统计 collected/errors/skipped/failed；`--strict-skips` 把收集期 skip（如模块级 `importorskip`）判为失败（D7 “无隐性 skip”）；`--json` 输出报告，`--tier`/`--mode`/`--timeout` 可选；CI job `collection-gate` 执行 `--mode collect --strict-skips --json` 并上传 artifact。
- 顺带修复：`omnilss-pro` 的 `pyproject.toml` 补 `pythonpath = ["src"]`（src 布局此前导致 `ModuleNotFoundError: omnilss_pro` 收集错误，25 项现已可收集）。
- 实测（本机，装齐 `.[server,dev]` 等价依赖）：omnilss 3384 / omnilss-server 1 / omnilss-pro 25 / benchmarks 10，合计 3420 项，**0 error / 0 skip**，exit 0。
- 已知排除（显式、有据）：`benchmarks/diagnostic/` 下 5 个独立诊断脚本依赖缺失数据文件或 R bridge，以 `--ignore` 显式排除并写入 `BENCHMARK_KNOWN_EXCLUSIONS`；新增破坏仍会失败。

**D0.4 交付记录（2026-10-02）**

- 工具 `tools/coverage_diff_report.py`：复用 D0.1 的 `build_coverage()`，把逐符号矩阵分到
  `missing` / `placeholder` / `compat_alias` / `evidence_gap`（四类阻断）+ `extra`（仅报告），
  与入库 ratchet 基线 `docs/development/symbol-coverage-baseline.json` 比对：新增即回归并
  exit 1，离开即 improvement 只报告；baseline 缺失/损坏一律 fail-closed。
- 首次基线（R exports 1080）：missing 402 / placeholder 75 / compat_alias 11 /
  evidence_gap 36 / extra 300。这是**冻结现状的债务清单**，不是已达成的覆盖率；
  口径为按 R 符号计数，与第 2 节按族计数的 18 placeholder 不可直接互比。
- 流水线：`tools/generate_coverage_artifacts.py` 用同一内存矩阵快照生成
  `symbol-coverage-diff.json`，ratchet 失败时流水线返回 1；CI `collection-gate` 新增该步骤
  并上传 artifact。报告里的路径一律仓库相对，保证提交物跨机器可复现。

**D1-E 批次一交付记录（2026-10-02）**

- 范围：PARETO / PARETO1 / WEI2 / RGE / NO2 —— 此前只有 `log_pdf`，p/q/r 全是 NaN 占位。
- 新增 `omnilss/src/omnilss/dpqr_batch_e.py` 承载核函数；`distributions_b16/b12/b13/b8` 的工厂
  改为把 `d/p/q/r` 传给 `build_ad_family`。
- 顺带修正两处与 R 不符的密度（原先与 R 是**不同的分布**）：WEI2 的
  `dweibull(scale = mu^(-1/sigma), shape = sigma)`；RGE 的
  `-log(sigma) + (1/nu-1)*log(1+z) - (1+z)^(1/nu)`。NO2 的 `sigma` 是方差。
- 已知 R 缺陷（不复制，登记在 `dpqr_batch_e.R_QUIRKS`）：`pPARETO/pPARETO1` 因 R 源码笔误
  `cdf < - log(cdf)` 忽略 `log.p`；`qPARETO/qPARETO1` 完全忽略 `lower.tail`。
- 验收：`tests/test_dpqr_batch_e.py`（13 项）= 不依赖 R 的不变量 + 逐点 R 对照。
- 效果：placeholder 符号 75 → 50。

**D1-E 批次二交付记录（2026-10-02）**

- 范围：PE2 / NOF / GAF。
- **NOF / GAF 由二参数扩为三参数**（`mu, sigma, nu`）。R 的 `NOF` 方差函数是
  `sigma^2*|mu|^nu`（`sigma1 = sigma*|mu|^(nu/2)`，即 `N(mu, sigma1)`），`GAF` 是
  `sigma^2*mu^nu`（`sigma1 = sigma*mu^(nu/2-1)`，即 `Gamma(shape=1/sigma1^2, scale=mu*sigma1^2)`）。
  Python 侧原先只声明 `(mu, sigma)`，等于把 R 的 `nu=0`（NOF）/ `nu=2`（GAF）默认值写死，
  拟合时会按错误的参数个数组装。现在 `nu` 的默认值沿用 R（`NOF nu=0`、`GAF nu=2`），
  因此退化点上行为不变，新增的是真正的方差函数分支。
- PE2 去掉 `nu` 的 `[eps, 10]` 裁剪（R 只要求 `nu > 0`），并补上 R 的 `nu > 10000` 均匀极限分支。
- 新增两条 R 缺陷登记：`qNOF` 未把 `log.p` 传给 `qnorm`；`dGAF` 在 `log` 分支之后才做
  `ifelse(x<=0, 0, fy)`，导致 log 密度返回 0。我们按数学正确行为实现（分别为
  正常处理 `log_p` 与 `-inf`）。
- 验收：`tests/test_dpqr_batch_e.py` 扩到 8 个族（新增"参数个数 = R 的 `nopar`"、
  "`q` 必须正确处理 `log_p`"两条不变量，以及"R 的 `qNOF` 确实忽略 `log.p`"的实测锁定）；
  `test_r_consistency_gaf.py` / `test_r_consistency_nof.py` 改为覆盖三参数分支；
  `test_all_families_dpqr_scan.py` 把批次一、批次二与 PIG2 从 `KNOWN_NAN_PLACEHOLDER`
  移出，使全族扫描门禁真正覆盖它们。
- 效果：placeholder 符号 50 → 35；ratchet 基线同步收紧。

**D1-E 批次三交付记录（2026-10-02）**

- 范围：LQNO / MN3 / MN4 / MN5（别名 `MULTIN` 经克隆自动获得）。批次内剩余 placeholder
  从 8 降到 4（GIG / exGAUS / EGB2 / LOGSHASH）。
- **LQNO 参数化纠错**：Python 侧原先被实现成三参数的 tanh "log quantile normal"，
  与 R 毫无关系 —— R 的 `LQNO` 是两参数族
  `Y ~ Normal(mu, sqrt(mu*(1+sigma*mu)))`，`mu.link = sigma.link = "log"`。
  现按 R 定义重写，`parameters` 由 `(mu, sigma, nu)` 改为 `(mu, sigma)`。
- 新内核模块 `omnilss/dpqr_batch_f.py`；`MN3/4/5` 的 AD `log_pdf` 委托给 `dMN*`，
  保证梯度与公开 `d` 永不漂移。
- 新登记三条 R 缺陷：`qMN3`/`qMN4`/`qMN5` 完全忽略 `lower.tail` 与 `log.p`
  （R 函数体根本没用到这两个形式参数）。我们按数学正确行为处理，并有实测测试锁定
  "是 R 的缺陷而不是我们的差异"；`qLQNO` 则确实会处理 `log.p`，两侧完全一致。
- `pLQNO` 上尾按 `ndtr(-z)` 而非 `1-ndtr(z)` 求 `log_p`，避免极端上尾的抵消误差。
- 修复 `tests/test_r_consistency_lqno.py`：模板生成的旧版把 `nu` 传给 R（直接
  `unused argument` 报错）且用负 `mu`（R 会 `stop`），两处都改为合法参数化。
- 验收：新增 `tests/test_dpqr_batch_f.py`（21 项）。连续族用 `p(q(u)) == u`，
  离散分类族改用广义反函数条件 `p(q(u)) >= u > p(q(u)-1)`，另加 "pmf 之和 = 1"、
  "类别落在 `1..k`"、"参数个数 = R 的 `nopar`"（`LQNO` 2 / `MN3` 2 / `MN4` 3 / `MN5` 4）。
  `test_all_families_dpqr_scan.py` 把 LQNO/MN3/MN4/MN5/MULTIN 移出
  `KNOWN_NAN_PLACEHOLDER`，全族扫描门禁开始覆盖它们。
- 效果：placeholder 符号 35 → 15；ratchet 基线同步收紧。

**D1-E 批次四交付记录（2026-10-02）**

- 范围：exGAUS / EGB2。批次内剩余 placeholder 从 4 降到 2（GIG / LOGSHASH，两者各有
  实质性阻碍，见下）。
- **EGB2 参数化纠错**：Python 侧原先把 R 的公式套在 `log(y)` 上（R 是 `z = (x-mu)/sigma`
  作用在**原始**响应上），密度整体错误；且 `mu.link` 写成 `log`，R 是 `identity`
  （`mu` 是全实数轴上的位置参数）。两者都按 R 修正。
- exGAUS 的密度原本就与 R 一致，补齐 `p/q/r`：`q` 用向量化二分（R 用 `uniroot`，
  tol = `double.eps^0.25 ≈ 1.2e-4`，是 **x 尺度上的绝对**容差，所以对 R 的比较按
  此放宽，自洽性 `p∘q = id` 另用紧容差断言）；`r` 利用 `exGAUS = N(mu,sigma)+Exp(mean=nu)`
  的定义直接卷积抽样，比 R 的"uniform 过数值分位数"便宜得多且同样精确。
- EGB2 的 `p/q` 走 F 分布（`pf`/`qf`，scipy），与 R 的 `pEGB2`/`qEGB2` 同一路线；
  R 源码里 `sigma < 0` 的分支在 `sigma.link="log"` 下不可达，不复制。
- **GIG 阻碍（暂缓）**：R 的 `pGIG` 有 outright bug ——
  `integrate(function(x) dGIG(x, mu = 1, ...))` 把 `mu` 写死成 1，CDF 根本不是本族的；
  更关键的是密度需要 `besselK(1/sigma^2, nu)`，**JAX 没有可微的 BesselK**，
  `build_ad_family` 的 AD 路径无法成立。需要一个独立的可微 BesselK 实现（级数/渐近
  展开 + 自动选域），作为单独的工作项。
- **LOGSHASH 阻碍（暂缓）**：参考环境是**已安装的 gamlss.dist 6.1.1**（D0.5 门禁实测
  `pinned`），`dLOGSHASH` 在该版本中不存在（只存在于 vendored 的 6.1-8 源码），
  无法做 R 对照。要么升级 R 环境，要么接受无 R 对照的 `smoke-only` 交付。
- 验收：`tests/test_dpqr_batch_f.py` 扩到 22 项（新增"数值分位数自洽性"不变量）；
  修复 `tests/test_r_consistency_egb2.py`（模板生成的旧版 `nu` 取 `[-2,2]`，
  R 会 `stop`）；`test_all_families_dpqr_scan.py` 把 EGB2/EXGAUS 移出
  `KNOWN_NAN_PLACEHOLDER`。
- 效果：placeholder 符号 15 → 10；ratchet 基线同步收紧。

**link 元数据门禁交付记录（2026-10-02）**

- 缺陷：`FamilyDefinition` 的 `links` / `link_functions` / `link_inverses` /
  `link_derivatives` 允许为 `None`。族还是 placeholder 时能力门禁会拒绝它，
  一切正常；**一旦 D1 把它升为 experimental，门禁放行，`rs_fit` 却在深处以
  `TypeError: 'NoneType' object is not subscriptable` 崩溃** —— 没有族级诊断。
  `PARETO1` 在批次一接上 d/p/q/r 后就正好踩中。
- 修复：补齐全部缺元数据族的 R 默认链接（PARETO1 / PARETO1o / DPO1 用
  `mu=log, sigma=log`，LOGSHASHo 用 `mu=identity, sigma=log, nu=log, tau=log`），
  146 个族现在 100% 具备完整 link 元数据。
- 顺带修正一处名称不一致：`BCT`/`BCTUNTR` 的 `tau` 标签写作 `logshift2`，
  不在 `links.LINK_REGISTRY` 里（正确名是 `logshiftto2`），
  `core/family_factory.py` 按名查表会 KeyError；`fitted_plot.py` 里硬编码了
  这个错名，一并归一。
- 门禁：`tests/test_family_link_metadata_gate.py`（4 项）—— 全族 link 元数据
  结构完整 + 名字可识别 + `inverse(link(mu)) == mu` 往返 + 导数有限。

**D2 首项交付记录（2026-10-03）：WEI2 / PE2 的 RS 拟合修复 + RGE 拟合受 R 阻碍定性**

- 批次一/二登记的"WEI2/RGE/PE2 的 RS 拟合在通用初始值下发散"已定位到真正的根因：
  **`FamilyHeuristicInit` 暖启动用 `mu = mean(y)` 劫持了 mu 初始化**（`rs_fit` 优先采用
  形状为 `(n,)` 的暖启动 mu 向量，族感知的 `_initial_mu_beta` 从未运行）。对 WEI2 尤其致命：
  其 `mu` 是 rate 参数（`scale = mu^(-1/sigma)`，`E[Y] = Γ(1+1/σ)/mu`），从 `mean(y)`
  出发的第一步 IWLS 把 `mu` 打到 `[3e-7, 0.19]`，随后的 sigma 步直接爆炸（1 轮外循环后
  `g_dev ≈ 6.2e6`）。
- 修复（两层）：
  1. `FamilyHeuristicInit` 对 WEI2/RGE/PE2 不再提供 `mu`（`_NON_MEAN_MU_FAMILIES`），
     让 `rs_fit` 落到族感知的 `_initial_mu_beta` —— 与外层循环对非 mu 参数的
     Phase 2.1"族感知初始化是唯一事实来源"策略对齐；
  2. `_fitting_init` 按 `gamlss.dist` family 对象的 `mu.initial`/`sigma.initial`/
     `nu.initial` 表达式逐字移植 WEI2/RGE/PE2 的初值（WEI2 的
     `s.Y.s = 1.283/sqrt(var(log y))`、`log mu = -s.Y.s*log(y) - 0.5772`；
     RGE 的 `mu = y + 0.45*sd(y)`、`sigma = 0.78*sd(y)`、`nu = 0.1`；
     PE2 的 `mu = (y+mean(y))/2`、`sigma = (|y-mean(y)|+sd(y))/2`、`nu = 1.8`）。
- 验收：新增 `tests/test_rs_fit_wei2_pe2.py`（6 项）—— WEI2/PE2 各自的最小数据集上
  与**实时 R gamlss** 活拟合对照（deviance rtol 1e-3，系数 rtol 1e-2 —— WEI2/PE2 的
  似然面在 (σ, ν) 处平坦，两侧收敛点系数相对差 ≤ 7e-3，deviance 相对差 ~1e-5），
  另有初值表达式与暖启动省略的纯 Python 单元断言。
- **RGE 拟合受 R 阻碍（定性）**：R 自己无法用 `gamlss()` 拟合 RGE —— `dRGE` 在
  未加保护的 IWLS 迭代步中硬 `stop`（"x must greater than mu-sigma/nu"）。实测对
  RGE 自身采样的数据、截距-only 模型、以及显式安全初值（`mu.start`/`sigma.start`/
  `nu.start`）全部失败。RGE 保留批次一的 d/p/q/r parity，拟合保持 experimental，
  与 GIG/LOGSHASH 同一"受 R 阻碍"状态哲学。
- 澄清：探针中 NO2 的 `g_dev ≈ 8.7e6` 是模型误设（常方差数据 + 方差随均值变化的
  NO2），在正确设定的数据上 NO2 4 轮收敛（`g_dev ≈ 322`），非算法缺陷。

**D2.2 交付记录（2026-10-03）：拟合收敛状态一等化 —— 失败不再静默**

- 先澄清 C6 的现状：`jax_rs_core` / `jax_rs_batch` 的 NaN-as-convergence **已在
  `2dcab9a` 修复**（deviance 非有限即中止循环 + `converged` 要求"未触顶且无
  failed 标志"），并有 `test_jax_rs_nan_handling.py` 锁定。真正未修的是
  **默认的 NumPy RS 路径与模型容器**。
- 缺陷：`GAMLSSModel` **没有 `converged` 字段**。收敛信息只藏在
  `additional_slots["rs_converged"] / ["cg_converged"]` 里，导致
  `g_dev = inf`、`mu` 被裁剪到 1e8 的发散拟合与正常拟合在模型层面**完全无法区分**
  —— 这正是 D2.2 要治的"溢出伪成功/失败静默"，而 R 的 `gamlss()` 对象直接暴露
  `$converged`。
- 修复：
  1. `GAMLSSModel` 新增一等字段 `converged: bool | None` 与
     `termination_reason: str | None`（默认 `None` = 未报告，保证 legacy 工件与
     `gamlss_ml` 暂存路径兼容）；
  2. 全部拟合路由填充实并**fail-closed**：deviance 非有限一律
     `converged=False` + `non_finite_deviance`，绝不因变化量判据而侥幸通过；
     迭代预算耗尽为 `max_iterations_reached`，正常收敛为
     `relative_deviance_converged`（与 CG 既有词汇表一致，另有
     `step_rejected` / `no_progress_step_rejected`）；
  3. 覆盖 NumPy RS（`rs_algorithm`）、CG v2、**CG full-hessian 内联路径**
     （`method="CG"` 的默认后端，历史上完全没接线）、JAX RS 集成
     （`jax_rs_integration`）；
  4. JSON 工件的 `meta.json` 写入这两个字段并在加载时读回（`.get` 兜底，
     旧工件读出 `None` 而非谎称成功）。
- 既有 `additional_slots["rs_converged"] / ["cg_converged"]` 保留，与新字段
  保持同步，向后兼容。
- 验收：`tests/test_model_convergence_reporting.py`（11 项）—— 正常收敛 /
  迭代预算耗尽 / 非有限 deviance 三种结局的分类，四条拟合路由（RS、CG、
  RS_JAX、MIXED）与序列化往返都填字段，legacy 工件读出 `None`，
  `convergence_status()` 的 slot 回退可用。
  回归面：序列化（4 文件）、golden manifest（2 文件）、模型指标/选择、
  MIXED/CG/RS/JAX-RS 算法与 gRPC 共 19 个测试文件全绿。
- 连带修掉 3 个**消费者缺陷**（都是"读错收敛信息"这一类）：
  1. `model_selection.compare_distributions` 读 `additional_slots["converged"]`，
     而 RS 路径只写 `rs_converged` → 恒为 False（这正是
     `test_model_selection.py::test_compare_distributions_with_failure` 长期
     红色的原因，现已改读统一访问器并按其注释本意修正断言）；
  2. `api/grpc/server.py` 在两个 slot 都缺失时**默认 `True`**（fail-open），
     会向客户端谎称已收敛 → 改为 fail-closed（未知即 `False`）；
  3. `mixed_algorithm` 的 metrics/verbose 读取统一走访问器，返回模型的
     一等字段与 MIXED 判定对齐。
- **新发现的 R-parity 缺口（待办）**：R 的 `gamlss()` 会按族校验响应取值范围，
  例如含负响应时 GA 直接 `stop("response variable out of range")`；Python 侧
  目前只发 `eta clipped` 警告并继续拟合。族级响应域校验应纳入 D2 的
  "失败可分类解释"，而不是靠收敛flag 事后发现。

**D2.3 交付记录（2026-10-03）：修复 `batch_jax_rs_fit` 崩溃 —— 批量路径补齐 `eta_clip_ratio`**

- 缺陷（`tests/test_jax_rs_batch.py` 3 项长期红灯，本轮作为"不完整的功能"处理）：
  `JaxRSResult` 新增 `eta_clip_ratio` 字段后，**批量 vmap 路径的构造点没跟上**，
  抛 `TypeError: JaxRSResult.__new__() missing 1 required positional argument`。
  即批量 API 完全不可用 —— 不是结果偏差，是直接崩溃。
- 修复：在 `_vmap_same_family_group` 的 `_single_fit` 内按 `core.run_rs` 同一公式
  （`eta <= lo + 1e-12 | eta >= hi - 1e-12` 计数占比）计算该量并回填。
- **驱动因素的实测结论**（原假设被推翻，值得留档）：`eta_clip_ratio` 由
  **残差趋零**驱动，而非斜率过大。NO 的 sigma 是 log link 且边界为 `(-6, 6)`，
  无噪数据把 `exp(eta)` 压向下界 → ratio 0.5；sd=0.4 → 0.0；sd=3.0 → 0.0。
  我最初按"大斜率触发"写测试，被实测打回后按真实机制重写。
- 结构性风险（**已登记，未消除**）：`jax_rs_batch` 的 vmap kernel 是
  `jax_rs_core` IRLS 主循环的**独立重写**（vmap 无法直接调用 core），
  `compute_gdev` / `cond_fn` / `body_fn` 逐段对应。本次缺陷正是这种重复的
  直接后果。两处裁剪逻辑一处用 Python `float` 标量、一处用 traced 数组，
  强行抽公共函数需在两种语义间妥协，故**保留各自实现**，改以 parity 断言
  锁定漂移。
- 验收：`tests/test_jax_rs_batch.py` 5 项全绿（新增 2 项）：
  1. parity 测试扩展为比对**全部 7 个字段**（原只比 `g_dev` rtol 1e-9 与
     `params` rtol 1e-8），新增 `iterations` / `converged` 精确相等与
     `eta_clip_ratio` atol 1e-12 —— 这条断言是防 vmap/core 漂移的锁；
  2. `eta_clip_ratio` 边界钉住行为的定向测试（按实测机制构造，非按猜测）。
  JAX RS 相关回归 4 文件：57 通过 / 5 跳过，无回归。

**D2.4 交付记录（2026-10-03）：族级响应域校验 —— R parity 补缺**

- 先纠正上一条记录里的一个说法：`stop("response variable out of range")`
  **并非只在绘图函数里**。真正的拟合入口校验在
  `gamlss/R/gamlss-5.R:960`（`gamlss()` 函数体内，任何 IRLS 步之前）：
  `if (!family$y.valid(y)) stop("response variable out of range")`。
  `gamlss.ggplots` / `pdfplot` 里的同名 stop 是绘图侧的独立检查。
- 另一处边界澄清：R 的 **`gamlssML` 不做该校验**（`gamlssML.R` 内无 `y.valid`），
  所以 Python 的 `gamlss_ml` 也保持宽松，只在 `gamlss()` 加校验 —— 这是刻意的
  parity 边界，不是漏接。
- 新增 `src/omnilss/response_domain.py`：
  - `_RESPONSE_DOMAINS` 表**逐条从 `gamlss.dist/R/<族>.R` 原文抄出**，共 18 族
    （GA/EXP/PARETO1/WEI/WEI2/WEI3 → `y>0`；ZIP/ZABB/ZIBB → `y>=0`；
    BEINF/BEINF0/BEINF1/BEOI/BEZI → 四个不同的单位区间变体；
    MN4/MN5/PARETO/ZIPF → 位移或截断支撑集）。
  - **表外的族一律放行**。初版我按分布常识给 PO/ZINB/ALPHA/BETA/DAGUM/
    LOGGAMMA/LOMAX/NOF/FGA/GGA/MIXED/BCPEO*/WEIBULL 也加了限制，随后核对
    R 源码发现这些族**根本没有 `y.valid`** —— 凭空加限制会拒绝 R 正常拟合的
    数据，**假拒绝比漏检更糟**，全部移除。这条 provenance 规则写进了模块注释。
  - `validate_response()` 硬失败（对齐 R 的 `stop`），另提供
    `response_violation_count()` 非抛出版本；NaN/Inf 对所有族一律拒绝
    （含 R 放行的族），因为没有族能拟合非有限响应。
- 接入点选在 `gamlss()` 内 `resolve_family` 之后、所有方法路由之前，
  因此 RS / CG / MIXED / RS_JAX / joint **五条路径行为一致** ——
  只护 NumPy RS 会让其他路径继续拟合 R 拒绝的数据。
- 验收：`tests/test_response_domain.py`（36 项），含**活 R 对照**两项：
  1. R 对 GA + 负响应确实抛 `response variable out of range`
     （bridge 把 R 的 `stop` 包成 `RuntimeError`，断言落在消息上）；
  2. 正响应数据两侧都成功且 deviance 一致（rtol 1e-3），证明没有过度拒绝。
  另有逐族的谓词边界断言（`y=0` 在 ZIP 合法、在 GA 非法等）、
  三条方法路由的拒绝一致性、NO 负响应正常拟合。
- **未接入 `gamlss_ml`**：与 R 的 `gamlssML` 一致（该函数不做此检查）。

**D2.5 交付记录（2026-10-03）：清零 gRPC 批量拟合的既有失败（测试期望纠偏）**

- 上一轮记为"`test_grpc_server.py::test_grpc_batch_fit_service_direct` 失败
  （capability admission）"。本轮定位结论：**服务代码没有缺陷，是测试期望过时**。
- 证据：`BatchFit` 与 `Fit` 复用同一个 `_fit_one`，准入检查
  `require_production_family_route(..., strict=True)` 逐字相同，不存在
  "批量比单条宽松/严格"的行为差异。实测各族 RS 路由的 strict 准入状态：
  `NO` → `implemented`（放行）；`GA` / `PO` / `WEI2` / `PE2` → `experimental`
  （拒绝）。测试的第二个请求用了 `family="GA"`，被正确拒绝，
  于是 `assert response.success is True` 失败。
- 修复：
  1. `test_grpc_batch_fit_service_direct` 第二个请求改为 `family="NO"` ——
     保持原测试意图（批量能跑通多个模型并返回递增 model_id / deviance），
     不再依赖非生产族；
  2. **新增** `test_grpc_batch_fit_rejects_non_production_family`，把此前
     隐性的准入契约显式化：`BatchFit` 对 experimental 族返回
     `success=False` + `1 of 2 fits failed`，且生产族同批仍成功
     （`[True, False]`），并断言 `gamlss` 只被调用一次
     —— 证明拒绝发生在**拟合调用之前**（准入层），不是拟合失败后的兜底。
     这条断言直接锁住"客户端不能靠批量绕开准入策略"。
- 教训：既有用例长期失败时，先确认是**代码错**还是**期望错**。本例是后者 ——
  GA 的能力状态从可直接拟合变为需显式准入，测试没跟上。若照"修服务让它放行"
  去做，等于把 fail-closed 的安全边界拆掉。
- 验收：`tests/test_grpc_server.py` 35 项全绿（该文件既有失败清零）。

**D2.6 交付记录（2026-10-03）：修复 HTTP 准入测试的 pydantic 加载缺陷**

- 既有失败
  `test_service_smooth_admission.py::test_http_helper_rejects_cg_smooth_structured`
  报 `PydanticUserError: FitRequest is not fully defined; you should define Any`。
- 根因（最小复现确认，非猜测）：`omnilss-server/main.py` 用了
  `from __future__ import annotations`，pydantic 因此把字段注解当**字符串**处理。
  测试用 `importlib.util.spec_from_file_location` 动态加载该文件，
  pydantic 解析注解时拿不到该模块的 `__dict__` 作为命名空间，
  于是连顶层已 `from typing import Any` 导入的 `Any` 都"未定义"。
  **正常 `import main` 完全正常**（实测实例化成功）——即服务运行路径无缺陷，
  纯属测试加载方式与延迟注解的冲突。
- 修复**只动测试**，不改服务源码：
  1. 抽出 `_load_server_module()`，加载后显式
     `FitRequest.model_rebuild(_types_namespace=vars(module))`，
     把模块自身命名空间交给 pydantic（pydantic 2.12.4 提供的参数；
     仅调 `model_rebuild()` 不传命名空间仍报 `PydanticUndefinedAnnotation`）；
  2. **新增** `test_server_fit_request_instantiates_under_normal_import` ——
     走正常 import 路径实例化同一个模型。将来若服务真的坏了、而
     `model_rebuild()` 又把它掩盖过去，这条会先失败。
- 验收：`test_service_smooth_admission.py` + `test_http_capability_formula_query.py`
  5 项全绿。

**新发现的既有失败（未修，记录在案）**：
（D2.6 期间登记的 `test_cg_smooth_admission.py` 4 项已于 D2.7 修复，见下。）

**测试基础设施备注**：`omnilss/tests/` 共 425 个测试文件，
`pytest tests/ -m "not slow"` 全量单次运行实测超过 1.5 小时未结束（已终止），
必须按目录/主题分块跑。分块策略是本项目的既有实践。

**D2.7 交付记录（2026-10-06）：修复公式解析器对 `pb (x)` 写法的识别缺陷**

- 既有失败：`test_cg_smooth_admission.py::test_cg_rejects_every_smooth_formula_location`
  的 4 个参数化用例（`~ pb (x)` × 4 个 CG backend）。
  实际报错是 `sigma formula '~ pb (x)' references variable 'pb' that is
  missing from data`，而测试期望的是 `method='CG' does not support ...`。
- 根因（实测定位，非推测）：**检测器与解析器不一致**。
  - `has_smooth_terms()`（`formula_parser.py:22`）的正则用 `\s*\(`，
    容忍空格 → `~ pb (x)` 判为 True；
  - 但实际 term 解析（`formula_parser.py:225`）用 `r'(\w+)\((.*)\)'`，
    要求 `(` **紧跟**标识符 → `pb (x)` 落进 linear_terms，变量名是
    字符串 `"pb (x)"`。
  - 后果：CG 路由的平滑项检查（`fitting.py:979`）因 `has_smooth_terms`
    传进的是已解析前的原字符串…实际上更严重的是数据校验
    （`_fitting_utils.py:373`）把 `pb` 当普通变量先报缺失，
    CG 的拒绝信息永远发不出来，用户看到的是误导性报错。
- 修复：解析正则改为 `r'(\w+)\s*\((.*)\)'`，与检测模式对齐。
  实测确认 `log (x)` 这类非平滑函数**仍正确判为线性项**（不在平滑词表内），
  没有过度放宽。
- 验收：
  - `test_cg_smooth_admission.py` 18 项全绿（既有失败清零）。
  - `tests/test_formula.py` 新增 7 项：带空格写法解析为平滑项、
    非平滑函数保持线性、以及**不变式测试**
    `has_smooth_terms(f) == bool(parsed.smooth_terms)`（对 8 个
    smoother × 有无空格共 16 种写法）—— 直接锁住"两处不许再漂移"，
    这是本次缺陷的成因。
  - 回归面：公式解析块 8 文件 146 项、平滑与 CG 拟合块 7 文件 65 项全绿。

**D2.8 交付记录（2026-10-06）：LOGNO 参数化错位修复 —— 从 g_dev 8.9e12 到与 R 逐位一致**

探测 RS 拟合覆盖时发现：LOGNO 的 `g_dev = 8.9e12`，而 R 是 `173.05`，
更糟的是它报 `converged=True`、`termination_reason=relative_deviance_converged`
—— 又一次"发散静默报告为成功"。

**缺陷一：d/p/q/r 四个函数的 mu 参数化错位（根因）**

R 的 gamlss.dist 有两个对数正态族，mu 语义不同：
- `LOGNO`： `dLOGNO = dlnorm(x, meanlog = mu, sdlog = sigma)` → mu 是 **log(Y) 的均值**
- `LOGNO2`：`dLOGNO2` 内部用 `mean = log(mu)` → mu 是 **中位数**

Python 的 `dLOGNO` / `pLOGNO` / `qLOGNO` / `rLOGNO` 全都用了 `log(mu)`
约定 —— 即数值上等于 R 的 **dLOGNO2**。活 R 判决性证据：

    dLOGNO (c(2.1509,2.3591), mu=0.8, sigma=0.4) -> 0.46200903 0.41830590
    dLOGNO2(c(2.1509,2.3591), mu=0.8, sigma=0.4) -> 0.02181086 0.01093765
    Python 修复前                                 -> 0.021811  0.010938   ← 等于 LOGNO2

四个函数已统一改为 R 的 meanlog 约定，d/p/q 与活 R 逐位一致（rtol 1e-8）。
（r 因 RNG 不同本就不同。）

**缺陷二：RS 用了观测 Hessian 而非期望 Hessian**

R 声明 `d2ldd2 = function(sigma) -2/sigma^2` —— **签名只接受 sigma**，
这只有在对 y 取期望后才可能，即 RS 用 Fisher scoring。
Python 用的是观测 Hessian `1/sigma^2 - 3(log y - mu)^2/sigma^4`：
- 当 `(log y - mu)^2 < sigma^2/3` 时它变**正** → RS 工作权重失效
- 实测：sigma=0.342 处 FD（观测）+61 而解析 -1654，量级与符号都不对
- 后果：sigma 崩到 exp(-39.8) ≈ 1e-18，deviance 340 → 9.5e13

已改为 R 的期望 Hessian `-2/sigma^2`（恒负定，与 y 无关）。

**修复效果**：Python g_dev `173.05494254762493` vs R `173.05494254762488`
（差 5e-14）；mu 系数逐位一致（0.76835679 / -0.07354675）。

**验收**：`tests/test_logno_parity.py`（8 项）—— d/p/q 对活 R 值、
mu 是 log 尺度（密度峰值在 exp(mu - sigma^2) 而非 mu）、
RS deviance parity、活 R 对照、期望 Hessian 恒负定且不随 y 变、
score 对 R 表达式 `((log y - mu)^2 - sigma^2)/sigma^3`。

**方法论要点**：
1. 收敛不等于 parity —— 本次又是 `converged=True` 掩盖了一个 8.9e12 的解；
2. 导数对不上时，用**有限差分**定位，别读代码猜；
3. R 导数函数的**签名**暴露语义：`function(sigma)` 不含 y ⇒ 期望 Hessian；
4. 同族存在多个参数化变体（LOGNO vs LOGNO2）时，必须以活 R 数值判定归属，
   不能看名字想当然。

**D2.9 交付记录（2026-10-06）：族 parity 批量扫描工具 —— 把逐族排查固化为三关**

D2.8 靠手工逐族排查找出 LOGNO 的缺陷，但 146 个族不可能都手查。本项把
那套方法固化成工具：`tools/rs_family_parity_scan.py` + `tools/r_parity_batch.R`，
一次跑完三关并输出覆盖率。

**三关设计**

| 关 | 比什么 | 阈值 | 需要 R |
|----|--------|------|--------|
| `dpqr` | d/p/q 数值对活 R `gamlss.dist` | rel 1e-6 | 是 |
| `deriv` | score 对密度中心有限差分 | rel 1e-3 | 否 |
| `rsdev` | RS 拟合 deviance 对活 R `gamlss()` | rel 1e-3 | 是 |

所有 R 工作在**单个 Rscript 进程**内完成（逐族起进程约 2s，100+ 族不可行）；
作业列表经**文件**传入（整扫描的 JSON 远超 Windows 命令行长度限制，
WinError 206）。

**工具自身的三次修正（都是"假失败"比"漏检"更危险的例子）**

1. **族发现**：`dir(distributions)` 只有 19 个名字，其余在 `distributions_b*.py`。
   改为从 `resolve_family` 的报错信息里提取全部名字 → **147 族**。
2. **网格标定**：原用 `UNIT_FAMILIES` / `COUNT_FAMILIES` 手工白名单，
   导致 LOGITNO（支撑 (0,1)）被按 0.3–4.0 探测、零膨胀计数族被按连续网格探测。
   改为**从族对象自身推导**：`type`（Continuous/Discrete/Mixed）定响应支撑，
   `links`（log/logit/identity）定每个参数的取值区间，`fixed_parameters`
   决定是否补 `bd`。白名单删除。
3. **支撑判定用相对阈值**：`d > 0` 不够 —— LOGITNO 在 y>1 处返回极小正数，
   (0.3,4.0) 网格因此"满分"通过却只测到 1 个有效点。改为要求
   `d >= 1e-6 * max(d)`。另：无法标定的族一律记 `skip` 而非 `fail`
   ——"没查到"和"查错了"是两件事，不能混为一谈。

标定修正的效果（同一份代码、同一批族，deriv 与 dpqr 两关不依赖拟合数据，
故不受后续调参影响）：

    deriv : 106/146 pass, fail=40  ->  120/146 pass, fail=20, skip=6
    dpqr  :  48/146 pass, fail=35  ->   63/146 pass, fail=36, skip=47

deriv 关 20 个"失败"是工具自己造的，真实候选缺陷只剩 20 个。

**顺带修掉的真实崩溃**：DPO1 密度调用 `jnp.lgamma` —— JAX 根本没有这个函数
（`jnp.gammaln` 也不存在），正确的是 `jax.scipy.special.gammaln`。
扫描器第一次发现它时整轮扫描被中止，现已修复并对单族异常做了容错。

**验收**：`tests/test_rs_family_parity_scan.py`（22 项）—— 族发现完整性、
标定不靠白名单、`skip` 不冒充 `fail`、**故意写错的 score 仍被抓到**
（保证标定没把关卡变成永真）、拟合样本不退化且落在支撑内、计数族样本保持整数。

**产出**：M2 需要的首个真实覆盖率数字，并给出按族排序的待修清单。

**已知待办（下一项的第一件事）**：扫描器缺**单族超时**。已加单族异常容错，
但某个族的 RS 拟合若不终止就会挂死整轮 —— 实测有一轮跑了 35 分钟仍未结束，
只能强杀。基线因此无法在合理时间内重测。：`rsdev` 关的基线尚未在最终参数下重测。
合成样本从"6 点网格平铺"改为按族 `type` 感知采样（原先响应只有 6 个不同取值，
退化到 SST 报出 deviance 是 R 的 80 倍、TF2 21 倍，根本分不清是缺陷还是数据
假象）；数据变"难收敛"后又把 `N_CYC` 降到 20、`N_OBS` 降到 48（两侧同降，
比较仍公平），否则整轮扫描超过 30 分钟。因此 **deriv / dpqr 两关的数字已可引用，
rsdev 的数字需重测后才可作为基线**。

**D2.11 交付记录（2026-10-06）：扫描器加单族超时 —— 让基线测得出来**

D2.9 留的待办落地。上一轮全量扫描跑了 **35 分钟仍未结束**，只能强杀：
逐族的 `try/except` 只能挡住"抛异常"，挡不住"根本不返回"。
R 的 deviance 越界、早退这类表现都在单侧对照里出现，若无硬上限，
一次异常就能让整轮基线无法生成。

**为什么必须用子进程**：Windows 没有 `signal.alarm`；线程也无法打断
正在跑的循环。唯一可靠手段是让每个族在子进程里跑，超时即 terminate。

**实现**（`tools/rs_family_parity_scan.py`）
- 抽出 `evaluate_family(family, ...)`：一族的全部关卡集中一处，
  原来是 `main()` 里的 lambda 三连；
- `_worker_loop` + `_Worker`：一个**常驻**子进程通过管道接单，
  启动开销（JAX import）只付一次；只有某族真的超时时才重启 worker；
- 超时族记 `skip`，detail 写明 `probe exceeded Ns and was killed`
  —— 与"无法探测"同属"没查成"，不记 `fail`；
- `--timeout`（默认 180s，`--timeout 0` 关闭退回进程内跑）；
  `--quiet` 关闭逐族进度；子进程按 EOF 自行退出，关闭时无 traceback 噪声。

**验收**：`tests/test_rs_family_parity_scan.py` 增至 25 项，新增
- 超时确实触发且判为 `skip`（用 10ms deadline：子进程还在 import JAX，
  两族必然超时，测的是判决形状而非时序）；
- `_timeout_row` 必须写明 deadline；
- `evaluate_family` 确实串起三关（防止重构漏掉某一关）。

**顺带的方法论**：超时也是一种"假失败"来源 —— 若把超时记成 `fail`，
一轮扫描里超时越多的族看起来越"坏"，会误导排查方向。

**D2.10 交付记录（2026-10-06）：BI 二项响应被 `bd` 二次缩放 —— 计数当比例用**

扫描器在 BI/DBI 上报出 `max rel diff 1.35e16`，追下去发现根因不在导数。

**根因：R 的二项响应是计数，我们当成了比例**

R 侧（活 R-4.6.0 判决性证据）：

- `gamlss()` 对二项族接受**两列矩阵** `cbind(成功, 失败)`，
  取 `y = 成功数`、`bd = 成功+失败`；
- `dBI` 的函数体就是 `dbinom(x, size = bd, prob = mu)` —— x 已是计数；
- `BI()$dldm = (y - bd*mu)/(mu*(1-mu))`，只有 y 是计数才量纲自洽。

Python 侧却在三处都做了 `y_counts = round(y * bd)`（`dBI`、
`g_dev_inc`、`dldm`），即**把分母又乘了一遍**：`bd=10` 时 `y=1`（1 次成功）
被当成 10 次成功。`pBI`/`qBI`/`rBI` 另有三处问题：逐元素 zip 在 `bd` 为标量
时直接崩、参数长度不同时不做 R 式循环补齐、`rBI` 干脆没有 `bd` 参数。

**为什么长期没被发现**：`test_r_consistency_bi.py` 全部用 `bd=1`（Bernoulli），
而 `bd=1` 时 `round(y*1) = y`，缺陷完全不可见。

**修复**：`dBI`/`pBI`/`qBI`/`rBI` 与 `g_dev_inc`/`dldm` 全部改为 R 的计数语义，
并补上 R 的 `rep(., length = max(...))` 循环补齐。活 R 逐位对照：

    dBI(c(0,2,4,6,8,10), mu=c(.25,...,.75), bd=10)
      R      : 0.0563135147095 0.1756529531060 0.2383666466221 ... 
      Python : 0.05631351 0.17565295 0.23836665 ...            （rtol 1e-10）
    qBI(c(.1,.3,.5,.7,.9), mu=0.4, bd=10)
      R      : 2 3 4 5 6
      Python : [2. 3. 4. 5. 6.]

**验收**：`tests/test_bi_binomial_denominator_parity.py`（10 项）—— d/p/q 对活 R
实测值、参数循环补齐、`y=1 ≠ y=10` 的回归断言（直接钉死原缺陷）、
`bd=1` Bernoulli 路径不受影响、`g_dev_inc`/`dldm`/`d2ldm2` 对 R 表达式，
以及一次**活的 R 进程**对照（经 `tools/r_parity_batch.R`）。

**方法论要点**：
1. deriv 关对 BI 的缺陷是**盲的** —— 有限差分与解析导数跑的是同一个错误变换，
   自洽但全错。只有拿活 R 的**数值**才能戳穿；
2. "既有测试全绿"不等于"这条路径对"——覆盖的是 `bd=1`，缺陷在 `bd>1`；
3. R 的响应约定要从 `gamlss()` 的入口读，不是从 `dXY()` 的函数体猜。

**D2.12 交付记录（2026-10-06）：BCT 拟合路径漏掉截断项 —— 密度与导数不同源**

扫描器在 BCT 上报 `deriv: sigma / nu / tau` 三个参数导数对不上
（rel diff 0.24–0.29），而 `mu` 对上。**这个"只有 mu 对得上"的形状就是线索。**

**根因：密度和拟合用的对数密度是两个函数**

Box-Cox t 是**截断**族。R 的 `dBCT` 末尾有：

    loglik <- loglik + fTz - log(pt(1/(sigma * abs(nu)), df = tau))

我们自己的 `dpqr_functions.dBCT` 也有这一项。但 `distributions.BCT` 里
喂给 `g_dev_inc` 与数值 score 的 `_log_density` **漏了它**。

实测（sigma=1.5, nu=0.7, tau=4）：

    max | log(dBCT) - (-g_dev_inc/2) |  =  0.2199204   （与 y 无关的常数）
    exp(0.2199204)                      =  1.2459
    log(1 / pt(0.952, df=4))             =  0.2199204   ← 判决性对上

常数偏移看着无害，deviance 上也大体无害 —— 但该项**依赖 sigma 与 nu**，
所以**每个 score 和 Hessian 都是错的**。而 `mu` 的导数恰好不受影响，
因为与参数无关的常数在求导时消掉 —— 这正是缺陷长期没被发现的形状。

**修复**：`_truncation_log_term()` 补上该项，用正则化不完全贝塔恒等式

    pt(t, df) = 1 - 0.5 * I_{df/(df+t^2)}(df/2, 1/2)   (t > 0)

JAX 只有 `t.pdf` / `t.logpdf`，**没有 Student-t CDF**，而该式可微；
这也正是 `BCTo._bcto_log_pdf` 已在用的写法。`|nu|` 取下界而非允许除零，
避免 `1/|nu|` 溢出成 inf/NaN 破坏梯度。

**修复效果**

    max | log d - (-g_dev_inc/2) |  ：0.2199204  ->  0.0
    mu / sigma / nu / tau score 对中心差分：rel 1e-11 ~ 2e-10
    dBCT 对活 R：12 位逐位一致

**验收**：`tests/test_bct_truncation_parity.py`（9 项）—— d 与 log d 对活 R
实测值、d 与 `g_dev_inc` 恒等、四个参数 score 对中心差分、
**截断项确实依赖 sigma 与 nu**（钉死"不得再当成常数"）、
以及 `BCTo` 仍**无**截断（防止把修复过度套到 untruncated 变体上）。

**顺带查出但本次未修：Box-Cox 族的 mu link 整簇错位**

把 R 与我们的 link 逐个对表，模式很清楚 —— `o` 后缀（原始参数化，mu 是
几何均值 → log link）与截断变体（mu 是位置参数 → identity link）
在我们这里**正好互换**：

| 族 | R `mu.link` | 我们 | R `tau.link` | 我们 |
|----|------------|------|------------|------|
| BCT | identity | **log** ✗ | log | **logshiftto2** ✗ |
| BCTo | log | **identity** ✗ | log | log ✓ |
| BCCG | identity | **log** ✗ | – | – |
| BCCGo | log | log ✓ | – | – |
| BCPE | identity | **log** ✗ | log | log ✓ |
| BCPEo | log | **identity** ✗ | log | log ✓ |

> **D2.16 已更正**：本段的前提当时是错的。R 里 `BCTo` / `BCCGo` / `BCPEo`
> 与 `BCT` / `BCCG` / `BCPE` 的 `d` 函数体**逐字相同**
> （`identical(deparse(body(dBCTo)), deparse(body(dBCT))) == TRUE`），
> 唯一差别就是 `mu.link` 在 `log` / `identity` 之间切换 —— `o` 后缀
> **不是** untruncated。真正的无截断族是 R 真实存在的第三组
> `BCTuntr` / `BCCGuntr` / `BCPEuntr`（`mu.link = identity`，与截断版相同），
> 我们此前完全没实现：D2.16 补齐了它们（见下）。`BCTo` / `BCPEo` 的
> `mu.link` 同在 D2.16 从 `identity` 改为 `log`（`BCCGo` 原本就对）。

**为什么本次没顺手改掉 link**：试过了，**改成 `identity` 后拟合直接发散**
（`mu ~ -1.5e7`、`termination_reason=non_finite_deviance`）。
原因是我们的 BCT 用**数值** score/Hessian，而 R 用闭式
（`d2ldd2 = -2*tau/(sigma^2*(tau+3))` 等）；自然尺度上 1e-4 步长的
数值 Hessian 撑不住。换 link 只是换优化几何，不改变最优解，
所以先把 link 改回 `log`（能稳定收敛），把根因留给下一项。

**已知剩余差距（不掩盖）**：BCT 在 500 轮下收敛到 deviance 137.66，
R 收敛在 136.55（rel 2.2e-3，仍高于 1e-3 阈值）。这是收敛速度问题，
不是错解 —— Python 侧 deviance 更高即拟合更差，没有出现 LOGNO 那种
"发散却报告收敛"。**修法是移植 R 的解析导数**，见下一项建议。

**方法论要点**：
1. "只有部分参数对得上"本身就是线索 —— 与参数无关的常数会从导数里消失；
2. 同一族里 `d` 与 `g_dev_inc` 应当描述**同一个**密度。差一个常数不是
   "只是归一化不同"，要先问它依不依赖参数；
3. 修族时先确认变体：`BCT` 与 `BCTo` 只差截断项，修复极易过度套用；
4. 改 link 前先问"这是错解还是只是优化几何不同" —— 本次两者都不是问题
   的根因，改了反而发散。

**D2.13 交付记录（2026-10-07）：BCT 移植 R 解析导数 + 对齐 R 的 link —— RS deviance rel 9.4e-6**

D2.12 结尾点名"修法是移植 R 的解析导数"，本项按 **先导数、后 link** 的顺序
执行。顺序不是偏好，是**实测约束**：单独换导数会让拟合变差（见下表）。

**第一步：移植 R 的六个闭式 score 与四个对角二阶导**

从活 R 抄 `BCT()` 的 `dldm / dldd / dldv / dldt` 与 `d2ldm2 / d2ldd2 /
d2ldv2 / d2ldt2`，替换 `_numeric_score` / `_numeric_hessian`
（step 1e-5 / 1e-4 的中心差分）。辅助量：

- `dldd` / `dldv` 需要 `h = dt(1/(sigma*|nu|), tau) / pt(1/(sigma*|nu|), tau)`；
- `dldt` 需要 `digamma`，`d2ldt2` 需要 `trigamma`（`jax.scipy.special.polygamma(1, ·)`）。

R 这几个导数本身**很粗**——`d2ldv2` 字面就是 `-7*sigma^2/4`，`dldt` 更是用
**硬编码 0.01 前向步长**去微分 `log pt`。所以 parity 的目标是"和 R 一致"，
不是"和密度精确一致"。删除 `_BCT_HESS_FLOOR`（那个族级夹取是为压制**数值**
Hessian 不稳而加的，解析化后多余；RS 后端已在使用点做 R 自己的 -1e-15 地板）。

**第二步：对齐 link —— `mu: log → identity`、`tau: logshiftto2 → log`**

同时把 `tau` 的下界从 `2` 改为 `eps`：R 的 `dBCT` 只要求 `tau > 0`
（`if (any(tau <= 0)) stop(...)`），我们此前把 `tau` 夹在 2 以上，
而 `logshiftto2` 即 `log(tau - 2)` **隐含 tau > 2 的地板**。

**三步实测对照**（60 点数据集，RS，n_cyc=500；R 报 **199.5647**）：

| 阶段 | deviance | 与 R 之差 |
|------|----------|-----------|
| 基线（数值导数 + 旧 link） | 200.7269 | +1.16 |
| **仅**解析导数（link 未动） | 209.5343 | **+9.97 —— 反而更差** |
| 解析导数 + R 的 link | **199.5666** | **+0.0019（rel 9.4e-6）** |

中间那行是本项最值得记的东西：解析导数配 `logshiftto2` 时，RS 把
`tau` 的 eta 一路推到 **-35.35**，即 `tau = exp(-35.35) + 2 = 2.0` ——
**正好撞在那块隐含地板上走不动**。扫描 tau 可见 deviance 单调下降并在
`tau → ∞`（Box-Cox 正态极限）处触底，这正是 R 的解所在：

    tau=2.001 → 209.79    tau=10  → 201.45    tau=1e3  → 199.66
    tau=4     → 204.43    tau=50  → 200.03    tau=1e6  → 199.637
    R 报 199.5647

也就是说旧 link **够不到**最优解，不是收敛慢。

**顺带修掉两个真实缺陷（都是本次改动暴露的）**

1. **`dBCT` 在 traced 值上崩溃**。截断项原来调 `scipy.stats.t.logcdf`，
   scipy 会对入参调 `__array__` → `TracerArrayConversionError`。而
   `fitting_cg` 恰好在 `jit` 内对 `family.d(..., log=True)` 求导来给候选步打分，
   于是**每个 CG 周期都失败**。改用不完全贝塔恒等式（实测
   `|betainc - scipy| ≤ 7.3e-15`，远在 12 位要求内）。
2. **`betainc` 对 `a`/`b` 无梯度**，而 `a = df/2`、df 就是 tau ——
   按 tau 求导会抛 `Betainc gradient with respect to a and b not supported`，
   同样致命于 CG。给 `_log_pt_t` 加 `custom_jvp`：对 `arg` 用解析式
   `d/darg log pt = -pdf/pt`，对 `df` 用中心差分（**R 自己的 `dldt` 就是这么绕的**）。

**验收**：`tests/test_bct_analytic_derivatives.py`（28 项）+ 修正后的
`tests/test_bct_truncation_parity.py`（9 项），共 **37 项全通过**。
覆盖：link 与 R 逐个对表、`tau` 下界不再是 2、解析 score 对密度导数
（tau 阈值 5e-5 且注明 R 自身误差 1.84e-5）、解析 Hessian 严格负、
**四个 score 与四个二阶导全部对活 R 逐位比对**、`dBCT` 可 jit 可求导、
`_log_pt_t` 对 scipy、以及 RS deviance 对活 R（rel < 1e-4）。

**一处过期期望已修正**：`test_bct_truncation_parity.py` 原先断言 tau score
与**我们自己的密度**中心差分 rel < 1e-6。这条现在**在 R 那里就不成立**——
实测 R 的 `dldt` 距其自身密度的精确导数 1.84e-5，而我们的解析 score 距
**R** 仅 2.8e-14。改基准为"离活 R 多近"，这才是 parity 的实际契约。

**方法论要点**：
1. **修复的中间态可能比修复前更差**。导数与 link 是两个耦合变量，
   单步只改一个会得到误导性的结论；A/B 必须留全tab的中间行。
2. **"换 link 只改优化几何"这句话有前提**：前提是导数已正确。
   导数错时，link 决定的是 RS 会撞上哪块隐含地板 —— `logshiftto2`
   悄悄把 `tau > 2` 写进了 link 里。
3. **抄 R 的导数就要连粗糙一起抄**。`d2ldv2 = -7σ²/4` 看着该修，
   但它就是 R 喂给 IRLS 的东西；"改进"会让 parity 变差。
4. **R 自身的数值近似会污染我们的验收阈值**。断言的基准要问"离谁近"，
   不能默认"离我们自己的密度近" —— 有时 R 才是更远的那个。
5. `jax.scipy.stats.t` 只有 `pdf`/`logpdf`（无 `cdf`/`logcdf`），但**对
   第三参数有梯度**；`betainc` 反之（有 `x` 的梯度，无 `a`/`b` 的）。
   特殊函数的梯度支持要逐个测，不能靠类推。

**Box-Cox 簇其余族（BCCG / BCPE / 截断克隆壳）的解析导数与 link 尚未移植**，
是下一项。见 D2.15。

**D2.15 交付记录（2026-10-07）：BCCG / BCPE 密度同源化 + 解析导数 —— deriv 由fail 转 pass**

扫描器此前报`BCCG deriv fail`（sigma rel 2.88e-01）、`BCPE deriv fail`
（mu rel 1.20e+00）。按 D2.13 定的顺序（先密度、再导数、再 link）处理，
结果是三族 `deriv` 全 `pass`。

**第一层缺陷：拟合路径的密度根本不是这个族的密度**

与 D2.12 在 BCT 上发现的形状相同，且**已扩散到簇内其余两族**：

- **BCCG** 漏掉截断项 `-log(pnorm(1/(sigma*|nu|)))`。实测
  `log dBCCG` 与 `-g_dev_inc/2` 相差**恰好 0.20179639**（与 y 无关的常数，
  等于 `-log(pnorm(1/(0.65*1.7)))`）；
- **BCPE 更严重**：漏掉标准化常数 `c` **且**漏掉截断项，kernel 还写成了
  `-0.5*|z|^tau` 而R 用 `-0.5*(|z/c|^tau`。由于该项随 y 变化，
  偏差**不是常数** —— 在探测网格上 spread 达 **1.539**。

两者的 `dpqr_functions.dX` 一直是对的。现在打分路径**直接 import d/p/q/r
用的那几个助手**（`_bcpe_c` / `_bcpe_ftrunc_log`），从机制上保证"一个族只有一个密度"。
修后 `max | log d - (-g_dev_inc/2) |` = **4.44e-16**（双族）。

**第二层：数值导数 + 错误的 link**（与 BCT 同型）

逐条抄R 的闭式（`BCCG` 三参数用正态 hazard `dnorm/pnorm`；`BCPE` 四参数含
`lgamma(1/tau)`、`d log(c)/d tau`、`pgamma`），link 统一为
`mu: identity`（原 `log`）、`sigma: log`、`nu: identity`、`tau: log`。

**验收**（60 点严格正数据，RS / n_cyc=500，两侧数据逐位相同）：

| 族 | 我们 | 活 R | 相对差 |
|---|---|---|---|
| BCCG | 166.308044 | 166.307802 | **1.45e-06** |
| BCPE | 161.923601 | 162.016463 | 5.7e-04（我们**更低**） |

**BCPE"我们更好"经过查证而非默认接受**：把双方系数拿到 **R 自己的
`dBCPE`** 上算，R 的系数给 162.016463、我们的给 **161.923601** —— 我们在
R 的目标函数上确实更优。这是 D2.12 记 PE 时同型的弱识别平坦区
（sigma 0.72vs 0.92、tau 4.95 vs 5.27沿平坦方向移动），**不是密度算错**。
测试因此把"不比R 差"和"在 R 目标函数上不劣"分成两条，且**导数逐位对活 R**
才是主断言 —— 单纯断言 deviance 收敛到某个值抓不住密度错误。

**验收**：`tests/test_bccg_bcpe_analytic_parity.py`（**42 项**）。覆盖 d 与
`g_dev_inc` 恒等、kernel 确含 `|z/c|^tau`、截断项确实依赖参数、link 与 R
对表、解析 score 对密度导数、解析 Hessian 严格负、**全部 score 与二阶导对活 R
逐位比对**（含 `d2ldm2` 在 `tau<1.05` 时R 退化为 `-dldm²` 的分支）、
以及端到端拟合 parity。回归：Box-Cox 全簇 **84 项**通过、克隆壳 8 项通过。

**方法论要点**：
1. **"d 与 g_dev_inc 是同一个密度"要逐族查，不能查一族的结论推全簇。**
   BCT 修完不等于BCCG/BCPE 修完—— 三者的漏项各不相同（常数 / 常数+缩放+常数）。
2. **偏差是常数还是随 y 变，决定了它有多危险。** BCCG 的是常数（看着无害，
   实则污染全部score），BCPE 的随 y 变（直接改变最优解）。两种都要查。
3. **"我们比 R 好"默认当作可疑，查证后才认**。查证方法是**把双方系数放到
   R 自己的密度函数上算** —— 这一步能区分"找到了更优解"与"优化了错误的目标"。
4. **R 各函数的签名不统一**：同一族里`d2ldm2` 有的带 `y` 有的不带，
   且带 `y` 的**值也可能不依赖 y**（R body 里的 `z` 是死代码）。
   用 `names(formals(...))` 查，不要猜。

**D2.16 交付记录（2026-10-07）：补齐 R 的无截断 Box-Cox 第三组 + 更正 `o` 后缀的误解**

上一轮的建议是**基于一个错误前提**的，纠正它本身就是这次交付的一部分。

**前提错在哪**："R 里 `BCTo` / `BCCGo` / `BCPEo` 是无截断变体，
我们的 `BCTUNTR` / `BCCGUNTR` / `BCPEUNTR` 是错位的克隆壳"。

在活 R 上逐条验证后：

1. `identical(deparse(body(dBCTo)), deparse(body(dBCT)))` == **TRUE**。
   `dBCCGo` vs `dBCCG`、`dBCPEo` vs `dBCPE` 同样为 TRUE。
   `o` 后缀的**唯一**差别是 `mu.link` 在 `log` / `identity` 之间切换。
   → `o` **不是** untruncated。
2. R 里**另有**一组真实存在的无截断族：`BCTuntr` / `BCCGuntr` / `BCPEuntr`
   （`grep("[Uu]ntr", ls(ns))` 命中且在 NAMESPACE 导出里）。
   它们的 `mu.link` 是 `identity` —— 与**截断版相同**，不是 log。
3. 这三个族在 R 里**没有 `d` / `p` / `q` 函数**，只有 `G.dev.incr`
   加一条"`F(1/(sigma*\|nu\|)) < 0.99` 就 warning"。它们的
   `G.dev.incr` 正是截断版密度去掉 `-log F(1/(sigma*|nu|))`。

所以真正的缺陷是：**这三个名字此前指向截断版实现**（名字说无截断、
行为有截断），而真正的无截断族**根本没实现**。

**截断进入的三处**（缺任何一处族就不自洽）：

- 密度里的 `-log F(1/(sigma*|nu|))`；
- `dldd` / `dldv` / `dldt` 里配套的 hazard
  `h = f(1/(sigma*|nu|)) / F(1/(sigma*|nu|))`（`dldm` 不含，
  因为截断点不涉及 `mu`）；
- `p` / `q` 经截断常数重标。

**link 与二阶导数都没变**：`d2ldm2` / `d2ldd2` / `d2ldv2` / `d2ldt2`
在两版间 `identical()` 为 TRUE（它们都不含截断常数）。
唯一例外是 `BCPEuntr` 的 `d2ldm2`，它**没有** `BCPE` 里
`if (any(tau < 1.05)) -dldm^2` 那个分支。

**实现**：新增 `dpqr_functions.d/p/q/rBCTuntr|dBCCGuntr|dBCPEuntr`
（共用 `_bcx_z` / `_bcx_inv_z`）；`distributions_boxcox_compat.py`
重写为 `distributions_boxcox_untr.py`，含三套抄自 R 的解析 score 与
对角 Hessian。`BCTo` / `BCPEo` 的 `mu.link` 同批从 `identity` 改为 `log`
（`BCCGo` 原本就对）。

**顺带发现并修掉的第二个缺陷**：无截断族的支撑是
`1 + nu*sigma*z > 0`，而截断族的截断点**恰好就是**这个边界
（`nu>0` 时 `-1/(sigma*nu)`），所以截断版的 `q` 从来看不见越界的 `z`。
无截断族会看见（`nu=1.3, sigma=0.7` 时边界 `-1.10`，
而 t(3.5) 在 `p=0.01` 处是 `-4.06`）。原来的
`jnp.maximum(q, eps)` 会返回一个极小正数、而它的 CDF 是 **0.17** ——
回环断裂且静默报错概率。改为返回支撑边界 `0`，并 `p(0) = 0`。

**验收**（40 点严格正数据，RS，两侧数据逐位相同）：

| 族 | 我们 | 活 R | 相对差 |
|---|---|---|---|
| BCTUNTR | 109.154038 | 109.154072 | **3.11e-07** |
| BCCGUNTR | 109.154041 | 109.154184 | **1.31e-06** |
| BCPEUNTR | 108.313741 | 108.313843 | **9.46e-07** |

全部 score 与对角二阶导对活 R 逐点比对，**最大 rel 9.9e-15**。

**测试**：`tests/test_boxcox_untr_analytic_parity.py`（**40 项**）。
`tests/test_r_consistency_boxcox_untr_more.py` **整份重写**：它原来断言
"`BCTUNTR` 的 `d` 等于 R 的 `dBCT`"—— 那正是缺陷本身
（"和 R 一致"其实是"和错的 R 函数一致"）。现改用 R 的
`BCTuntr$G.dev.incr` 作参考，`rbus` 的 `eval_family.R` 新增 `gdev` 类型。
`tests/test_family_capabilities.py` 里拿 `BCCGUNTR` 当 compat-alias 样本
的那条改用 `DBI` / `DBURR12` / `COUNT_1_31`。

**三个族已从 compat alias 名单移出**（它们不再是别名），
RS 路由不再报 `FamilyCapabilityError`。

**方法论要点**：
1. **"族名里的后缀"要从活 R 验证，不能从字面推断。**
   `o` 看起来像 original/untruncated，实际是 `mu.link` 的差别。
   一条 `identical(deparse(body(f)), deparse(body(g)))` 就能定案。
2. **"我们的族等于 R 的某个族"这种断言会把缺陷锁死。**
   既有测试写的是"`BCTUNTR` 的 `d` 对 `dBCT`"—— 缺陷存在时它一直是绿的。
   断言的参照物必须先独立确认是对的。
3. **R 的 `G.dev.incr` 不能向量化调用**：body 首行
   `z <- ifelse(nu != 0, ...)` 在标量 `nu` 下坍缩成长度 1，
   于是每个观测的 kernel 都在 `y[1]` 的 `z` 上求值 ——
   实测首元素正确（3.714485）、其余偏 up to 0.95。R 对照必须逐元素。
4. **差分对象是 `-g_dev_inc/2` 而不是 `g_dev_inc`**：差后者得到的是
   2 倍 score，表现为"每个 score 都错 rel 1.5"，看起来像全族皆错。
5. **支撑边界要单独测**：截断版的 `q` 从不越界，无截断版会 ——
   沿用截断版的 clamp 会静默返回一个错的概率。
5. `dnorm(x)` 漏掉 `1/sqrt(2*pi)` 会让 hazard 整体放大 2.5066。
   —— 这条是我自己的探针 typo，但足以伪装成"R 的导数错了"，排查时先怀疑自己。

**遗留**：~~`BCTo` / `BCCGo` / `BCPEo`（对应我们的 `BCTUNTR` / `BCCGUNTR` /
`BCPEUNTR` 克隆壳）尚未对齐~~ —— **D2.16 已完成**，但不是按上面那条
错误的对应关系做的，见 D2.16 段。

**D2.17 交付记录（2026-10-07）：`BCTo` / `BCPEo` 从"完全无法拟合"到可拟合**

D2.16 让 `o` 族登上了族表，但**它们根本跑不起来**。这次把整条链路打通。

**现象**：`BCTo` 与 `BCPEo` 的每一次 RS 周期都抛异常：

```
BCTo    ValueError: Betainc gradient with respect to a and b not supported.
BCPEo   NotImplementedError: Differentiation rule for 'igamma_grad_a' not implemented
```

两个族在族表上可见、能 `resolve`、但没有任何一次成功拟合。

**根因**：两处都在**独立重写的 log-pdf**里，不在密度本身。

| 族 | 自带的 log-pdf 碰到 | JAX 的限制 |
|---|---|---|
| `BCTo` | `betainc(tau/2, 1/2, x)` | 对 `a`/`b` **无梯度**，而 `a` 就是 `df/2` |
| `BCPEo` | `gammainc(1/tau, s)` | 对第一参数 `a` 无规则（`igamma_grad_a`）|

D2.13 给 `_log_pt_t` 加的 `custom_jvp` 能修**一阶**导，但 RS 的 Hessian
要对 JVP 再求一次导 —— 那会**重新进入** `betainc`。
**`custom_jvp` 修不了这一类问题：它只补一层。**

**修复**（两处，缺一不可）：

1. **密度去重**。R 的 `dBCTo` body 与 `dBCT` 逐字相同（D2.16 已用
   `identical(deparse(...))` 证明），`dBCPEo` 同 `dBCPE`。
   `_bcto_log_pdf` / `_bcpeo_log_pdf` 各自的 ~45 行重写改为委托
   `dpqr_functions.dBCT` / `dBCPE`（一个族一个密度，D2.15 的纪律），
   顺带白拿 D2.13 的 `custom_jvp`。
2. **`build_ad_family` 加 `score_overrides` / `hessian_overrides`**。
   有 override 的参数**跳过 warm-up**（warm-up 会对每个参数 trace
   `_vmap_hess`，即使结果丢弃 —— 那次 trace 就是抛错的地方）。
   两族从 `BCT()` / `BCPE()` 借入 **R 的闭式导数**。
   这不是绕过去 worse：对这个簇，R 的公式**就是** parity 目标
   （D2.15 已证明"改进"R 的粗糙二阶导会让 parity 变差）。

**验收**（40 点严格正数据，`y ~ x`，`c_crit = 0.001` 对齐 R 默认）：

| 族 | 我们 | 活 R | rel | converged |
|---|---|---|---|---|
| `BCTo` | 106.819450 | 106.822650 | 3.00e-05 | True |
| `BCPEo` | 106.764103 | 107.149897 | 3.60e-03 | True |

`d` / `p` / `q` / `G.dev.incr` 逐元素对活 R（`rtol=1e-9`）全过。

**一个自己踩的坑：把收敛判据不一致误判成"我们优化得更好"**

修完之后我看到"六个 Box-Cox 族全部 `converged=False`、
我们 deviance 一律比 R 低"，差点当成新缺陷。判定过程：

1. **同一密度互评**：把 R 的系数放进**我们**的密度算 `sum(G.dev.incr)`，
   得到 **106.839788**，与 R 报的 deviance 一致 → 目标函数相同，
   我们确实停在了另一个点。
2. **score 检验**：R 系数的 score 和**显著非零**
   （BCT: mu −0.145、sigma +0.116、nu +0.028）→
   **R 报的拟合点不是驻点**，R 提前停了。
   `tau` 方向 score 峰值仅 3.9e-9 —— 完全平坦，这就是 R 自己的 trace
   在停止迭代时 deviance 还在以 1.1e-4/周期下降的原因。
3. **真根因**：`gamlss.control` **没有 `delta` 参数**
   （签名止于 `save, ...`）。我传的 `delta=1e-8` 被 `...` **静默吞掉**，
   R 实际用默认 `c.crit = 0.001`；而我这边传的是 `c_crit=1e-8`。
   **两侧用了不同的停止规则**，偏差 1000 倍。

R 的 RS 判据是 `while (abs(G.dev.old - G.dev) > c.crit && iter < n.cyc)`
—— **绝对** deviance 变化；我们的是
`abs_change < tol or rel_change < rtol` —— 绝对**或**相对。

对齐到 `c_crit=0.001` 后**六族全部收敛**，rel 3.0e-05 ~ 3.6e-03。

**方法论要点**：

1. **`custom_jvp` 只补一层**。一阶能导 ≠ 二阶能导；
   只要 RS 走 Hessian，重新求导就会回到特殊函数。
   凡是"用 AD 派生"的族，log-pdf 都不能碰无二阶导的特殊函数。
2. **"我们比 R 好"默认当作可疑**（D2.15 的纪律，本轮再次生效）。
   判定手段是**同一密度互评** + **score 和是否为零**，
   而不是比较两边的 deviance 数字。
3. **"R 报的数"与"你传给 R 的参数"是两件事**。
   `...` 会静默吞掉不认识的参数名，然后你就在拿两套不同的配置做对比。
   **对照前先 `names(formals(f))` 核对，不要猜。**
4. **R 的收敛判据是绝对量**。parity 对比必须用同一判据；
   把自己的判据收紧 1000 倍再比"谁更优"是自欺。
5. **`fit[["mu.fv"]]` 是 n 元素 fitted values，不是系数对**。
   取 `[1]`/`[2]` 当截距/斜率重建 `mu(x)` 会得到 `mu(40) ≈ 71`
   （而 `y` 在 [0.4, 4]），把 deviance 抬到 464。
   这条我错了两次，第二次改用 `y ~ 1`（mu 是标量）才彻底消除歧义。
6. **`gamlss.family` 对象没有 `d`/`p`/`q` 槽**。
   `!is.null(o$p)` 为 TRUE 但 `o$p(...)` 报
   "attempt to apply non-function"（`$` 部分匹配到了 `parameters`）。
   dpqr 是命名空间级函数 `pBCTo` / `qBCTo`，第一参数名是 `q`。
7. **别用便宜的替身伪造失败模式**。我先试 `jnp.round` 当
   "一阶可导二阶不可导"的替身，实测它二阶返回 0 而不是报错 ——
   什么也没复现，测试会为错误的原因通过。
   改用真实的 `_log_pt_t`（grad1 = 0.0845，grad2 抛
   `Betainc gradient with respect to a and b not supported`，与线上逐字一致），
   并另加一条测试证明**不加 override 时同一构造确实会失败** ——
   否则"override 生效"和"override 被忽略"两种情况都测得过去。

**新增测试** `tests/test_boxcox_o_variants_fittable.py`（26 项）：
能拟合 / 拟合值有限 / 密度逐字等于基族 / 导数来自基族而非 AD /
override 是叠加而非开关 / warm-up 跳过 override /
不加 override 时 AD 确实失败 / link 表 / `d`/`p`/`q`/`gdev` 对活 R /
RS deviance 对活 R / "我们更低只是因为跑得更久"的推理被钉住。

**回归**：Box-Cox 簇 129 项 + dpqr/能力 28 项 + `build_ad_family`
相关 7 项全过。既有失败 `test_bccguntr_dpq_against_bccg_reference`
（rel 3.167e-05，`rtol=1e-05`）经 `git stash` 判定为**改动前既已失败**
（未含本次改动时报同样的数）。

**全族 parity 扫描现状**（146 族，`tools/rs_family_parity_scan.py`，
`--timeout 60`，D2.16 的欠账，本轮结清）：

| 关卡 | 通过 | 失败 | skip |
|---|---|---|---|
| `dpqr` | 60 / 95 (63.2%) | 35 | 47 |
| `deriv` | 130 / 139 (93.5%) | 9 | 7 |
| `rsdev` | 47 / 61 (77.0%) | 14 | 81 |

`rsdev` 的 14 个真实缺陷里，**`ST1` / `ST2` 报的
`Betainc gradient with respect to a and b not supported`
与本轮修的是同一类缺陷** —— 同一处 `ad.py` 的 override 机制应该能覆盖，
是下一个明确的入手点。其余：`SST` 11751 vs 135（rel 8.6e+01）、
`TF2` 3399 vs 144（rel 2.3e+01）、`ST3C` rel 1.8e-01、
`ZAPIG` / `ZAZIPF` deviance 非有限、`SI` 为 `nan`、
`GIG` 仍是 placeholder。81 个 skip 主要是 R 侧拟合不收敛
（`BCT` / `BCCG` 就在其中），属"探针覆盖不足"而非"实现有缺陷"。

**族覆盖已核对完整**：注册 146 族 = 扫描 146 族，**零遗漏**。
（核对时我一度以为 `BCTO` / `BCTUNTR` 没被扫到 —— 错在拿 R 风格的
混合大小写 `BCTo` / `BCTuntr` 去比对 Python 侧的**大写注册名**。
这个混淆本身就是 D2.16 那条教训的重演：**两套命名空间的名字不能直接对齐**。）
`BCT` / `BCCG` 的 `rsdev` 是 `skip` 而非 `fail`，原因是 R 侧在
`n.cyc=5000` 下仍报 "Algorithm RS has not yet converged" ——
**探针的收敛口径比实现更严**，同样是 D2.17 上面那条判据问题的延伸。

**D2.18 交付记录（2026-10-07）：`ST1` / `ST2` 从无法拟合到可拟合（RS 闭式导数）**

**现象**：扫描器对两族报

```
rsdev  fail  python raised ValueError: Betainc gradient with respect to a and b not supported.
deriv  fail  tau: raised ValueError
```

每次 RS 周期都抛异常，从未成功拟合。

**根因比 D2.17 更硬**：`_t_cdf_jax` 调 `betainc(df/2, 0.5, x)`，
而 `betainc` 对 `a`/`b` 无梯度。Box-Cox 簇里 D2.13 的 `custom_jvp`
至少修好了**一阶**导、只有 Hessian 重新进入；**这里连一阶都失败**，
所以 `custom_jvp` 无处可挂 —— 它是作用在"已经求出来的导数"上的。

**修复**：抄 R 的 8 个对角导数（`dldm`/`dldd`/`dldv`/`dldt`
+ `d2ldm2`/`d2ldd2`/`d2ldv2`/`d2ldt2`），经 D2.17 新增的
`score_overrides`/`hessian_overrides` 注入。**密度一行未改** ——
实测本来就对（6 档 profile rel ≤ 1.8e-15），只有导数是坏的。

**ST1 与 ST2 的差别不只是参数**：ST2 的 `w = nu*sqrt(lam)*z` 带 `lam`，
整个密度用 `df = tau+1`，于是 `dldm`/`dldd` 多出
`dwdz = nu*sqrt(lam)*(1 - lam*z^2/(tau+1))` 一项。
R 的 `lam` 也出现在 **ST1** 的 `dldm`/`dldd` 里（尽管 ST1 的 `w = nu*z`
用不到它）—— 这是 R 的真实代码，照抄不"整理"。

**三处刻意保留的粗糙**：

1. **四个二阶导全是 `-dldX^2` 并夹在 -1e-15**。这是粗糙的观测信息近似，
   不是真 Hessian。D2.15 的结论是：目标是"R 喂给 IRLS 的东西"，
   "改进"它会让 parity 变差。
2. **`dldt` 保留 R 的硬编码步长** —— ST1 在 `tau` 上用 0.001，
   ST2 在 `df = tau+1` 上用 0.0005。同一个导数、两个步长，是 R 的不一致。
3. **`dldv` 无保护地除以 `nu`**。`nu = 0` 时是 0/0，R 确实返回
   `NaN`（`d2ldv2` 返回 `NA`）。加"保护"会偏离参照物；
   R 自己的 `nu.initial` 是 0.1，优化器不会坐在那个边界上。

**验收**：

| 项 | 结果 |
|---|---|
| 8 个导数 × 6 档 profile 对活 R | ST1 worst 7.78e-10、ST2 worst 1.10e-09 |
| `d` 对活 R | 6 档 × 8 点 rel ≤ 1.8e-15（未改动，实为回归保护） |
| RS deviance `y ~ 1` | ST1 109.571554 vs R 109.573586（rel 1.85e-05）|
| | ST2 109.571770 vs R 109.575506（rel 3.41e-05）|
| ST1–ST5 `deriv` 关 | 5/5 pass（ST1、ST2 由 fail 转 pass）|

**发现 R 自身的一个缺陷：`ST2` 的 `dldt` 漏了 `dw/dtau`**

R 的 `j` 项在差分 `pt(w, df=tau+1)` 时**固定 `w`**，但
`w = nu*sqrt(lam)*z` 通过 `lam` 依赖 `tau`。活 R 上实测（y=0.3,
mu=sigma=1, nu=0.5）：

| tau | R 的 `dldt` | 中心差分（精确） | ratio |
|---|---|---|---|
| 0.5 | 0.929927 | 0.990414 | 0.939 |
| 1.0 | 0.298691 | 0.328777 | 0.909 |
| 2.0 | 0.083972 | 0.096027 | 0.875 |

**R 自己就偏低 6~12%**。扫描器的 `deriv` 关因此判 `ST2` fail
（差 9.2e-2）。

判定过程（不是猜的）：
- 中心差分在 h 从 1e-3 到 1e-7 **完全收敛**（值不变）→ 不是步长问题；
- ST1 在同一网格 rel 3e-7 通过 → 不是网格问题；
- 我抄的值与 R 的值 rel ≤ 1.2e-6 → 不是抄写错误。

**处置**：照抄 R（D2.15 纪律），在 `rs_family_parity_scan.py` 的
`_DERIV_TOLERANCE` 里按 `(族, 参数)` 登记 `("ST2","tau"): 5e-1`
并写明理由。测试里另有两条把这个判定钉住：
一条断言我们**等于 R 的错值**，一条断言我们**离精确导数 > 10%** ——
将来谁把它"改对"，前者会失败，那就是该重新审视 parity 的信号。

**方法论要点**：

1. **`custom_jvp` 只在"一阶能求出来"时有用**。二阶不可导的族里，
   若一阶也失败，就必须抄 R 的闭式导数，不要指望补丁。
2. **`ifelse(NA, a, b)` 在 R 里返回 `NA`，`jnp.where` 不一样**。
   R 的 `d2ldv2 = ifelse(< -1e-15, ., -1e-15)` 在 `dldv` 为 `NaN` 时
   返回 `NA`；`jnp.where` 会因为 `NaN < x` 为 False 而返回 `-1e-15`。
   必须显式传播 NaN。
3. **`build_ad_family` 的 `d` 会把长度 1 的数组塌成标量**
   （`_prepare_jax_args` 在所有参数 size 都是 1 时 `target_size = 1`）。
   传 `y = jnp.asarray([v])` 拿到的是 0 维结果 —— 值对、形状没法索引。
   与活 R 逐元素对照时要传标量。
4. **R 的 `dST1` 第一个形参名是 `x` 不是 `y`**
   —— 用 `dST1(xi, ...)` 位置传参没问题，写成 `dST1(y=xi, ...)` 就
   `object 'xi' not found`。
5. **"抄来的值与 R 一致"仍可能是错的**（ST2 的 `dldt`）。
   一致只说明抄对了，**对不对要单独判**。判据是拿活 R 的同一表达式
   做中心差分看谁离真值近。

**新增测试** `tests/test_st1_st2_parity.py`（22 项）：
`betainc` 一阶不可导的前提被钉住（否则"override 生效"和
"override 被忽略"都测得过去）/ 导数来自 R 表而非 AD（`__qualname__`
指向 `_make_st_derivative_tables`）/ override 对同模块的 ST3/ST4/ST5
**不生效**（它们仍走 AD）/ 密度与 8 个导数对活 R /
二阶导是带 floor 的负平方 / `nu=0` 时 `dldv` 是 NaN、二阶也是 NaN /
ST1≠ST2 且 ST2 的 `dldm` 多 `dwdz` 项 / ST2 的 `dldt` 缺陷被钉住 /
link 表 / RS deviance 对活 R。

**回归**：新测试 22 项、ST 族既有 compat 测试（除下述既有失败）、
能力矩阵 28 项全过。

**既有失败台账（新增一条）**：`test_r_consistency_st_compat.py` 的
`test_st1`–`test_st4` 各 2 个 profile 共 **8 个 SUBFAILED**，
报 `R Error: nu must be positive`（探针参数问题）。
`git stash` 判定：**改动前完全相同的 8 个失败**（含我未改的 ST3/ST4），
非本次引入。

---

#### D2.18 第二轮：`ST1`/`ST2` 的 `tau >= 1e6` 正态极限分支 + `ST1` 的 `rsdev` 定性

上一轮的两处待补项都在这里收口。全族扫描 v6 显示 `deriv` 由 130/139 升到
**132/139**、`rsdev` 由 47/61 升到 **48/61**（`ST2` 由"抛异常"转 pass），
但 `ST1` 的 `rsdev` 变成 fail：`deviance 140.111 vs R 138.379 (rel 1.251e-02)`。

**最终基线（v9，全族 146 族，21 分钟）**：

| 关卡 | v5 (D2.17) | v6 | **v9（D2.18 收口）** |
|---|---|---|---|
| `deriv` | 130/139 (93.5%) | 132/139 (95.0%) | **132/143 (92.3%)** |
| `dpqr` | 60/95 (63.2%) | 60/95 (63.2%) | **63/99 (63.6%)** |
| `rsdev` | 47/61 (77.0%) | 48/61 (78.7%) | **48/61 (78.7%)** |

分母变化是**探针覆盖抖动**（R 侧拟合报错 → `skip`），不是回退：
`deriv` 多 4 个 skip、`dpqr` 多 4 个 skip，`rsdev` 分母不变。
v6 → v9 的**唯一实质变化**是 `ST1` 的 `rsdev` **fail → warn**。
`rsdev` 剩 **12 个真实缺陷**（`ST1` 已退出）：

```
SST    11751    vs R  135.126  (rel 8.596e+01)
TF2     3399.12 vs R  144.172  (rel 2.258e+01)
BCPE      94.15 vs R  122.826  (rel 2.334e-01)
ST3C     168.41 vs R  142.293  (rel 1.835e-01)
NET      136.18 vs R  143.029  (rel 4.791e-02)
PE2      122.55 vs R  125.929  (rel 2.685e-02)
JSU      152.02 vs R  149.215  (rel 1.877e-02)
GPO      213.59 vs R  212.425  (rel 5.502e-03)
GIG      FamilyCapabilityError: placeholder nu
SI / ZAPIG / ZAZIPF   deviance 非有限（nan / inf / inf）
```

`ST1`/`ST2` 的 `dpqr` 仍 fail（`q` rel 1.336e-05 / 8.672e-06），
**v6 已存在、与本次改动无关**，属"影响正常使用"的候选（见下）。

**排查第一步就踩了自己的坑**：我先拿自己另一份探针数据去比扫描器记录的
R 数字（152.39 vs 138.379），差点得出"我们大幅偏差"的结论。
**扫描器 48 个族的样本来自同一个 `default_rng(20261006)` 共享流**，
某一族的样本取决于它前面排了多少族 —— 换种子重放得到的是另一份数据，
跨数据比较毫无意义。改法：直接调扫描器自己的 `_build_jobs` 把那份数据导出
（`tmp/d218_scan_data.json`），两侧用同一份。

**`ST1` 的 `rsdev` 差距：是 R 提前停止，不是我们的缺陷**

用同一份数据、两侧各自跑到收敛：

| | deviance | nu | tau（自然值） |
|---|---|---|---|
| R（`n.cyc=20`，第 5 次迭代即停）| 138.379017 | 0.4621 | 1.4e56 |
| 我们（`n_cyc=1000`）| **134.604639** | 10.4526 | 3.0e6 |

R 的停止原因可以直接看出来 —— 逐个迭代预算打印它的轨迹：

```
R ST1 n.cyc=3   dev=138.381491 it=3 conv=FALSE tau_eta=129.2813
R ST1 n.cyc=4   dev=138.379046 it=4 conv=FALSE tau_eta=129.2813
R ST1 n.cyc=5   dev=138.379017 it=5 conv=TRUE  tau_eta=129.2813
```

`tau_eta` 从第 3 次迭代起就**冻结在 129.2813 不动**，`nu` 也一直停在 0.462。
R 的 RS 判据是**绝对量** `abs(G.dev.old - G.dev) > c.crit`（`c_crit` 默认
0.001），deviance 在 138 量级时两次迭代的差就小于 1e-3，于是立即停机，
**`nu` 方向根本没被探索过**。我们继续下降 1000 次，找到低 3.8 的解。

**判定手段是"用 R 的密度评估我们的系数"**（D2.15 立的规矩，
D2.17 再次用过）：把我们的自然尺度系数代进 R 自己的 `dST1`：

```
R's density at OUR params = 134.604639   (we reported 134.604639, diff 1.315e-07)
```

**逐位吻合**（1.3e-07 是 R 自身 `ifelse` 向量化 quirk 的量级）。
密度实现与 deviance 计算都与 R 同一个目标函数，**我们确实找到了更优解**。
`ST2` 同样如此（我们 130.555409，R 133.063980，R 密度在我们系数上给
130.555409）。这与 D2.15 的 BCPE（161.9236 < R 162.0165）是同一类：
弱识别的平坦区。

**处置**：在 `rs_family_parity_scan.py` 新增 `_RSDEV_R_STOPPED_EARLY` 登记，
只对**我们更低**的方向开口，且要求两侧数字仍与登记值一致 —— 数字一旦漂移就
退回 fail 并提示重新查证，防止后来的回归借着旧备注混过去。
另外该族按**登记时的预算**（`n_cyc=1000`）重跑，否则是拿我们 20 次的平台
去比 R 的 5 次停机点，量的根本不是同一件事。

**顺带修掉的真实缺陷：`dST1`/`dST2` 缺 `tau >= 1e6` 正态极限分支**

查 R 源码时发现（`print(dST1)`）R 的密度是**两套 kernel 加一个开关**：

```r
loglik1 <- pt(w, df=tau,   log.p=TRUE) + dt(z, df=tau,   log=TRUE) + log(2) - log(sigma)
loglik2 <- pNO(w, 0, 1, log.p=TRUE) + dNO(z, 0, 1, log=TRUE)   + log(2) - log(sigma)
loglik  <- ifelse(tau < 1e+06, loglik1, loglik2)
```

`ST3`/`ST4` 都有这个开关，**`ST1`/`ST2` 是仅剩的两个漏项**。
它不是装饰：**R 自己的 `ST1` 最优解就落在 `tau = 1.4e56`，正在这个分支里**，
所以漏掉它意味着我们的密度**在 R 的解处就是错的**。

活 R 两侧对照（`y=0.35, mu=1.2, sigma=0.8, nu=0.5`）：

| tau | `dST1` | `dST2` |
|---|---|---|
| 999999（t kernel）| -1.779029676131917 | -1.779029636275371 |
| 1e+06（正态极限）| -1.779029378512898 | -1.779029378512898 |
| 1.4e+56 | -1.779029378512898 | -1.779029378512898 |

分支两侧差 3e-07，**可观测** —— 这是判据成立的前提。
两族在 `tau >= 1e6` 之后完全相同，交叉印证。

修复后 ours vs R：t kernel 侧 ≤ 1.7e-09（`_t_cdf_jax` 的 `betainc`
在 `df ~ 1e6` 处精度下降，**既有**限制，与本分支无关），
**正态极限侧 4.44e-16**。

**但这个修复没有改变 RS deviance**（ST1 `n_cyc=20` 仍是 140.110594）。
原因是 `df = 4.4e6` 的 t 分布**数值上已经收敛到正态**，R 的显式分支只是把
极限做精确、不改变数值。修复的价值是**对齐 R 在 `tau` 任意取值下的密度**，
以及让"我们的密度在 R 的解处也是对的"这句话成立。

**`_norm_logcdf` 用 `log_ndtr` 而非手写 `log(0.5*erfc(...))`**

先写了 `log(0.5*erfc(-x/√2))`，实测在 `x = -3` 处与 `log_ndtr` 差 **3.90**
（`-2.706` vs `-6.608`），且开不开 x64 都一样 —— 这不是舍入，是公式写错。
`log_ndtr` 正是 R 的 `pNO(log.p=TRUE)` 走的同一条路，且实测
`grad` 与 `grad(grad)` 都有限，能过 `fitting_cg` 在 `jit` 内对
`family.d(..., log=True)` 求导那一关。**这一条是逐个测出来的，不是类推。**

**新增测试**（`test_st1_st2_parity.py` 由 22 项增至 **26 项**（18 个测试函数，8 个参数化））：

- 正态极限分支两侧密度对**活 R**（查询而非硬编码 —— 抄错的 16 位常数
  会变成一个**通过的**测试，那是 parity 测试最不能有的失败模式）；
- `test_normal_limit_branch_is_not_inert`：断言两分支**可区分**。
  否则删掉分支后上面那条仍然全绿 —— 它只探 R 自己不会在两侧都访问的点；
- `test_st1_and_st2_agree_above_the_switch`：R 在 `tau >= 1e6` 后两族
  逐位相同，差了就说明只接进了一个族。

**D2.18 方法论补充**：

6. **跨数据比较 deviance 毫无意义**。扫描器所有族的样本出自同一个
   `default_rng` 共享流，要复现必须调它自己的 `_build_jobs`，
   不能另起种子"造一份差不多的"。
   **这条我踩了两次**，第二次更隐蔽：为"快点验证"跑
   `--families ST1,...,ST5` 子集，结果 R 基准从 138.379 变成 142.148 ——
   **子集改变了 rng 流的位置**。**只有全族扫描的数字与历史可比。**
7. **`rel` 偏差的方向决定它是缺陷还是发现**。只有**我们更高**才是明确的
   缺陷；**我们更低**要先怀疑 R 提前停止，判据是用 R 的密度互评
   + 打印 R 的逐迭代轨迹看哪个方向被冻结。
8. **`coefAll()` 在 R 里返回 link 尺度，不是自然尺度**。
   R 的 `ST1` 最优 `tau_eta = 129.28` 对应自然值 `1.4e56`；
   当成自然尺度会把 `sigma = -0.0239`（负数！）送进密度，静默得到 NaN。
   两边都用 `[[par]].coefficients`（link）与 `[[par]].fv`（自然）成对打印，
   一次定死。
9. **`git stash` 造对照时，别让后台任务正在读磁盘**。
   我在全族扫描跑到一半时 stash 了工作区，那一轮结果混了旧代码，
   只能作废重跑（21 分钟）。**stash/pop 前后确认没有后台任务在跑**。
10. **分支/开关类修复必须证明它可观测**。正态极限分支两侧密度只差
    3e-07 —— 若两侧相同，这条测试删掉分支后仍然全绿，等于没测。

#### D2.21：`BI`/`DBI` 的 rel 1.25e+90 —— 是**扫描器**在撒谎，不是族有缺陷

**现象**（全族扫描 v12）：

```
BI    dpqr  fail  d: max rel 1.246e+90; p: max rel 9.437e-01; q: 1.000e+00
DBI   dpqr  fail  d: max rel 1.246e+90; p: max rel 9.437e-01; q: 1.000e+00
```

全registry 最严重的条目。逐层查证后：**两个族的 d/p/q 全部正确**，
缺陷在 `tools/rs_family_parity_scan.py`。

**rel 大到荒谬时先怀疑探针**：1e90 不是数值分歧，
是"参数送错了"的签名。真族缺陷不会有这种量级。

**根因两处，必须叠加才产生 1e90**：

1. **位置传参把 `bd` 和 `mu` 送错了**。R 的签名是 `dBI(x, bd, mu, log)`
   （**`bd` 在 `mu` 前**），我们的是 `dBI(x, mu, bd, log)`。
   `_python_dpqr` 按位置传 `[x, bd_array, *params]` →
   **`mu` 收到 10**、`bd` 收到概率 → `mu` 被 clip 到 `1 - 1e-10`，
   二项系数爆炸。R 侧本来就按名传（`names(aa) <- argnames`），
   **Python 侧没有** ——两边可以一个对一个错。

   ```
   位置传参  [3.2e-03, 2.0e+06, 7.5e+24, 9.2e+33, 8.5e+52, 1.1e+62, 1.1e+81, 1.2e+90]
   按名传参  [0.0563, 0.0980, 0.2213, 0.2307, 0.2307, 0.2213, 0.0980, 0.0563]
   ```

   末尾那个 `1.2e+90` 与报出的 `1.246e+90` **精确吻合**。

2. **离散族的 `y` 网格含非整数**。`calibrate` 的二项候选是
   `np.arange(n) * (bd / (n - 1))`，`n=8, bd=10` 下给出
   `0, 1.43, 2.86, 4.29, 5.71, 7.14, 8.57, 10`。
   R 的 `dbinom`对非整数 `x` 返回 **`-Inf`** 并打印
   `non-integer x = 1.428571` → **8 点里 6 点在 R 侧不可评估**；
   我们的 `dBI` 用 `jnp.round(x)` 照算。

   **两边对"函数是否有定义"都不一致的网格点，不可能产出有意义的 parity 结论。**

**叠加关系**（这部分值得单独记）：

| |非整数网格 | 整数网格 |
|---|---|---|
| 错位调用峰值 | **1.2e+90** | 1.4e+02 |

"两个 bug"最自然的读法是"任一即可解释症状"，**这里不是**：
只修网格会留下"错但看着合理"的数字（比两个都错更难查），
只修传参会留下 R 侧无法评估多数网格点。

**修复**：`_python_dpqr` 改为 `fn(**kwargs)`，参数名取自
`fam.parameters`（带 `TypeError` 兜底回退到位传）；
`calibrate` 的二项候选改为整数网格 `np.arange(n) * step`。

**验收**（全族扫描 v12）：

|关卡（可测口径） | D2.20 之后 | D2.21 之后 |
|---|---|---|
| `deriv` | 134/144 (93.1%) | 134/144 (93.1%) |
| `dpqr` | 63/100 (63.0%) | **65/100 (65.0%)** |
| `rsdev` | 50/60 (83.3%) | 50/60 (83.3%) |

```
BI    dpqr  fail -> pass
DBI   dpqr  fail -> pass
```

**零回退**。两族的 `deriv` 保持 `pass`。

**新增测试** `tests/test_parity_scanner_probe_contracts.py`（7 项）：
这个文件测的是**扫描器**而非族。按名传参后密度落在 [0,1]、
`k=0` 项与闭式 `(1-mu)^bd` 吻合到 1e-12、
网格整数性（对活 R 的 `-Inf` 边界）、`dBI` 对整数计数与闭式
`C(n,k)p^k(1-p)^(n-k)` 逐点吻合、错位调用返回**正的**"对数密度"
（+139.03，任何对数密度都不可能为正）、
以及非整数 `x` 的取整行为被明确记录为**决策**而非疏忽。

**本轮我自己的新测试翻了四次**（把 plain density 当 log、
对已是 density 的值又套 `np.exp`、把 `|d|<1` 用在 log 上、期望值算错）——
**每次都是"没看实际数值就下断言"**。D2.19 立的"先查探针"纪律
同样适用于自己刚写的测试。

**遗留**：`BI`/`DBI` 的 `rsdev` 仍 `skip`，原因是
`R fit error: y values must be 0 <= y <= 1` ——
扫描器的 `_fit_data` 不支持 R 的二项 `cbind(成功, 失败)` 两列响应
（D2.10 已查明该约定）。这属于**新增功能**而非缺陷修复，
列为下一项。

**D2.21 方法论要点**：

1. **rel 大到荒谬时先怀疑探针**。1e90 是"参数送错了"的签名，
   不是数值分歧。真族缺陷不会有这种量级。
2. **R 的形参顺序要逐族查**，不能假设与 Python 一致。
   R 侧按名传参而 Python 侧按位传参，**两边可以一个对一个错**。
3. **离散族网格必须整数**。两边对"函数是否有定义"不一致的网格点，
   不可能产出有意义的 parity 结论。
4. **多个缺陷可能必须叠加才产生症状**。只修一个会留下"错但看着合理"的
   数字，比两个都错更难查。
5. **测试失败时先打印实际值，不要先改断言**。这条纪律对自己新写的测试
   同样成立。

#### D2.22：`BI`/`DBI` 的 RS 拟合把成功计数当比例（dev 8566.07 → 与 R 逐位一致）

**现象**：

```python
gamlss("y ~ x", family="BI", data={"y": succ, "x": x, "bd": bd}, method="RS")
# converged=True, deviance = 8566.07433006277
```

R 在同样 60 个观测上是 **217.97377537960233**，且真值处的 deviance 只有
221.6 —— 我们比真值还高 38 倍。同时伴随
`eta clipped for parameter 'mu': 60/60 values out of [-15.0, 15.0]`。

**三处独立缺陷叠加**，修任何一处都仍然是坏的：

| # | 位置 | 缺陷 |
|---|---|---|
| 1 | `_initial_mu_beta` | `BI` 分支写 `logit(clip(y,1e-6,1-1e-6))` —— **y 是成功计数不是比例**。R 的 `BI()$mu.initial` 是 `mu <- (y + 0.5)/(bd + 1)`（活 R 求证），既缺 `/bd` 也缺半计数修正。同文件 `BB` 分支有 `/bd` 但同样缺半计数修正 |
| 2 | `rs_algorithm.rs_fit` | `other_params`（喂给 score/Hessian 的 `context`）与 `dev_kwargs`（喂给 `g_dev_inc`）都只从 `family.parameters` 构造，而 `estimable_parameters` **刻意排除 fixed parameters** → `bd` 从未到达任何一处。`g_dev_inc` 的 `bd` 默认 `None`，`None` 选**Bernoulli** 分支 `y*log(mu)+(1-y)*log(1-mu)`，把计数 7/10 读成 7 |
| 3 | `warm_start_policy.FamilyHeuristicInit` | 对 BI 给出 `fitted["mu"] = mean(y) = 5.78`。`mu` 是比例而 `y` 是计数，`logit(5.78)` 是 NaN → link clip 到 +15 → **第一个 IRLS 步之前**就从 `mu=1.0`、dev 21360.6 起步 |

**诊断轨迹**（每一步只解一层，故可分离）：

```
mu = mean(y) = 5.78        -> logit clip     -> mu = 1.0,  dev = 21360.6
mu = 1.0,  bd 缺失         -> Bernoulli dev   -> 在 mu=1-3e-7 与 3e-7 之间两周期震荡
mu = logit(计数)           ->                -> dev = 8566.07, converged=True（报出的 bug）
三处全修                   ->                -> dev = 217.973775379603 == R，逐位一致
```

**症状学上的教训**：每一个中间阶段 `converged` 都是 `True`。`converged`
报告的是步长塌缩，不是目标函数到达极小；在两个被 clip 的 eta 值之间来回
震荡的失控过程，从外面看就是完美收敛。

**验收**（同一份 fixture，`default_rng(7)`，60 观测，bd=10）：

| | 修复前 | 修复后 | R | rel |
|---|---|---|---|---|
| `BI` deviance | 8566.074330 | **217.973775379603** | 217.97377537960233 | 逐位 |
| `DBI` deviance | 8566.074330 | 217.973775379603 | 217.95942206518868 | 见下 |

`DBI` **尚未 parity**：R 的 `DBI` 估计两个参数（`mu, sigma`，
`sigma` 系数 0.020070403755973468，dev 217.95942206518868），我们的
`DBI` 是单参数壳（`parameters == ("mu",)`）因而退化成 `BI`。这是**族
实现的结构缺口**，不是本轮缺陷；已登记为下一项，并用
`test_dbi_reproduces_our_bi_shell_not_rs_two_parameter_dbi` 钉住现状
（连同"R 的 DBI 是另一个数"这句一起写进断言，免得后人把巧合读成 parity）。

**扫描器侧（D2.21 遗留项解除）**：`BI`/`BB` 的 `rsdev` 从 `skip` 变`pass`。

```
BI    deriv pass   dpqr pass   rsdev pass   251.671 vs R 251.671
BB    deriv skip   dpqr skip   rsdev pass   203.879 vs R 203.879
DBI   deriv pass   dpqr pass   rsdev fail   FamilyCapabilityError（compat alias 门禁，正确工作）
```

`_fit_data` 现在为 `fixed_parameters` 含 `bd` 的族发出 R 唯一接受的
两列响应，`r_parity_batch.R` 按`payload$response_formula` 拼
`cbind(y, fail)`，我们这一侧从同样两列重建 `bd`（两边必须最小化**同一个**
二项似然）。`DBI` 的 `fail` 是门禁在正确工作 —— 它在
`family_capabilities` 里注册为 compat alias，`gamlss()` 不透出
`allow_compat_alias`，测试因此走 `rs_fit` 而不是绕过 fail-closed 准入。

**测试**：`test_bi_dbi_binomial_parity.py` 19 项（含活 R 复算参考值）、
`test_parity_scanner_probe_contracts.py` 14 项。相关回归 68 + 86 + 82 项
零失败；`test_r_consistency_dbi` 的 3 个失败经 `git stash` 造对照确认为
**既有失败**（R 侧 `x must be <= the binomial denominator 1`，期望过时）。

**本轮自己的测试失误**（四次，全部源于没先看实际值）：

1. `convergence_status()` 在当前构建上返回**布尔**不是词汇字符串
   （词汇在 `termination_reason` 上）—— 与 D2.2 记的"一等等待词汇表"
   有出入，已在断言里注明。
2. 以为 Bernoulli 分支对 `y=7` 给出**负**增量 deviance。实际
   `(2y-1)*log(0.5)`：y=7 是 +1.386，y=9 才是 −21.73。负 deviance在
   `y=9` 上才出现 —— 这个观察反而成了最可诊断的信号。
3. `gamlss()` 没有 `allow_compat_alias` 形参（只在内部 helper 上）。
   一开始按 DBI 也走 `gamlss()`，spy 记录 0 次调用 —— **长得和"bug 复现"
   一模一样**。
4. 用 DBI 的 R 参考值断言我们的 DBI：忘了先确认我们的 DBI 有没有
   `sigma` 参数。先 dump 双方参数列表再断言，才把"族缺口"和"拟合缺陷"
   分开。

**遗留**：
- `DBI` 缺 `sigma` 参数（族实现缺口，非拟合缺陷）。
- `DBI` 的 `rsdev` 因 compat alias 门禁 fail；解法是给它一条生产路线或
  在 `gamlss()` 透出 `allow_compat_alias`，属准入设计变更。
- `BE`/`ZABB`/`ZIBB`/`ZINBI` 的 `mu.initial` 与 R 不同（活 R 已dump：
  R 是 `(y + mean(y))/2` 与 `rep(0.5, length(y))`，我们是
  `logit(clip(y))` 与 `(y+0.5)/(bd+1)`）。`BE` 的 `rsdev` 目前 pass，
  说明 IRLS 能从不同初值收敛到同一点 —— 但初值不同仍是 parity 缺口，
  列为下一批。

**D2.22 方法论要点**：

1. **`converged=True` 不等于到达极小**。它报告步长塌缩。两周期震荡的
   失控过程看起来就是完美收敛。凡是 `converged=True` 但目标值明显
   不合理，先怀疑震荡。
2. **固定参数是似然的一部分，必须像估计参数一样传下去**。
   `estimable_parameters` 排除 fixed 参数是对的（不进设计矩阵），
   但 `context` / `dev_kwargs` 从 `family.parameters` 取就会连带
   把它排除 —— 这个耦合没有任何测试覆盖，改了没人知道。
3. **"响应约定"要在**入口**读，不在密度函数里读**。`dBI` 完全正确，
   `g_dev_inc` 也基本正确，错的是"y 是计数还是比例"这个约定在三个地方
   各被重新解释了一次。
4. **`skip` 与 `fail` 的差别要在**工具**里修**。族是对的、探针是错的
   这类问题（D2.21）会让最贵的资产（已验证的密度代码）看起来该被重写。
5. **分不清是族缺口还是拟合缺陷时，先 dump 双方的参数列表**。
   我们的 `DBI` 与 R 的 `DBI` 连参数个数都不同 —— 这个事实一查就清楚，
   但我是在写断言失败之后才去查的。

#### D2.23：`DBI` 真正的双参数似然（Borg 1988）—— 从 `BI` 转发壳到三关全绿

**现象**：

D2.22 结束时 `DBI` 是 `BI` 的转发壳：`parameters == ("mu",)`、没有 `sigma`、
同一个 kernel。它报出 `BI` 的 deviance `217.973775379603`，而 R 的 `DBI` 报
`217.95942206518868`；并且因为登记在 fail-closed 的 `compat_alias` 名单上，
`rsdev` 关根本不会跑它（`FamilyCapabilityError`）。

```
DBI   deriv skip   dpqr pass   rsdev fail   FamilyCapabilityError（门禁正确工作）
```

**根因一：`C` 是有限求和，不是数值积分**。

`dDBI` 是二项 pmf 被ratio 项的幂倾斜、再用 `C(mu, sigma, bd)` 归一化。
`C` 的入口名（`.C("getBI_C2", ...)`）和 "C" 这个字母都让人以为是数值积分。
**它不是。** `R/gamlss.dist/src/getBI_C2.c`（65 行，已 vendored 进本仓库）
就是完整定义，`j = 0..bd` 的有限和。

三个闭式猜测全错（Borg 的 beta 混合、`log B(a,b)` 形式、log-difference），
**误判的线索**在第一张探针网格上：`mu ∈ {0.2, 0.5, 0.8}` 上 `C` 看起来
与 `mu` 无关 —— 那张网格**关于 0.5 对称**，而 `C` 也对称。扩到
0.01/0.99 后依赖性立刻显现。

> **对称于被探测点的探针网格会掩盖你要找的效应。**

**根因二：`GetBI_C` 先做 `sigma <- 1/sigma`**。

再调 C 的 `getBI_C2`，其 `ss` 写成那个值的函数。我们的 `_dbi_log_c` 在
边界做一次求逆。搞反了会得到一个"看起来合理但不是 R 的"的密度。

**根因三：`dDBI` 在 `|sigma - 1| < 0.001` 时切到纯 `dbinom`**。

这不是优化，是让 `C(1) = 0` 精确而非 `-1.4e-16`。而 R 的
`sigma.initial` 是 `rep(1.1, length(y))`，所以 RS 拟合会**穿过**这个分支。

**根因四（工具侧）：形状判据只看 `bd`**。

`_dbi_log_c` 原先按 `scalar = jnp.ndim(bd) == 0` 决定是否 squeeze。但扫描器的
`calibrate` 传的是**标量** `_PROBE_BD = 10.0` 配**长度 n 的参数向量** →
squeeze 在 `(n,)` 结果上触发 → `cannot reshape array of shape (n,) into shape ()`。

**这个异常读起来完全不像形状 bug**：它到达 `gate_derivatives` 时表现为
*每个* 探针候选都抛异常，`calibrate` 返回 `ok=False`，族被记成
`deriv: skip` —— 一处看起来恰似"没有可用支撑网格"的**静默覆盖率损失**。
修法是按**三个参数的广播形状**决定返回形状，全标量才返回标量。

**根因五（工具侧）：`_DERIV_TOLERANCE` 缺一条登记**。

修好广播后 `deriv` 从 `skip` 变 `fail`，唯一失配点是 `sigma = 1.0` ——
恰好落在根因三的分支窗口内。活 R 求证：

| sigma | R `dldd` | R 自己密度的中心差分 |
|---|---|---|
| 0.9995 | -0.480365218462 | **0.000000000000** |
| 1.0000 | -0.480176057746 | **0.000000000000** |
| 1.0005 | -0.479987093435 | （窗口内） |
| 1.0020 | -0.479421170164 | -0.479421170829 |

窗口内 log-density 关于 sigma **精确为常数**，任何中心差分必然是 0；
而 R 的 `dldd` **不跟随分支**。**R 与它自己不一致。** 我们逐位复现了
R 的 `dldd`（实测差 1.24e-08，R 侧 `numeric.deriv` 精度）。按 D2.13
Box-Cox 同一条纪律，登记容差 + 活 R 证据，**不改我们的代码**。

**活 R 参考（全部求证，非硬编码）**：

```
DBI()$parameters   mu, sigma          DBI()$d        NULL
DBI()$p  list(mu=TRUE, sigma=TRUE)   # 参数化标记，不是函数
DBI()$q        NULL                 pDBI/qDBI 在命名空间但未挂到族上
DBI()$mu.link  "logit"              DBI()$sigma.link "log"
DBI()$sigma.initial  expression(sigma <- rep(1.1, length(y)))
DBI()$mu.initial     expression({ mu <- (y + 0.5)/(bd + 1) })
DBI()$G.dev.incr     -2 * dDBI(y, mu, sigma, bd, log = TRUE)
DBI()$d2ldm2        d2ldm <- <dldm>; -dldm^2;  地板 -1e-15
DBI()$d2ldd2        同上（对 dldd）
```

**验收**：

| 项目 | 结果 |
|---|---|
| `C` 对活 R（224 点，bd 2/5/10/20 × mu × sigma） | worst diff **1.52e-14** |
| log 密度对活 R | worst diff **4.26e-13** |
| 密度归一化（每点 $\sum_x d(x) = 1$） | 全部 `1.000000000000` |
| `pDBI` CDF vs R | 12 位逐位吻合 |
| `qDBI` quantile vs R | 逐位吻合（`[2,3,4,5,6,10]`） |
| score 对**我们自己**密度（FD） | 1e-09 .. 1e-10 |
| score 对活 R | worst ABS `dldm` 2.454e-05 / `dldd` 3.432e-06（R 的 `numeric.deriv`） |
| Hessian | `-score²` 地板 `-1e-15`（Fisher scoring，不是 `d2logL`） |
| capability | `ImplementationStatus.EXPERIMENTAL`（脱离 `COMPAT_ALIAS`） |
**"族变真"会连带打四处断言，全部是期望过时、无一是安全边界受损**：

| 位置 | 原假设 | 处置 |
|---|---|---|
| `test_dbi_compat` | `parameters == ("mu",)` | 改为 `("mu","sigma")` + links |
| `test_bi_dbi_binomial_parity` | 钉住"DBI 还是壳" | 改写为 `test_dbi_reaches_rs_two_parameter_parity`（D2.22 的失败信息本身就是移交说明） |
| `test_family_capabilities` ×2 | 拿 `DBI` 当 compat_alias 代表样本 | 换成 `DBURR12`（仍在名单），**不删断言** |
| `test_release1_family_capability_gate` | 断言 `UNSAFE` 里**两种状态都在场** | 改为断言"每族属于二者之一" |

最后一处值得单说：DBI 离开 `COMPAT_ALIAS` 后**不再属于 `UNSAFE`**，而
`EXPERIMENTAL` 在 `require_production_family_route` 上**没有 opt-in**
（该函数刻意不提供迁移开关）→ 三条生产路由全部拒绝，代码
`experimental_requires_opt_in`。**"不再是兼容别名"与"可进生产"是两句话**，
所以补了 `test_dbi_stays_fail_closed_after_leaving_the_compat_alias_list`
逐路由钉住，并用 `inspect.signature` 断言 opt-in 不存在 —— 用
`except TypeError` 探测的话，将来有人**故意**加上这个开关时，测试会以错误的
理由继续绿。

**全族扫描 v13 → v14**（`--timeout 60`，146 族）：

| 关 | v13 | v14 | 回退 | 增益 |
|---|---|---|---|---|
| `deriv` | 134 pass / 10 fail / 2 skip | 134 / 4 / 8 | **0** | 0 |
| `dpqr` | 65 pass / 35 fail / 46 skip | 60 / 34 / 52 | **0** | 0 |
| `rsdev` | 50 pass / 10 fail / 85 skip / 1 warn | 53 / 9 / 83 / 1 | **0** | **3**（`BI`/`BB`/`DBI` skip→pass）|

`deriv` 与 `dpqr` 的计数下降**全部**来自 `--timeout 60`：6 个慢族
（`SI`/`SICHEL`/`ZABNB`/`ZASICHEL`/`ZIBNB`/`ZISICHEL`）被杀，三关一起消失
（`dpqr` 键整个缺失）。用 300 秒预算补跑，18 项里 **17 项逐项相同**；
唯一差异 `SI rsdev`（`fail(nan)` → `skip(R未收敛)`）是子集运行改变 rng 流
位置所致，全量运行下与 v13 一致。**代码零回退。**


**RS 拟合**（`default_rng(7)` fixture，60 观测，bd=10）：

| | 我们 | R |
|---|---|---|
| deviance | 217.95942032221294 | 217.95942206518868 |
| rel | **8.0e-09**（我们更低） | |

我们更低 → 先怀疑 R 提前停（D2.18 判据第 2 条）。两步判定都做了。
**注意**：这个 217.959 的对比跑在 `default_rng(7)` 的自有 fixture 上；
扫描器 gate 比较的是另一份数据（见下），两者不可混用。

**扫描器 gate 实际比较的那份数据**（`_build_jobs(["DBI"], n=8)`，48 观测，bd=10）：

```
ours                                            208.042407377517236
R's dDBI evaluated at OUR coefficients           208.042407377517264
R's gamlss() deviance                           208.043104514376921
R's dDBI at R's own fv                          208.043104514376921
```

R 的密度把我们的数原样返回（差 2.8e-17，末位）→ 两侧最小化**同一目标
函数**。R 加预算不动：`n.cyc = 20 / 40 / 100 / 300` 全部报
208.043104514376921。我们低 3.3e-06 rel 是 R 绝对 `c.crit` 提前停，**不是缺陷**。

**复现这份数据要同时对齐三件事，錯任何一件都会得到一个"看起来合理但
属于别的问题"的数**（本轮为此走了一整段弯路）：

| # | 事实 | 传错的后果 |
|---|---|---|
| 1 | `_build_jobs` 共享一条 `default_rng(20261006)` 流（D2.18 判据 1） | `--families DBI` / 39 族前缀 / 146 族全量是三份不同数据 |
| 2 | `main()` 的 `--n` 默认**8**，不是 `N_OBS`。`n` 传给 `_family_grids` → 决定**响应网格** → 决定**哪 48 个观测**（不是观测**数量**，那恒为 `N_OBS=48`） | dev 113.69 而不是 208.04 |
| 3 | gate 用 `sigma_formula="~1"`、`max_iter=N_CYC`（20） | 与公开默认不同，数值不同 |

实测三个"看起来都像证据"的数：113.69（`n=N_OBS`）/ 137.57（单族+默认拟合）
/ 138.44（单族+gate 配置），**没有一个是 gate 真正比较的那个**。
本轮把这三点做成 `_gate_sample()` 的断言。

### 一个新的 R 探针坑：两参数族的 `coef()` 是**扁平向量**

D2.18 判据第 3 条记录的是"`coefAll()` 返回 link 尺度"。DBI 又暴露了
**结构**层的同一类错位：

```r
length(coef(fit))                  # 2  <- 只回了 mu 的 (Intercept), x
fit$coefficients                   # NULL
fit[["parameters"]][["mu"]]$coefficients   # subscript out of bounds
fit$mu.coefficients                # <- 顶层按参数名平铺，DBI 的正解
fit$sigma.coefficients             # 长度 1（sigma ~ 1 的截距，link 尺度）
fit$mu.fv / fit$sigma.fv           # 48 个拟合值，不是系数
o$dDBI                             # NULL（族上没挂 d），要用 gamlss.dist::dDBI
```

即：**旧版 gamlss 的 `fit[[par]]$coefficients` 形状在当前版本不存在**，
顶层就是 `<par>.coefficients`。取错会得到 `subscript out of bounds` 或
`sprintf` 收到无 `dimnames` 的数组——两者都读起来像 R 侧的问题。

**扫描器三关**（D2.23 前 → 后）：

```
DBI   skip fail pass  ->  pass pass pass
```

`dpqr` 的 pass 依赖一条工具侧改动：R 的 `DBI()$d` 是 `NULL`（族上没挂
`dDBI`），所以 R 侧缺对手时记 pass + note 而不是 fail —— 否则会把
"R 没提供" 记成 "我们不一致"。

**D2.23 方法论要点**：

1. **探针网格对称于被探测点时会掩盖你要找的效应**。`mu ∈ {0.2,0.5,0.8}`
   上 `C` "与 mu 无关" 是网格的产物，不是 `C` 的性质。发现方式是**扩到
   不对称网格**，而不是加精度。
2. **C 入口的名字不承诺它的数学形式**。`.C("getBI_C2")` 读起来像数值积分，
   源码是 65 行有限和。三个闭式猜测的共同错误是"信名字、不读源码"。
3. **形状判据必须看全部参数，不能只看一个**。按 `bd` 判标量，在
   "标量 `bd` + 向量参数"这个**扫描器实际使用的组合**上抛 reshape 异常，
   而它到达工具层时表现为"所有候选都失败" → 静默 `skip`。
   **一个坏形状可以伪装成一族没有支撑网格。**
4. **R 与它自己不一致的地方，要复现 R，不是复现自洽**。`dDBI` 分支而
   `dldd` 不分支，窗口内 R 自己密度给 0 而 R 的 score 给 -0.48。
   我们逐位抄了 R，登记 `("DBI","sigma")` 容差 + 活 R 证据表。
   与 D2.13 Box-Cox 完全同构。
5. **复现 gate 比较的数据要对齐三件事：族列表、网格 `n`、拟合配置**。
   只对齐族列表不够——`main()` 的 `--n` 默认是 **8**（不是 `N_OBS`），
   它经 `_family_grids` 决定响应网格，进而决定**哪 48 个观测**。本轮实测
   三个"看起来都像证据"的数（113.69 / 137.57 / 138.44）全都不是 gate
   真正比较的 208.04。**一个复现脚本错了不会报错，它只会给你另一个
   问题的答案。**
6. **前缀等价性可以省时间，但只在明确条件下**：`[f for f in families if
   f <= X]` 与全量对 X 之后的族**完全等价**（rng 状态只取决于 X 之前抽过的
   量），实测 DBI 的 deviance 在 39 族前缀与 146 族全量下都是
   113.6901910280584。但这只在"你比的是 X 自己"时成立；要比 gate 的数
   就得连 `--n` 和拟合配置一起对齐。
7. **R 拟合对象的结构会随参数个数变**。两参数族 `coef(fit)` 是扁平向量
   （只回 mu），`fit$coefficients` 是 NULL，`fit[[par]]$coefficients`
   根本不存在——正解是顶层 `fit$mu.coefficients`。这与 D2.18 判据第 3 条
   是同一类错位，只是发生在**槽位名**而不是**尺度**上。

#### D2.24：把"停在非驻点"变成可观测的量（BCPE `rsdev` rel 2.334e-01 的正确归因）

**现象**（全族扫描 v14）：

```
BCPE  dpqr   pass
BCPE  deriv  pass
BCPE  rsdev  fail   deviance 94.1536 vs R 122.8256  (rel 2.334e-01)
```

方向是**我们更低**，按 D2.18 判据第 2 条应先怀疑 R 提前停。第一次调查
得出的是"两个数值上不连通的盆地"：`sigma` 低于 1.1077（含 R 的 0.508149）
时 deviance 返回 `Inf`，两侧皆然。**这个结论是错的**，错因值得单独记。

**根因：R 在自己的最优点上根本没到驻点，却报 `converged = TRUE`**

活 R 在 gate 真正比较的那份 48 观测样本上：

```
R  n.cyc=20    dev=122.825553739168
R  n.cyc=30    dev=122.825553739168
R  n.cyc=100   dev=122.825553739168
R  n.cyc=300   dev=122.825553739168
R  n.cyc=1000  dev=122.825553739168      converged = TRUE
```

逐迭代 deviance：`160.5062 → 125.5833 → 123.5814 → 123.1782 → 122.9885
→ 122.8992 → 122.8605 → 122.8426 → 122.8262 → 122.8256`。最后一步变化
**0.0006 < `c_crit` = 0.001**，R 就停了。而在 **R 自己报告的最优点**上，
eta 尺度 score 远离零：

```
dldm = -2.1556e+01    sum(dldm * x) = -4.1895
sigma * dldd = -4.8000e+01           dldv = -8.2223
tau * dldt   = -3.8259e-09
```

我们这一侧同样没到：`n_cyc=20` 是 94.153575（`max_iterations_reached`，
`tau` 还在爬），`n_cyc=40` 到 **94.115024891890840**。R 自己的 `dBCPE`
在我们的 `n_cyc=40` 系数上给 **94.115024891890670**（差 1.7e-13），密度
逐位一致，两侧最小化**同一个**目标函数。

**所以两侧的 `converged` 都不是在报告最优性**：R 报的是"deviance 的绝对
变化够小了"，我们报的是"步长塌缩了"（D2.22 已登记）。

**交付：把 eta 尺度 score 作为一等诊断发布**

`rs_algorithm.eta_gradient_norms()` 计算并发布
`max_p ||X_p' W dlogL/deta_p||_inf`，落在 `model` 的
`eta_gradient_norm` / `eta_gradient_norm_by_param` 槽位。它需要链式法则里
的 `dmu/deta`：natural 尺度的 `dld` **跨参数不可比**——`BCPE` 的
`sigma.link` / `tau.link` 是 `log` 而 `nu.link` 是 `identity`，于是
`sum(tau * dldt) = 1e-09` 可以与 `sum(dldm) = -21.6` 并存。

**这个量不 gate `converged`，这是本轮最重要的结论**

第一版是 gate 的：`eta_grad > 1e-3` 就清掉 `converged` 并新增终止原因
`stalled_at_nonzero_gradient`。阈值来自一个看起来很干净的分离——正常族
8.3e-14 ~ 9.4e-05，卡住的族 1.0 与 1.6e+09。**用"沿梯度走一步能不能真的
降 deviance"一测，这个分离是假的**：

| 族 | `eta_gradient` | Fisher 步的 deviance 增益 | 判定 |
|---|---|---|---|
| `NO` | 1.7e-09 | 0 | 真收敛 ✓ |
| `GA` | 2.9e-05 | 7.1e-12 | 真收敛 ✓ |
| **`PO`** | **1.6e-03** | **1.9e-08** | **假阳性 ✗** |
| **`WEI2`** | **1.2e-02** | **2.9e-06** | 真非驻点 ✓ |
| `GG` | 2.6e-01 | 8.1e-05 | 真非驻点 ✓ |
| `BCPE` | 1.16e+00 | 9.0e-03 | 真非驻点 ✓ |
| **`LOGITNO`** | **2.48** | **9.1e-13** | **假阳性 ✗** |
| `LNO` | 8.8e+12 | 4.3e+12 | 真非驻点 ✓ |

`PO`（1.6e-3，误报）与 `WEI2`（1.2e-2，命中）**区间重叠七倍**，任何阈值
都分不开：梯度量级量的是"离驻点多远（score 单位）"，不是"还有多少目标
函数可达"。gate 版本会把一个已收敛的 Poisson 拟合误标成停滞。

这与 D2.19 的参数地板判据同构：**唯一可接受的检验是问目标函数在另一侧
做什么，而不是比较量级。** 因此 `ETA_GRADIENT_TOL` 保留为文档常量，
数字发布出去，由调用方决定。

**D2.24 方法论要点**：

1. **通过非驻点的一维切片量的是轨迹，不是地形**。`sigma < 1.1077`
   返回 `Inf` 那个"盆地壁垒"，是在**我们自己仍未收敛的迭代点**上做的
   切片。`tau` 从 17631 爬到 473368 时，同一个 `sigma` 阈值跟着变。
   判据：**切片前先确认基准点是驻点**，否则读出的"障碍"是算法走过的路。
2. **"我们更低"要连续追问三次**：R 的逐迭代轨迹、R 在**自己**最优点上的
   score、R 的 `n.cyc` 扫描。前两次给出"R 停早了"，第三次排除"某个特定
   预算的巧合"。
3. **`g_dev_inc` 已经是完整 deviance**。把它的和乘 2（deviance 定义成
   `-2*logL` 时的自然反射）会让 `7.6e-16` 的 Newton 步看起来把 deviance
   移动了 168——本轮据此得出过"判定全错"的**反向**结论。
   **一个错的复现脚本不报错，它只会给你另一个问题的答案**（D2.23 第 5 条
   在新场景下重现）。
4. **量级分离不是判据，"走一步能否下降"才是**。见上表。与 D2.19 同构：
   读代码区分不出来的两个性质，必须让目标函数自己回答。
5. **两侧的 `converged` 都可能在撒谎，而 R 的更严重**。R 在 deviance
   122.83 处报 `converged=TRUE` 且 `n.cyc` 加到 1000 一位不变；我们至少
   会报 `max_iterations_reached`。**parity 对照不能把 R 的 `converged`
   当基准**——它是一个被 `c_crit` 绝对判据定义的事实，不是最优性证明。

#### D2.20：`TF2` 的链式法则项与 `TF` 的正态极限分支（rsdev rel 2.3e+01 →逐位一致）

**现象**（全族扫描 v11）：

```
TF2  dpqr   pass
TF2  deriv  fail   mu rel 7.519e-01、sigma rel 1.276e+00
TF2  rsdev  fail   deviance 3399.12 vs R 144.172  (rel 2.258e+01)
```

**`dpqr` 绿 + `deriv` 红**是特征信号：密度对、导数属于另一个密度。

**根因三处，逐层剥开**：

1. **重参数化壳转发了裸导数**。`TF2` 是 `TF` 的壳（`sigma1 = sqrt((nu-2)/nu) * sigma`），
   但三处score/hessian 直接把**原始 sigma** 转发给 `TF`，链式法则全丢：

   ```python
   def dldd_tf2(y, mu, sigma, nu):
       return fam.score_functions["sigma"](y, mu, sigma, _nu_gt2(nu))   # sigma 而非 sigma1
   ```

   实测（对我们**自己**的密度做有限差分，`y=0.3, mu=0, sigma=1, nu=3`）：

   | | 解析 | FD | 差 |
   |---|---|---|---|
   | `mu` | 0.38834951 | 1.10091743 | 7.1e-01 |
   | `sigma` | -0.88349515 | -0.66972477 | 2.1e-01 |
   | `nu` | 0.03111859 | -0.18480405 | 2.2e-01 |

   **hessian 与转发后的 score 自洽到 1e-11** —— 这就是它昂贵的原因：
   RS 在迭代一个**内部自洽但不属于所最小化密度**的梯度，
   `converged=True` 与看似合理的 Hessian 同时成立。

2. **修好 score 后 deviance 反而更差**（3399 → **6835**，`nu_eta = -36.04` 撞
   `logshiftto2` 地板）。因为 `d2ldv2` 是**唯一驱动 nu 方向**的项，而 R 写的是
   `TF()$d2ldd2(sigma1, nu)` —— 一个**只收两个参数的闭式**
   `-(2*nu)/((nu+3)*sigma1^2)`。我们转发了自己的 AD 版（收四个参数），
   在 `nu=3` 返回 `-1.0` 而 R 返回 `-3.0`，`d2ldv2` 因此变成 `-0.00983`
   而非 `-0.12802`（**13 倍**）。

3. **`TF` 的密度缺 `nu > 1e6` 正态极限分支**（D2.18 建立的 kernel 开关模式的漏项）。
   R 的 `dTF` / `dTF2` 都有`ifelse(nu > 1e+06, dNO(...), (1/sigma)*dt(...))`，
   我们的 `dTF` 与 `_log_density`都没有。而 **R 自己的 `TF2` 最优解就落在
   `nu = 3.6e20`**（`nu_eta = 47.33`）—— 漏掉这个分支意味着我们的密度
   **在 R 的解处就是错的**：单点log-lik `-23.68` vs R 的 `-1.2956`，
   48 点累加后 deviance **2334.79 vs 144.17**。

   两处都要改：`dpqr_functions.dTF`（RS 评估的）与
   `distributions.TF._log_density`（AD 微分的）—— **只改一处会重新制造
   密度/导数分裂**。

**修好后**：

| | 修复前 | 修复后 | R |
|---|---|---|---|
| deviance | 3399.12 | **144.171705** | 144.171705 |
| 密度在 R 系数点 | 2334.79 | **144.171705**（差 6.8e-12） | 144.171705 |
| `deriv mu` | rel 7.5e-01 | **逐位吻合 R** | — |

三个 score 与 `d2ldv2` 现在**逐位吻合活 R**（`nu` = 2.5 / 3 / 5 / 11.2 / 30）。

**我们停在 `nu_eta = 240`、R 停在 47.33，但 deviance 相同** —— `nu`方向
**弱识别平坦**（固定 `mu`/`sigma` 在 R 的解上扫）：

| `nu_eta` | `nu` | deviance |
|---|---|---|
| 4.0 | 5.7e+01 | 144.742391 |
| 10.0 | 2.2e+04 | 144.173133 |
| 20.0 | 4.9e+08 | 144.171705 |
| 47.33（R 的停止点） | 3.6e+20 | 144.171705 |
| 240.0（我们的停止点） | 1.7e+104 | 144.171705 |

`nu_eta >= 20` 之后平坦到 15 位数字，**两个停止点是同一个目标函数值**。
R 提前停是因为 `c_crit = 0.001` 是**绝对**判据而 deviance 在百量级
（D2.18 同一模式）。所以断言是**双向带容差**，不是相等。

**顺带修正**：`TF2$nu.initial = 4`，而 `TF$nu.initial = 10` ——
**同参数名、同文件、不同值**，不能合并分支（D2.19 教训）。R 的
`TF2$nu.dr = pmax(.Machine$double.eps, exp(eta))`、���
`nu.linkfun = log(nu - 2 + 1e-05)`（加的是 `1e-05` 不是 `eps`，
在 `nu≈47` 处两者都远小于值，**不是根因**）。

**最终全族扫描确认**（146 族，21 分钟）：

| 关卡（可测口径，排除 skip） | D2.19 之后 | D2.20 之后 |
|---|---|---|
| `deriv` | 133/144 (92.4%) | **134/144 (93.1%)** |
| `dpqr` | 63/100 (63.0%) | 63/100 (63.0%) |
| `rsdev` | 49/60 (81.7%) | **50/60 (83.3%)** |

`TF2` **三关全绿**：

```
TF2  deriv  pass
TF2  dpqr   pass
TF2  rsdev  pass   144.172 vs R 144.172
```

`TF` 也保持三关全绿（`rsdev 133.116 vs R 133.114`）→ **正态分支改动无回退**。

相对 v6 的**所有**状态变化都在 `skip → pass/fail` 方向
（`ST3C deriv skip→pass`、`ST3C dpqr skip→fail` 且 `d` 项消失、
`ZABNB`/`ZASICHEL`/`ZIBNB`/`ZISICHEL deriv skip→fail`）——
按 D2.19 纪律这是**信息量增加，不是回退**。

**新增测试** `tests/test_tf2_parity.py`（17 项）：score 对自身密元的 FD
（不依赖 R，参数化 4 档`nu`）、score 对活 R 逐点表、hessian 对 **R 的公式**
（含 `TF()$d2ldd2` 两参数闭式）、旧转发会失败、R 的 `G.dev.incr` 公式
（大写 G，小写查是 NULL）、links/valid/初值、`nu <= 2` 时 R 为 NaN 的证据、
我们地板不压平密度、deviance 双向断言 + 平坦区交叉互评。

**一个被证伪的假设（写下来是因为它反直觉）**：
我原以为"去掉 `ds1dd` 链式因子会重现缺陷"，写成测试后**失败**——
去掉后 deviance 仍是 144.172（rel **1.4e-13**），拟合毫发无损。
原因是 `sigma` 在给定 `mu`/`nu` 时**不独立可识别**：
`sigma1 = sqrt((nu-2)/nu) * sigma` 是双射，sigma score 的缩放会被
sigma 估计本身吸收，回到同一个最优。

→ **原缺陷之所以可见，是因为三个 score 同时错 + 密度缺正态分支，
不是任何单项。** 已把它改成两个测试：
`test_dropping_the_ds1dd_term_is_absorbed_by_the_other_parameters`
（钉住这个负面结果，防止有人"简化"回去）和
`test_dropping_the_normal_limit_branch_does_break_it`
（去掉正态分支 → deviance 2334.79，**这一项才不可吸收**）。

**D2.20 方法论要点**：

1. **"密度对、导数错"要往密度里看，不是只修导数**。本轮三处根因里有两处
   是**密度**的问题（`d2ldv2` 用错函数、`dTF` 缺正态分支），
   修导数只解决了 1/3。
2. **修好一处可能让结果更差**。score 修好后 deviance 从 3399 涨到 6835 ——
   **不要在单步变好时停下**（D2.13 的 BCT 教训第三次重演）。
3. **"RS 会用哪些函数"要逐个查**：deviance 用 `dTF`，导数用 `_log_density`，
   两者**必须同步**。只改一处等于重新制造密度/导数分裂。
4. **R 的两参数闭式 ≠ 我们的四参数 AD 版**。`(sigma, nu)` 与
   `(y, mu, sigma, nu)` 签名不同就足以偷换函数，且**值差 3 倍**。
5. **别假设每项修复都独立必要**。见上面被证伪的假设 ——
   `sigma` 方向不可识别，链式项错了也未必显现。
6. **弱识别平坦区要用"扫参数看目标函数"定性**，不能只看两个停止点差得远。
7. **本轮四个假发现全在探针侧**：`max(val, -1e-15)` ≠ R 的 `ifelse`；
   `jsonlite` 的 NULL 渲染成 `{}` 而非 `null`；links 的 JSON 键是
   `mu.link` 而非 `mu`；`_scanner_sample` 的缓存只装两族导致 4 个 R 对照
   **静默 skip**（长得和通过一样）。**先查探针这条纪律救了四次。**

**回归**：新增 `test_tf2_parity.py` 17 项；
`test_st3c_sst_parity.py` 15 项（并修好其 `_scanner_sample` 缓存范围）；
`test_st1_st2_parity.py` + `test_rs_family_parity_scan.py`；
能力矩阵与覆盖门禁；覆盖工件同 commit 再生，`coverage_diff_report.py`
→ `result: OK`。

**既有失败台账不变**：`test_r_consistency_st_compat.py::TestSTCompatConsistency`
8 个（`R Error: nu must be positive`）—— 上轮已 `git stash` 确认改动前完全相同，
本轮未触碰该文件。

#### D2.19：`ST3C` / `SST` 的 link 与初值（rsdev rel 1.8e-01 / 8.6e+01 → 2e-06）

**现象**（全族扫描 v9）：

```
ST3C  rsdev  fail  deviance168.41  vs R 142.293  (rel 1.835e-01)
SST   rsdev  fail  deviance11751   vs R 135.126  (rel 8.596e+01)
```

SST 的 deviance 是 R 的 **87 倍** —— 这个量级通常意味着密度在某处被系统性
改写，而不是"优化没收敛"。但**这次不是密度**。

**根因三处，其中两处是我先前判断错的**：

1. **`ST3C.nu.link` 是 `identity`，R 是 `log`**（确凿缺陷）。
2. **`_initial_parameter_value` 对 `tau` 的兜底是 `1.0`，R 的 `ST3C` 用 `10.0`、
   `SST` 用 `4.0`**（确凿缺陷）。从 `tau = 1.0` 起步会让RS 停在一块平台上
   出不来 —— 我们收敛在 `tau_eta = -0.0`（即 `tau = 1.0`），
   而 R 的最优在 `tau = 1.8e11`（t 分布分支的远端）。
3. **单修 link 无效**：只改 link 时 deviance 仍是 168.410。
   **初值才是让数字移动的那一处**（D2.13 的BCT 教训在此重演：
   *导数/link 是耦合变量，单步只改一个会得出误导性结论*）。

**我先前判错的两处**（都已用活 R 推翻）：

- 我以为 `SST.tau.link = logshiftto2` 是抄错 —— **R 自己就是这样**
  （`SST()$tau.link` 实测为 `logshiftto2`），我们没错。
- 我以为 `_sst_log_pdf` 里的 `tau = maximum(tau, 2.0 + eps)` 地板是
  自创约束 —— **它标的正是 R 的有效域**。R 的 `dSST` 里 `m2` 带
  `(tau - 2)` 分母，`tau <= 1.9` 时 `m2 < 0`、R 自己的 `sqrt` 返回 `NaN`
  （实测 `tau = 0.5/1/1.5/1.9/2.0` 全为 `NaN`，`2.0001` 起才有限）。
  活 R 还打印 `In sqrt(m2 - m1^2) : NaNs produced`。

**密度是清白的**，逐元素对活 R 验证：

- 在 R 自己收敛的参数点上（`nu = 1.00127`），t 分支（`tau = 10`）
  吻合 **2.2e-15**，正态极限分支（`tau = 1.8e11`）吻合 **4.4e-16**；
- 用 R 的**完整**系数组代入我们的密度，得 `142.292666` / `135.126290`，
  与 R 报的 `142.292666` / `135.126290` 差 **2.8e-14 / 5.7e-14**
  → **两侧最小化的是同一个目标函数**，差距全在搜索路径上。

**验收**（扫描器同一份48 点样本，`n_cyc = 500`）：

| | 修复前 | 修复后 | R | 修复后 rel |
|---|---|---|---|---|
| ST3C | 168.410146 | **142.292313** | 142.292666 | **2.5e-06** |
| SST | 11747.690857 | **135.126590** | 135.126290 | **2.2e-06** |

残余的 2e-06 是 **R 自己的 `c_crit = 0.001` 绝对停止判据**造成的 ——
R 同样没到精确最优。

**第四处根因（修完拟合后才由全族扫描暴露出来）**：
`_st3c_log_pdf` 里的 `tau = maximum(tau, 1.0 + eps)` 地板。

修完上面三处后重跑全族扫描，`ST3C` 的 `dpqr` / `deriv` **由 skip 变fail**
（v6 时它们是 skip —— 从未测到，不是通过）：

```
ST3C  deriv  fail  tau: max rel diff 9.697e-02  (analytic [0, 0.] vs fd [0, 0.09697])
ST3C  dpqr   fail  d: max rel 8.243e-02; p: 1.589e-02; q: 8.247e-01
```

定位过程：扫描器网格是 `nu == tau == linspace(0.5, 3.0, 48)`。逐点对活 R，
偏差随 `tau` **单调递减**（`tau=0.5` 时 0.265→ `tau=0.87` 时 0.049），
48 点里只有 1 点吻合。分轴扫描精确定位：

```
tau 轴 (nu=1.0, y=0.3):
  BAD  tau=0.3   ours=-1.2309075820904529   R=-1.6431769009605328   d=4.12e-01
  BAD  tau=0.5   ours=-1.2309075820904529   R=-1.4346687547696897   d=2.04e-01
  BAD  tau=0.7   ours=-1.2309075820904529   R=-1.3248233004784622   d=9.39e-02
  OK   tau=1.0   ours=-1.2309075820904529   R=-1.2309075820904525   d=4.44e-16
  ...  (tau >= 1 全部OK)
nu 轴 (tau=2.0, y=0.3):  5 点全部吻合到 1e-16
```

→ **`tau < 1` 时我们的密度完全不随 `tau` 变**：地板把它压成了常函数，
导数恒为 0。R 的 `ST3C()$tau.valid` 是 `all(tau > 0)`，**没有 1 的地板**。
修为 `maximum(tau, eps)` 后：

- 13 个 `tau`/`nu` 扫描点全部吻合到 1e-15；
- 扫描器精确网格 **48/48 点吻合到 1e-12，max 3.109e-15**。

**最终全族扫描确认**（21 分钟，146 族）：

| 关卡（可测口径，排除 skip） | 修复前 | 修复后 |
|---|---|---|
| `deriv` | 132/144 (91.7%) | **133/144 (92.4%)** |
| `dpqr` | 63/100 (63.0%) | 63/100 (63.0%) |
| `rsdev` | 49/60 (81.7%) | 49/60 (81.7%) |

逐族状态变化**只有一条，且是变好**：

```
ST3C  deriv  fail -> pass
ST3C  dpqr   d 项消失：修复前报 "d: max rel 8.243e-02"，修复后只剩
                    "p: 1.589e-02; q: 8.247e-01"
```

**无任何族回退。**（`ST3C rsdev` 保持 `pass`、`SST dpqr` 保持 `pass`。）
`ST3C dpqr` 整体仍是 `fail` —— `p`/`q` 的偏差与 `tau` 地板无关，
属 D1（d/p/q parity）范围，`q` 的 8.2e-01 是留给 D1 的存量。

**为什么 `SST` 的 `2 + eps` 地板不能一起改** —— 两处形状完全一样
（一行 `jnp.maximum`），但性质相反：

| | R 的边界 | 边界另一侧 | 地板是否正当 |
|---|---|---|---|
| `SST` | `m2` 带 `(tau-2)` 分母 | `m2 < 0` → R 自己 `sqrt` 得`NaN`，并**直接 abort** | **正当**：标的是 R 的有效域 |
| `ST3C` | `tau.valid = all(tau > 0)` | 密度有定义且随 tau 变化 | **不正当**：删掉了整个参数区域 |

**判据不是"地板看起来是否必要"，而是"R 在边界另一侧是否真有定义"** ——
两者都只能靠实R 求值区分，读代码看不出来。上面那个分轴表就是区分过程。

**本轮我自己的探针造了四个假发现**（都先怀疑探针、再怀疑实现）：

1. 密度对照把 `tau` 硬编码成 `4.0`，而 R 的 dump 循环 `tau in {2.5, 4, 10}`
   → **整屏"不匹配"，worst 4.7**。
2. 互评时代借了我们的 `mu` 代替 R 的（R 的 mu 是 `[2.292, -0.0512]`，
   我们是 `[1.597, -0.0449]`；mu 经`z` 进入密度，不能借）
   → 报出 **76.9** 和 **8.4e16** 的假差距。
3. 测试文件硬编码 `R-4.4.1/bin/Rscript.exe`（真实路径带 `x64`）
   → **所有 R 对照用例静默 skip**，看起来像全绿。
4. `tau` 地板修完后全网格复核残差仍有 **1.66e-03**，一度以为没修干净——
   实际是**我手抄 48 点 `y` 网格时只保留了 4 位小数**
   （真实 `y[2] = 0.4574468085106383`，我写 `0.4574`，差 4.68e-05）。
   改为从 `_family_grids` 导出**全精度** JSON 给 R 求值后，
   **48/48 点吻合到 1e-12，max 3.109e-15**。

**处置**：测试改为复用 `rs_family_parity_scan._rscript_path()`
（与扫描器同一份候选列表），路径找不到时 skip 会显式报出。

**新增测试** `tests/test_st3c_sst_parity.py`（**15 项**）：

- `test_density_matches_live_r_at_rs_optimum`：密度在 R 收敛点、
  t 分支与正态分支两侧都对活 R —— **这条排在最前，因为后面所有结论都建立在它上面**；
- `test_our_density_reproduces_rs_deviance_at_r_coefficients`：用 R 的**完整**系数
  复现 R 的 deviance，把上面第 2 个假发现钉死；
- `test_links_match_r` / `test_sep1_sep2_keep_identity_nu_link`：
  前者从活 R 对象读 links（硬编码表只作冗余断言，R 改links 时 fixture 先失败），
  后者**防过度修复** —— `SEP1`/`SEP2` 同文件、同参数名，R 确实用 `identity`；
- `test_initial_values_match_r`：对活 R 核对 `nu`/`tau` 初值；
- `test_st3c_default_tau_of_one_would_reproduce_the_defect`：说明旧兜底 `1.0`
  为什么会产出扫描器报的那个 168.410；
- `test_sst_tau_floor_marks_rs_own_domain_not_an_invention` +
  `test_r_sst_is_nan_below_tau_two`：把 `SST` 的 `tau >= 2` 地板钉在
  **R 的实测行为**上，而不是我们的口味 —— 将来 R 若在 `tau < 2` 有定义，
  该改的是地板；
- `test_st3c_density_varies_with_tau_below_one`：**`ST3C` 的 `tau < 1` 地板**。
  先断言**单调性**（不依赖 R，R fixture 不可用时仍会响），
  再断言**与活 R 全扫描点吻合到 1e-13** —— 前者保证"会变"，后者保证"变对了"。

**D2.19 方法论要点**：

1. **元数据对了不代表问题解决**。单修 `ST3C.nu.link` 时 deviance 一位未动
   （168.410 → 168.410）。**验收标准是 deviance 落在 R 上，不是元数据看起来对。**
2. **抄 R 的"错"要分清是错还是有效域**。`logshiftto2`（SST.tau.link）与
   `tau >= 2` 地板看起来都像我们自创的，R 的实现证明它们都是 R 自己的。
3. **同文件同形不等于同族**。`grep '"nu": "identity"'` 在 `distributions_b15.py`
   命中 4 处（`SEP1`/`SEP2`/`ST3C`/`LOGSHASH`），其中**只有 `ST3C` 是错的**。
   向 R 逐族求证而不是批量替换。
4. **探针报错先查探针**。本轮四个假发现全部出在探针侧，且每一个都"证据充分"
   —— 整屏不匹配、76.9 的差距、76 个 skip、修完仍有 1.66e-03 残差。
   **先证明密度对，再谈优化差异。**
5. **Rscript 路径写错 = 静默全绿**。测试里硬编码 R 路径的失败模式与通过一模一样。
   复用工具的发现函数。
6. **skip → fail 是信息量增加，不是回退**。v6 时 `ST3C` 的 `dpqr`/`deriv`
   是 `skip`（从未测到），v10 变`fail`（测到且失败）。
   **判据是"是否测到"，不是"是否通过"** —— 覆盖率停滞时更该关心这个。
7. **地板的正确性只能靠实R 求值**。`SST` 的 `2 + eps` 与 `ST3C` 的 `1 + eps`
   形状完全一样，性质完全相反。分轴扫描（固定一轴扫另一轴）是从
   "整体不符"缩到"单轴不符"的最快手段。

**回归**：新增 `test_st3c_sst_parity.py` 15 项；
`test_st1_st2_parity.py` 26 项；`test_rs_family_parity_scan.py`；
能力矩阵与 ST compat（除下述既有失败）；覆盖工件同 commit 再生，
`coverage_diff_report.py` → `result: OK`。

**既有失败台账**：`test_r_consistency_st_compat.py::test_st1`–`test_st4`
共 8 个 SUBFAILED（`R Error: nu must be positive`）—— 上轮已用
`git stash` 确认改动前完全相同，本轮未触碰该文件。



### D1 分布 family d/p/q/r parity（P1，8–12 周）

- 批次 A：ST/SEP/SST（独立参数化、尾部、q 稳健反演）；
- 批次 B：Box-Cox `o`、BEINF/BEZI/BEOI/LG/LNO/LOGITNO/NET（截断归一化、混合质量、边界）；
- 批次 C：DEL/DPO/SI/WARING/YULE/NBF/GPO/NBII（离散递推、整数 support）；
- 批次 D：ZALG/ZIP/ZIP2/ZIPF 等零膨胀/零调整（零质量、尾部、log-p）；
- 批次 E：GB/GIG/NO2/NOF/PARETO/PE2/PIG2/RGE/WEI2（特殊函数、正实 support）；
- 批次 F：EXGAUS/EGB2/LOGSHASH/LQNO/SN/MN 与全部剩余 placeholder（18 个清零）。

**验收**：每族 d/p/q/r 通过固定点、边界、单调性/归一化、广播、upper-tail/log-p、抽样与实时 R 对照；JSON/Markdown/placeholder 队列同步再生。

### D2 拟合 parity（P2，6–10 周）

- 每族最小数据集（连续/计数/零膨胀/边界/极端）；
- 记录系数、fitted、deviance、迭代、收敛与失败原因；
- 修复 NaN-as-convergence、溢出伪成功、MIXED 路由、ZAGA/GU 缺陷、链接尺度热启动污染；
- **验收**：RS 全覆盖 + 每族至少一个 R golden；CG/MIXED 按族型分层覆盖；失败可分类解释。

### D3 核心 API 与 smoother（P3，10–16 周）

顺序：family/link/check → formula/`Formulae2data`/`predictAll`/update/refit → residuals/diagnostics/hatvalues/vcov/logLik → GAIC/stepwise/find.hyper → smoother（`lo/random/re/ri/pcat/fp/bfp/cy/pvc/pbm/pbz/pbo/pbc/pbp/scs`）。

**验收**：每 smoother 通过 parsing、basis、penalty、lambda/EDF、fit、newdata、R golden、admission 八项；未通过者明确拒绝。

### D4 周边包（P4，12–20 周）

D4.1 `gamlss.tr`+`gamlss.cens` → D4.2 `gamlss.mx`+`gamlss.add` → D4.3 spatial/foreach/prepdata → D4.4 ggplots 数据层+`gamlss.data`+WeatherGermany。每包按同四层证据验收。

### D5 L3 服务与文档

生产路由 fail-closed（placeholder/alias/experimental 拒绝）；序列化往返；TLS/认证回归；API 参考、R↔Python 参数化迁移对照表、8 篇核心教程校准。

### D6 L2 ondil 在线层（独立验收）

共享 distribution contract；离线等价初始化、chunk 顺序不变性、warm-start、漂移、缺失值、并发、流式 golden、性能/内存基线。L1 R golden 不覆盖在线 estimator。

### D7 质量与发布门禁

`--require-r` 全绿、无隐性 skip、coverage diff、目标/相关 suite 与分层 collection 日志、中文任务提交、可复现命令；性能 benchmark 仅在一致性通过后运行并区分 cold/warm、CPU/GPU。

### D8 项目结构与文档治理（本次提交启动，持续执行）

见第 7 节规范；每次提交前执行结构检查（无临时文件入库、无死链接、生成物与运行时一致）。

## 4. 里程碑

| 里程碑 | 交付物组合 | 建议版本 |
|---|---|---|
| M0 | D0 全部 + D8 结构治理落地 | 0.4.x |
| M1 | D1 全批次：目标 R family 全部四层（DPQR parity），placeholder 清零 | 0.5.0 |
| M2 | D2 拟合 parity：RS 全覆盖、CG/MIXED 分层覆盖 | 0.6.0 |
| M3 | D3 核心 API 与准入 smoother 闭环 | 0.9.0 |
| M4 | D4 周边包 + D5 服务文档 + D7 发布门禁 | 1.0.0 |

D6（ondil）与 M1–M2 并行推进，独立发布，不阻塞 1.0。

## 5. 排期（容量估计，非承诺）

| 周期 | 主线 | 并行 |
|---|---|---|
| W1–W2 | D0 审计基座 + D8 结构治理 | 文档链接修复 |
| W3–W6 | D1-A/B/C + D2 数据集与诊断 schema | D6 contract |
| W7–W10 | D1-D/E + D2 RS 扩展 | CI/覆盖自动化 |
| W11–W14 | D1-F 清零 + D2 CG/MIXED 分层 | D6 流式 golden |
| W15–W20 | D3 family/core/预测/诊断 | 基准真实化 |
| W21–W28 | D3 smoother + D4.1 | D5 文档 |
| W29–W40 | D4.2–D4.4 + D5 + D7 发布 | D6 性能基线 |

依赖规则：D0 未完成不得宣称 M0；D1 未通过的 family 不得进入 D2 拟合宣称；D3 smoother 依赖 D2 稳定；D4 依赖 D3 核心 API 冻结。

## 6. 完成定义（DoD）

**单任务 DoD**：R source map → Python/JAX 设计 → 实现 → 单元/属性/R 对照/拟合测试（适用时）→ golden/manifest/环境证据 → 状态更新 → 目标与相关 suite + 分层 collection + `git diff --check`。

**里程碑 DoD**：第 4 节对应交付物全部达标 + D7 门禁全绿 + 文档/教程/CHANGELOG 同步 + 中文提交记录。

## 7. 项目结构规范（清理后的目录约定）

```
OmniLSS/
├── omnilss/            # L1 主实现（src + tests + tools）
├── ondil/              # L2 参考（外部子项目，忽略）
├── gamlss-python/      # 转录参考（外部子项目，忽略）
├── R/                  # R 原始源码（语义基准，忽略）
├── benchmarks/         # 基准与 R 一致性脚本（results 为生成物，忽略）
├── docs/
│   ├── development/    # 计划与审计（仅保留当前有效版本）
│   │   └── task-reports/  # 单任务交付记录（按任务归档）
│   └── testing|validation|api|...  # 用户与验证文档
├── tools/              # 仓库级生成/检查工具
├── tmp/                # 本地临时产物（忽略，禁止提交）
└── DEVELOPMENT_PLAN.md # 项目总纲
```

文件生命周期规则：

1. **计划/审计文档不保留过期副本**：同主题只保留最新版，历史版本由 Git 历史承载；
2. **生成物必须可再生**：coverage、capability 矩阵、placeholder 队列随代码同批再生成，禁止手改；
3. **单任务报告归档到 `docs/development/task-reports/`**，不散落在 `docs/development/` 根部；
4. **会话笔记与临时脚本不入库**：`SESSION_SUMMARY_*`、`tmp/`、`mnt/`、checkpoint、缓存目录一律忽略并可随时删除；
5. **mkdocs nav 不得包含死链接**；新增用户文档必须提供 `_cn` 对应文件并互链（内部开发文档豁免）。

## 8. 立即执行清单（第一批）

- [x] D0.1 重写 coverage 工具为真实 R↔Python 逐符号矩阵（2026-09-26 完成；工具 `tools/generate_symbol_coverage.py`，生成物 `docs/development/r-symbol-coverage.{json,md}`，报告 `docs/development/task-reports/d0-1-symbol-coverage-matrix-2026-09-26.md`）；
- [x] D0.2 golden manifest 模板与 `--require-r` 门禁对齐（2026-09-26 完成；模板 `omnilss/tests/fixtures/r_golden/manifest_template.json`，门禁 `tools/golden_manifest_gate.py`，报告 `docs/development/task-reports/d0-2-golden-manifest-gate-2026-09-26.md`）；
- [x] D0.3 四套分层 collection 命令固化到 CI（2026-09-26 完成；工具 `tools/tiered_collection.py`，
  CI job `collection-gate`，report `docs/development/task-reports/d0-3-tiered-collection-2026-09-26.md`；
  分组为本项目四测试面，`ondil`/`gamlss-python` 仅作参考对比）；
- [x] D0.4 registry/R 源码变化自动差异报告（2026-10-02 完成；工具 `tools/coverage_diff_report.py`，
  ratchet 基线 `docs/development/symbol-coverage-baseline.json`、报告快照 `docs/development/symbol-coverage-diff.json`，
  回归测试 `omnilss/tests/test_coverage_diff_report.py`，CI job `collection-gate`，
  报告 `docs/development/task-reports/d0-4-coverage-diff-report-2026-10-02.md`）；
  - D0 至此四项齐备：D0.1 矩阵 / D0.2 golden 门禁 / D0.3 分层 collection / D0.4 回归 ratchet。
- [ ] D1-A ST/SEP/SST R golden 与独立 kernel；
- [ ] D1-B Box-Cox-o 与有界混合批次；
- [ ] D1-C 离散递推批次；
- [ ] D2.1 全族最小拟合数据集与收敛诊断 schema；
- [ ] D2.2 修复 NaN/overflow/MIXED 路由并加回归；
- [ ] D3.1 冻结 family/link/formula/prediction 契约；
- [ ] D5.1 迁移对照表与 8 篇教程校准；
- [ ] D6.1 OmniLSS↔ondil distribution contract；
- [ ] D8.1 结构检查脚本（死链接、临时文件、生成物一致性）。

## 9. 风险与决策

- **范围风险**：146 注册项 ≠ 146 个 R family；只按四层证据计数。
- **数值风险**：x64/JAX 版本/特殊函数/优化器差异；失败必须分类（公式、参数化、优化器、数值稳定性）。
- **证据风险**：inventory 冒充 coverage、R 缺失转 skip、静态空 golden——均为发布阻断项。
- **进度风险**：全库收集超时 ≠ 通过；采用分层日志与超时定位。
- **决策冻结**：gamlss2 不阻塞 1.0；服务路由 fail-closed；benchmark 不先于 parity；外部参考项目（gamlss-python/ondil）只读。



