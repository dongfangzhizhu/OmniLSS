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



