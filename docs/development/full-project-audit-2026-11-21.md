# OmniLSS 全面项目审核报告（2026-11-21，D2.270 基线）

> 审核对象：`main` 分支 HEAD（`7aedafc`，D2.270）、`R/` vendored R 源码（12 个 R 包）、`omnilss/`、`omnilss-pro/`、`omnilss-server/`、`benchmarks/`、`tools/`、`docs/`、`tutorials/`、`ondil/`、`gamlss-python/`。
> 审核方法：全部指标由**本机实时命令与权威工件复算**（非引用历史文档），逐项核对 Surface / Kernel / Parity / Admission 四层证据、14 道门禁脚本、子项目测试与代码卫生。
> 配套后续开发计划：见根目录 `DEVELOPMENT_PLAN.md` 第二十三节。
> 历史对照：本报告取代 `docs/development/full-project-audit-2026-11-05.md`（D2.167 基线）。

---

## 0. 执行摘要（D2.167 → D2.270 跃迁与当前真实定位）

自 2026-11-05（D2.167）上次全面审核以来，项目经过 103 个增量里程碑（D2.168–D2.270），在 **R 符号覆盖面、核心框架、周边 R 包、生产准入与 ondil 契约** 上取得了跨越式进展，但距离 **1.0.0 正式发布** 仍存在明确的「证据收口、工程漂移、文档失真与独立轨道（在线/多元/GPU）」缺口。

| 维度 | 2026-11-05 (D2.167) | 2026-11-21 (D2.270 实测) | 变化与定性 |
|---|---:|---:|---|
| 范围内 R 符号 surface 覆盖（11 包，共 1036） | 687 / 1036 (66.3%) | **1025 / 1036 (98.94%)** | +338 符号；10 个包 100%，仅 `gamlss` 缺 **11** 个 |
| 全量 `surface_missing`（含 out-of-scope `gamlss2` 42 个） | 391 | **53**（范围内 11 + `gamlss2` 42） | 较 baseline(173) 净清 120 项 |
| `kernel=implemented` 符号 | 163 | **606** | +443 |
| `parity=unverified` 符号（全量 / 范围内） | 425 / — | **97 / 54** | 大幅收敛（仅 `gamlss` 36 + `gamlss.dist` 18） |
| `parity=golden:validated` + `r_consistency_test` | 34 + 172 = 206 | **84 + 215 = 299** | 活 R 实证显著增强 |
| `admission=production_safe`（符号 / 族） | 5 符号 / 1 族 (`NO`) | **110 符号 / 22 族** | M3 准入闭环，但仍占 146 族的 15.1% |
| 运行时 `implemented` / `experimental` / `compat_alias` / `placeholder` | 1 / 125 / 20 / 0 | **22 / 104 / 20 / 0** | 0 placeholder 保持 |
| 里程碑状态 | M0 未闭 | **M0 / M1 / M2 / M3 全部达成**；M4 仅完成周边包 surface/证据 | 版本号仍停在 `0.4.0` |
| 分层测试收集（4 层） | 497 文件 | **620 文件 / 5,822 用例**（0 err / 0 skip） | `omnilss` 5786、`pro` 25、`bench` 10、`server` 1 |
| 门禁实测通过率 | — | **12/14 PASS，2 项 DRIFT FAIL** | `m1_assessment` 与 `layered_parity` 产物漂移 |

**核心定性**：
1. **R 表面迁移已进入收官阶段（98.94%）**：11 个范围内 R 包中，除核心 `gamlss` 尚缺 11 个符号（平滑/随机效应 3、逐步选择 2、诊断绘图 5、杂项 1）外，其余 10 个包（含 `gamlss.dist` 与 8 个周边包）全部达到 100% surface 覆盖。
2. **「表面完成」≠「1.0 生产就绪」**：146 个注册族中仍有 104 个 `experimental`、421 个 `blocked` 符号；`cg_fit`/`sampling`/`smooth_terms`/`rs_jax_fit`/`ad_hessian` 能力矩阵的 `validated` 计数仍全部为 0；部分已迁符号（如 `gamlss.mx` 高级分支、`pvc(by=...)`、`gamlss.cs` 自动 GCV）仍为显式 `NotImplementedError`。
3. **当前最急迫的是「止漂移 + 修子项目测试 + 文档/版本对齐」**：两项里程碑产物已漂移失败、`omnilss-pro` 测试在真 `grpc` 下有 2 处真实失败、`README.md`/`HANDOFF.md`/`CHANGELOG.md`/`docs/tutorials` 严重落后于代码现状。

---

## 1. 环境与实测验证记录

| 组件 | 2026-11-21 实测版本 / 事实 | 备注 |
|---|---|---|
| `.venv` | Windows 运行环境（`uv` 管理）：Python 3.10.16 / JAX 0.6.2 (`x64=True`) / NumPy 2.2.6 / SciPy 1.15.3 / OmniLSS `0.4.0` | 当前未安装 `ruff`、`black`、`mypy`（可通过 `uv` 同步 `.[dev]` 补齐） |
| R | R 4.6.0 (2026-04-24 ucrt) + `gamlss` 5.5.0 + `gamlss.dist` 6.1.1 + `jsonlite` | 活 R 对照可用；本机无 Rtools/`make`（源码安装类用例 skip） |
| `R/` vendored | 12 个 R 包 / 1,354 文件（`gamlss` 5.5-3、`gamlss.dist` 6.1-8） | 权威 R 源码对照，受保护 |
| `ondil/` | 0.4.8（119 源码 `.py`，共 300 文件） | **仅作设计参考**，严禁引入其代码或作为依赖 |
| `gamlss-python/` | 0.2.0（52 `.py`，34 分布族，共 124 文件） | **仅作设计参考**，严禁引入其代码或作为依赖 |
| `debianjaxgpu/` | WSL Debian GPU 运行环境（`uv` 管理，7,137 文件，5.22 GB） | 受保护，未改动；**至今未执行 GPU parity/性能验证** |

---

## 2. 权威覆盖率与能力矩阵（实测）

### 2.1 逐 R 包覆盖与四层证据（`docs/development/r-symbol-coverage.json`）

| R 包 | 版本 | 范围 | 导出 | Matched | Missing | Kernel (impl/exp/alias) | Parity (gold/rcons/capv/test/unsup/unver) | Admission (prod/exp/int/blk) |
|---|---|---|---:|---:|---:|---|---|---|
| `gamlss` | 5.5-3 | In | 152 | 141 | **11** | 141 / 0 / 0 | 0 / 13 / 0 / 103 / 0 / **36** | 0 / 111 / 30 / 0 |
| `gamlss.dist` | 6.1-8 | In | 654 | **654** | 0 | 233 / 420 / 1 | 84 / 176 / 55 / 72 / 249 / **18** | **110** / 77 / 46 / **421** |
| `gamlss.add` | 5.1-14 | In | 15 | **15** | 0 | 15 / 0 / 0 | 0 / 4 / 0 / 11 / 0 / 0 | 0 / 15 / 0 / 0 |
| `gamlss.cens` | 5.0-8 | In | 5 | **5** | 0 | 5 / 0 / 0 | 0 / 4 / 0 / 1 / 0 / 0 | 0 / 5 / 0 / 0 |
| `gamlss.tr` | 5.1-10 | In | 8 | **8** | 0 | 8 / 0 / 0 | 0 / 5 / 0 / 3 / 0 / 0 | 0 / 8 / 0 / 0 |
| `gamlss.mx` | 6.0-4 | In | 12 | **12** | 0 | 12 / 0 / 0 | 0 / 3 / 0 / 9 / 0 / 0 | 0 / 12 / 0 / 0 |
| `gamlss.spatial` | 3.0-3 | In | 9 | **9** | 0 | 9 / 0 / 0 | 0 / 2 / 0 / 7 / 0 / 0 | 0 / 9 / 0 / 0 |
| `gamlss.foreach` | 1.1-10 | In | 13 | **13** | 0 | 13 / 0 / 0 | 0 / 1 / 0 / 12 / 0 / 0 | 0 / 13 / 0 / 0 |
| `gamlss.prepdata` | 0.1.19 | In | 59 | **59** | 0 | 59 / 0 / 0 | 0 / 1 / 0 / 58 / 0 / 0 | 0 / 59 / 0 / 0 |
| `gamlss.ggplots` | 2.1-34 | In | 109 | **109** | 0 | 109 / 0 / 0 | 0 / 5 / 0 / 104 / 0 / 0 | 0 / 109 / 0 / 0 |
| `gamlss.data` | 6.0-9 | In | 0 | 0 | 0 | — | — | — |
| `gamlss2` | 0.1-0 | **Out** | 44 | 2 | 42 | 2 / 0 / 0 | 0 / 1 / 0 / 0 / 0 / 43 | 0 / 1 / 1 / 0 |
| **合计（范围内）** | — | — | **1036** | **1025** | **11** | 604 / 420 / 1 | 84 / 214 / 55 / 380 / 249 / **54** | 110 / 418 / 76 / 421 |

### 2.2 范围内仅剩的 11 个缺失符号（全在 `gamlss` 核心包）

| 类别 | R 符号 | R 源码出处 | 优先级与说明 |
|---|---|---|---|
| 平滑 / 随机效应（3） | `re`, `gamlss.re` | `R/gamlss/R/re_gamlss_lme_test.R` (174 行) | **高**：随机效应接口（nlme 风格）；需确定 Python 侧混合效应后端策略 |
| | `gamlss.lo` | `R/gamlss/R/lo_new.R` (275 行) | **高**：LOESS 拟合回调（`lo` 已有规格，`gamlss.lo` 缺失） |
| 模型选择（2） | `stepGAIC.CH` | `R/gamlss/R/stepGAIC-03-10-13..R` (522 行) | **中**：并行/交叉验证式 GAIC 逐步选择变体 |
| | `stepTGDAll.A` | `R/gamlss/R/stepTGDAll-A-parallel.R` (180 行) | **中**：全参数验证偏差逐步选择 |
| 诊断 / 校准 / 绘图（5） | `bp` | `R/gamlss/R/bp.R` (529 行) | **中**：bucket plot（与 `ggplots` 中 `moment_bucket` 可共享计算核） |
| | `calibration` | `R/gamlss/R/lms.R` (396 行) | **中**：LMS 百分位曲线校准 |
| | `fittedPlot` | `R/gamlss/R/fitted-plot.R` (195 行) | **低**：参数拟合曲线展示 |
| | `get.rqres`, `rqres.plot` | `R/gamlss/R/rqresplot_new.R` (169 行) | **中**：随机化分位数残差提取与多实现诊断图 |
| 杂项（1） | `gamlssNews` | `R/gamlss/R/gamlss-5.R` | **极低**：版本新闻打印（可做轻量兼容封装） |

### 2.3 运行时能力矩阵（`omnilss.family_capabilities.capability_matrix()`）

- 注册族总数：**146**（`implemented`: **22**, `experimental`: **104**, `compat_alias`: **20**, `placeholder`: **0**）。
- Feature 级状态：
  - `production_safe`: **22 validated** / 124 experimental
  - `rs_fit`: **23 validated** / 123 experimental
  - `prediction`: **23 validated** / 123 experimental
  - `r_consistency`: **39 validated** / 107 unsupported（注：`highfreq-families-frozen.json` 统计 `_R_CONSISTENCY_EVIDENCE` 为 40 族，与运行时 `capability_matrix()` 的 39 族差 1，需核查键名/别名映射差异）
  - `cg_fit` / `sampling` / `smooth_terms`: **0 validated** / 146 experimental
  - `rs_jax_fit` / `ad_hessian`: **0 validated** / 6 experimental (`NO/GA/PO/BI/WEI/TF`) / 140 unsupported

---

## 3. 门禁脚本、测试与子项目实测发现（真实缺陷清单）

### 3.1 14 道自动化门禁实测结果

| 门禁命令 | 结果 | 发现与根因 |
|---|---|---|
| `tools/generate_symbol_coverage.py --check` | ✅ PASS | 工件与源码/测试 hash 同步 |
| `tools/coverage_diff_report.py` | ✅ PASS | 0 regression；但 `symbol-coverage-baseline.json` 仍停在 `missing=173`（当前 53），落后 120 项，棘轮保护松弛 |
| `tools/milestone_manifest.py --check` | ✅ PASS | M0 5 道门禁全绿 |
| `tools/highfreq_family_freeze.py --check` | ✅ PASS | 22 promoted / 0 pending / 18 excluded |
| `tools/m1_assessment.py --check` | ❌ **FAIL** | **漂移**：`golden_validated_total` 从 86→84（`dpqr_r_consistency` 2→4），因 D2.266 新增活 R 测试使 2 个 dpqr 符号被 `r_consistency_test` 优先级覆盖；`m1_gate_met` 仍为 `True`，产物未同步再生 |
| `tools/m2_assessment.py --check` | ✅ PASS | M2 四项判据全绿 |
| `tools/m3_assessment.py --check` | ✅ PASS | M3 6/6 判据全绿（含活体探针） |
| `tools/m3_admission_evidence.py --check` | ✅ PASS | 22/22 admitted，0 violations |
| `tools/m4_scope_inventory.py --check` | ✅ PASS | 8/8 周边包 surface 完整；但 `verdict.notes` 文案仍残留过时句子「`gamlss.mx` is the first package with a surface gap」 |
| `tools/contract_freeze.py --check` | ✅ PASS | family/link/formula/prediction 契约完整 |
| `tools/p2_defect_classification.py --check` | ✅ PASS | 90 个非 pass 项全归类（0 unclassified） |
| `tools/layered_parity.py --check` | ❌ **FAIL** | **漂移**：66 cell 分类不变（54 pass / 11 flat_ridge / 1 r_error），但 `('NO','MIXED')` `max_abs` 7.55e-10→**8.14e-11**、`('NO','RS')` `dev_rel` 8.57e-16→**4.29e-16**（精度提升反而触发精确浮点相等校验失败）；D2.244 后未重生成产物 |
| `tools/golden_manifest_gate.py --check` | ✅ PASS | 22 cases 校验通过 |
| `tools/ondil_contract_scan.py --check` | ✅ PASS | 22 族 0 序错位 / 0 支持失配 / 5 已知边界违反棘轮冻结 |
| `tools/pytest_shards.py --shards 8 --check` | ✅ PASS | 600 文件，每片 75 文件 |
| `tools/tiered_collection.py --mode collect --strict-skips` | ✅ PASS | **5,822 用例**，0 error / 0 skip |

**CI 接线缺口**：`.github/workflows/ci.yml` 与 `release.yml` 尚未接入 `m1_assessment`、`m2_assessment`、`m3_assessment`、`m4_scope_inventory`、`layered_parity`、`p2_defect_classification` 与 `ondil_contract_scan` 的 `--check` 门禁，因此上述两处漂移在提交时未被拦截。

### 3.2 子项目与运行时缺陷（本次实测暴露）

1. **`omnilss-pro/tests/test_client_tls.py` 存在 2 处真实失败**：
   - 复现：`.venv\Scripts\python.exe -m pytest omnilss-pro/tests -q` → `2 failed, 22 passed, 1 skipped`。
   - 根因：`test_client_tls.py:34-41` 仅在 `import grpc` 抛 `ImportError` 时注入 `_grpc_stub`（带 `.secure` 属性）；当环境安装了真实 `grpcio`（如本机 `.venv` 或 CI `omnilss[server,dev]`）时，`client._channel` 是真实 `grpc.Channel`，断言 `client._channel.secure` 直接抛 `AttributeError`。
   - 漏网原因：CI 的 `collection-gate` 对 `omnilss-pro` 只跑 `--mode collect`，未跑 `--mode run`！
2. **`omnilss-server` 测试覆盖极薄且存在工程隐患**：
   - 仅 1 个测试文件 1 个用例（`test_server_api.py`），未覆盖鉴权 (`OMNILSS_API_TOKEN`)、DoS 限流/包体限制、非法平滑方法拒绝、production-safe 族门禁拒绝。
   - `_RATE_LIMIT_STATE` 为无界内存 dict（按客户端 IP 持续增长无清理机制）。
3. **源码内部自触发 `DeprecationWarning` 与 `RuntimeWarning`**：
   - `omnilss/src/omnilss/fitting.py:1396, 1520` 与 `algorithms/rs_algorithm.py:1035` 仍在内部调用已废弃的 `fitting._parse_formula`，在拟合与测试收集阶段持续刷屏 `DeprecationWarning`。
   - `algorithms/rs_algorithm.py:702`：`abs(old_deviance - deviance) < tol` 在 `old_deviance == deviance == inf` 时触发 `RuntimeWarning: invalid value encountered in scalar subtract`（应先做 `np.isfinite` 守卫）。
4. **代码库膨胀与双轨架构债**：
   - `omnilss/src/omnilss` 现已达 **307 个文件、101,372 LOC**；出现多处超大单文件：`dpqr_functions.py` (9,156 行)、`ggplots.py` (3,452 行)、`fitting.py` (2,623 行)、`distributions.py` (2,593 行)、`__init__.py` (2,065 行)、`rs_algorithm.py` (1,816 行)。
   - `FamilyDefinition` 旧体系与 `core/distributions/DistributionProtocol` 新体系长期并行，维护成本高。
   - `mypy` 在 `ci.yml` 与 `release.yml` 中均处于 `continue-on-error` / advisory 状态。

### 3.3 文档与版本严重失真（阻塞 1.0 对外发布）

1. **`README.md` 严重过时**：仍写着「80+ Distribution Families」「≈105 registered / 31 validated」「30-day architecture stabilization phase」「ST1–ST4/TF2/DBI 未封装」「仅支持 pb/ps/cs，pbm/cy/fp/re 等未实现」，与当前 146 族、1025/1036 符号覆盖、M0–M3 达成完全矛盾。
2. **`HANDOFF.md` 停留在 2026-10-31（`6567edf`，D2.134）**：落后 136 个提交，仍列「71 个既有失败待收口」。
3. **`CHANGELOG.md` 停留在 `0.4.0`（2026-08-12）**：缺失 D2.92–D2.270 全部里程碑与功能增量。
4. **版本号停滞**：`omnilss/pyproject.toml` 与 `_version.py` 仍为 `0.4.0`，未按 M1 (`0.5.0`) / M2 (`0.6.0`) / M3 (`0.9.0`) 节奏推进。
5. **教程体系割裂且大量占位**：存在两套互不关联的目录——`docs/tutorials/`（8 篇 1–2 KB 短稿，`quickstart.md` 仍写「八核心 + 约 20 个族」，且未挂入 `mkdocs.yml` 导航）与根目录 `tutorials/`（118 篇，其中 `phase1/03..23`、`phase2/*` 多为 ~1.6–2.3 KB 提纲模板，`_cn.md` 多为 ~320 B 占位链接）。

---

## 4. 与参考项目对照（D2.270 现状更新）

| 能力维度 | OmniLSS (D2.270) | R `gamlss` 生态 | `gamlss-python` (0.2.0) | `ondil` (0.4.8) |
|---|---|---|---|---|
| 分布族数量 | **146**（22 生产准入） | ~120 导出族 | 34 | 26（含 6 多元） |
| 活 R parity 证据 | **强**（三闸 0 fail + 22 族 golden + 66 cell 分层） | 基准 | 弱 | 无（非目标） |
| family / link / `gen.Family` | **完整**（D2.169/D2.243 落地，logNO/logitNO 对活 R 1e-11） | 完整 | 有 | 部分 |
| 平滑项 | **pb/ps/cs/cy/pbm/pbo/pbp/pbz/pbc/pvc/fp/bfp/pp/pcat/ri/random/scs/tps/tensor**（缺 `re`/`gamlss.re`/`gamlss.lo` 与部分分支） | 完整 | 仅 pb/pbz | 无 |
| 周边包（cens/tr/mx/add/spatial/foreach/prepdata/ggplots） | **8 包 100% surface + 活 R 测试**（mx 等少数高级分支抛 `NotImplementedError`） | 完整 | 无 | 无 |
| JAX 可微 / GPU / Deep GAMLSS | **有**（CPU 已验；GPU `debianjaxgpu` 待实测；`RS_JAX` 仅 6 族实验） | 无 | 无 | 无 |
| **在线 / 流式估计** | **仅分布契约适配层（P5-1a/b）**；缺 `partial_fit`/`update`/遗忘因子/递推 Gram | 无 | 无 | **完整**（4 在线估计器） |
| **多元分布（MVN/MVT）** | **无**（仅有 `MULTIN` 多项分布） | MN3/4/5/MULTIN | 无 | **6 个多元分布 + 矩阵链接** |
| **在线正则化路径（CD / ElasticNet）** | 仅离线 `fit_lasso_coordinate_descent` | 无 | 无 | **在线 CD / LASSO / ElasticNet** |

**关键结论**：
- 对 **`gamlss-python`**：OmniLSS 已在分布、平滑、周边包、活 R 实证、可微与服务化上实现**全面反超**；仅需借鉴其基于 `patsy` 的复杂公式交互/因子对照简洁性。
- 对 **`R gamlss`**：表面覆盖已达 **98.94%**，剩余核心缺口是 **11 个缺失符号**、**104 个 experimental 族的证据晋级**、**未移植分支（如 `gamlss.mx` 协变量先验、`gamlss.cens`/`gamlss.tr` golden 语料、`cg_fit`/`smooth_terms` 能力准入）**。
- 对 **`ondil`**：分布契约桥接（P5-1a/b）已打通，核心能力差在 **在线/流式估计器（P5-1c）** 与 **多元分布（MVN/MVT + 矩阵链接）**。

---

## 5. 本次临时/无用文件清理台账（2026-11-21 已执行）

共清理 **2,480 个临时/缓存/冗余文件，释放 16.61 MB**，同时严格保护了所有测试承重文件与参考/运行环境。

| 清理对象 | 文件数 | 释放空间 | 性质与说明 |
|---|---:|---:|---|
| `tmp/`（**保留 5 个承重 R 参照文件**） | 567 | 5,472.6 KB | 清理全部一次性探针脚本、分片日志、sqlite 锁文件、临时 JSON/CSV/txt（保留 `bi_dbi_fixture_r.json`, `bi_dbi_ref.R`, `d223_c_ref.csv`, `d223_full_ref.csv`, `d223_ref.R` 防测试静默 skip） |
| `omnilss/tmp/` | 5 | 32.7 KB | gitignored `.pytest_cache` |
| `.hypothesis/` + `omnilss/.hypothesis/` | 721 | 535.2 KB | gitignored Hypothesis 测试缓存 |
| `.reasonix/` | 130 | 42.2 KB | gitignored Reasonix 工具会话运行态 |
| `.kiro/` | 5 | 56.6 KB | gitignored 2026-07 旧 IDE Release 1 草案（已被 `DEVELOPMENT_PLAN.md` 取代，无代码读取） |
| `benchmarks/results/raw/` + `reports/` | 46 | 779.6 KB | gitignored 2026-05~07 旧基准原始输出（保留已跟踪的 `gpu_crossover_analysis*.md`） |
| `benchmarks/diagnostic/*.csv` | 12 | 498.5 KB | gitignored 旧诊断临时 CSV |
| `Rplots.pdf` | 1 | 109.0 KB | R 绘图误生成的未跟踪临时 PDF |
| `omnilss/uv.lock` | 1 | 608.0 KB | `.gitignore` 明确排除的库项目锁文件 |
| `docs/development/placeholder-migration-queue-2026-10-03..11-08.md` | 18 | 106.0 KB | 18 份内容重复的逐日 0-placeholder 历史快照（保留被 `test_coverage_scope.py` 校验的最新 `2026-11-10.md`） |
| 项目内 `__pycache__` / `.pytest_cache`（37 目录） | 974 | 8,766.6 KB | Python 编译字节码缓存（严格排除 `.venv`/`debianjaxgpu`/`R`/`ondil`/`gamlss-python`） |
| **合计** | **2,480** | **16.61 MB** | 受保护目录 `.venv` / `debianjaxgpu` / `R` / `ondil` / `gamlss-python` / `.codex` / `.workbuddy` 全部核验完好 |

> 注：`tutorials/datasets/*.csv`（11 文件，~280 KB）虽被 `.gitignore` 的全局 `*.csv` 忽略，但为教程配套示例数据集，予以保留；`docs/development/` 中历史阶段审计文档保留作演进追溯（建议后续阶段归档至 `docs/archive/`）。
