# OmniLSS 全量 GAMLSS 迁移：2026-09-25 全面复审与完整开发计划

> **复审日期**：2026-09-25
> **复审对象**：当前工作树、`R/` 原始源码、`omnilss/`、`gamlss-python/`、`ondil/`、测试/工具/证据产物。
> **文档定位**：全量 R GAMLSS 迁移专项的执行基线；项目总纲仍以根目录 `DEVELOPMENT_PLAN.md` 为准。
> **重要限制**：本次复审不覆盖用户工作树中未提交的实现修改；这些修改必须先通过本计划的回归和证据门禁，不能直接计入完成量。

## 1. 复审结论

OmniLSS 已具备可运行的 JAX 分布回归核心、R golden 基础设施、RS/CG/MIXED 入口、平滑/预测/服务协议和 capability fail-closed 机制；但“全量迁移”仍未完成。当前应使用证据层级而不是注册数量管理进度。

### 1.1 可复现的当前基线

| 对象 | 复审结果 | 证据/解释 |
|---|---:|---|
| OmniLSS 版本 | `0.4.0` | `omnilss/pyproject.toml` |
| registry family | `146` | `docs/development/r-export-coverage.json`，生成时间 2026-09-25 |
| implementation status | implemented 1 / experimental 102 / compat_alias 25 / placeholder 18 | 同一 coverage JSON |
| DPQR evidence `VALIDATED` | 35 | coverage JSON 的 `evidence_tier_dpqr` |
| fitting evidence `VALIDATED` | 2 | coverage JSON 的 `evidence_tier_fitting` |
| R 源码目录 | 13 个包目录，12 个 `NAMESPACE` | `R/` 实际扫描 |
| R 文件规模 | 363 个 `.R` 文件 | `R/` 实际扫描；`gamlss` 79、`gamlss.dist` 124 |
| R 包版本 | `gamlss` 5.5-3；`gamlss.dist` 6.1-8 | `R/*/DESCRIPTION` |
| OmniLSS 测试文件 | 417 个 `test_*.py` | `omnilss/tests` |
| gamlss-python | 52 个 `.py` 源文件，版本 0.2.0 | `gamlss-python/pyproject.toml` |
| ondil | 119 个 `.py` 文件，版本 0.4.8 | `ondil/pyproject.toml` |
| 本机环境 | Python 3.10.19 / JAX 0.6.2 CPU / NumPy 2.2.6 / SciPy 1.15.3 / R 4.6.0 | 2026-09-25 本机检查 |
| x64 状态 | 当前 `jax_enable_x64=False` | parity 前必须显式打开并在日志中确认 |
| 全库收集 | 本轮命令未在工具时限内完成 | 不得宣称零 collection error 或全库通过 |

### 1.2 必须修正的旧结论

1. 2026-09-11 的 `145 / 84 / 35` 状态已经过时；当前应引用 2026-09-25 生成的 `146 / 102 / 18`。
2. `tools/generate_r_export_coverage.py` 当前只读取 Python capability registry，**没有解析 R NAMESPACE、R 函数定义、Python 导出、测试和 golden manifest**；现有 JSON 是 registry inventory，不是真实 R↔Python 逐符号 coverage。
3. 根 `pytest.ini` 已排除 `ondil`、`gamlss-python`、`debianjaxgpu`；旧文档“43 个 ondil/临时脚本收集错误”不能作为当前事实，必须分别报告各项目收集结果。
4. `family_capabilities` 的 metadata 与 R 数值等价证据不是同一概念；名称可解析不代表 Kernel、Parity 或 Admission 已完成。
5. `gamlss-python` 是 NumPy/SciPy 转录参考，`ondil` 是在线估计参考；两者均不能替代 R parity 或 JAX kernel。

## 2. 统一完成定义：四层证据 + 五类对象

对每个分布 family、核心 API、smoother、算法和服务 route 分别记录：

1. **Surface**：可导入、可解析、参数名/link/support 元数据正确；
2. **Kernel**：真实公式实现，无 placeholder、同名克隆、静默近似或错误参数化；
3. **Parity**：固定 R 源码/包版本、参数点、输入、seed、dtype 下通过 d/p/q/r 或 API 对照；
4. **Admission**：在目标方法、fit、newdata、序列化和服务路由中通过 fail-closed 门禁。

| 对象 | 必须记录的证据 |
|---|---|
| family | R family、R `d/p/q/r` 文件/函数、参数化、support、link、Python factory、四层状态 |
| DPQR | 固定点、边界、单调性/归一化、upper-tail/log-p、随机抽样、异常行为 |
| fitting | RS/CG/MIXED、系数、fitted、deviance、迭代、收敛/失败原因 |
| smoother | parser、basis、penalty、lambda、EDF、fit、newdata、R golden |
| service | family/method/smoother admission、错误码、资源限制、序列化往返 |

`implemented` 只表示达到 production-safe；`experimental` 不得自动进入生产路由；`compat_alias` 不计独立 R family；`placeholder` 必须明确拒绝或保持可见失败。

## 3. 对照项目审核与边界

### 3.1 R 原始源码

R `gamlss.dist` 6.1-8 负责 family、参数链接、support、d/p/q/r、导数、矩和边界；R `gamlss` 5.5-3 负责 formula、RS/CG/混合拟合、预测、诊断、模型选择和平滑。每项任务必须先建立 `R source map`，标注文件、函数、参数检查和特殊分支。

### 3.2 gamlss-python

可借鉴公式转录、参数说明、解析导数、SciPy 特殊函数策略、RS/CG 控制参数、`pb/pbz` 和 R reference 流程。不可直接复用其 NumPy/SciPy kernel、随机数语义、dtype、自动微分、API 或服务路由。必须记录 `R 文件 → gamlss-python 文件 → OmniLSS JAX 文件`。

### 3.3 ondil

`ondil` 只进入 L2 在线工作流：可共享参数、support、link、score、信息矩阵 contract，但离线 R golden 不证明在线 estimator 正确；还必须验证 chunk 顺序、warm-start、漂移、缺失值、并发、性能和内存。

## 4. 当前差距与优先级

### 4.1 分布层

当前重点是 18 个 placeholder、25 个 alias、大量仅有 Surface/实验 Kernel 的 family，以及仅 35 项 DPQR evidence 为 VALIDATED。优先风险为：ST/SEP/SST 独立参数化和 q 反演；Box-Cox `o`、BCCG/BCT/BCPE 截断与方差标准化；DEL/DPO/SI/WARING/YULE/NBF/GPO/NBII 离散递推；ZALG/ZIP/ZIP2/ZIPF 零质量与尾部；GIG/NO2/NOF/PARETO/PE2/RGE/WEI2/EXGAUS/EGB2/LOGSHASH/LQNO/MN 特殊函数、长尾和占位实现；ZASICHEL/ZISICHEL、ZAGA/GU 的 NaN 与收敛问题。

### 4.2 拟合、核心 API 与平滑

优先保证 link/family 构造、`Formulae2data`、`predictAll`、update/refit、诊断/残差/hatvalues/vcov/logLik、GAIC/stepwise，再逐个准入 smoother。已有 `pb/ps/cs` 不能推导其它 smoother 已完成。

### 4.3 工程与证据

覆盖生成器、R artifact、x64 启动检查、collection 报告和 CI fail-closed 是所有后续统计工作的阻塞项。

## 5. 完整开发计划（WBS）

### P0：审计基座与可重复验证（1–2 周，硬阻塞）

**P0.1 真实 R↔Python 覆盖矩阵**：解析 12 个 `NAMESPACE` 的 exports/S3/S4/import、R 文件函数定义、Python registry/exports、测试路径和 golden manifest。输出逐符号 JSON、Markdown、placeholder queue；字段至少包含 `r_package/r_symbol/r_file/python_symbol/python_file/surface/kernel/parity/admission/evidence/commit`。旧工具必须改名或明确标注为 `registry-inventory`。

**P0.2 R golden manifest**：固定 R/Python/JAX/NumPy/SciPy 版本、本地源码 hash、x64、seed、参数点、容差和命令；R 不可用时只能标 `smoke-only`。

**P0.3 测试收集分层**：分别执行并保存 `omnilss/tests`、`ondil/tests`、`gamlss-python/tests` 和根 tests 的 collected/error/skipped；目标每组 zero collection error。

**P0.4 环境与 x64 门禁**：parity/golden 入口强制检查 Rscript、R 包、`JAX_ENABLE_X64=True`、backend、版本和 dirty worktree；Windows CPU、WSL GPU、CI 各有 manifest。

**P0.5 自动差异报告**：CI 对 R missing、Python extra、alias、placeholder、无 evidence、registry 与 JSON 不一致失败。

**P0 验收**：coverage 可从源码重生成；inventory 与真实 coverage 分离；分层 collection 无错误；golden 可复现；未验证项可定位。

### P1：DPQR kernel 与 R parity（8–12 周）

每族执行：R source map → 参数规范 → JAX kernel → support/boundary/property → R d/p/q/r artifact → 状态晋升。

| 批次 | 范围 | 退出条件 |
|---|---|---|
| P1-A | ST/SEP/SST | 独立参数化、尾部、q 反演、R golden |
| P1-B | BEINF/BEZI/BEOI/LG/LNO/LOGITNO/NET、Box-Cox-o | 混合质量、截断项、边界、normalization |
| P1-C | DEL/DPO/SI/WARING/YULE/NBF/GPO/NBII | 递推、整数 support、极端分支 |
| P1-D | ZALG/ZIP/ZIP2/ZIPF 及 zero variants | 零质量、尾部、log-p、采样 |
| P1-E | GB/GIG/NO2/NOF/PARETO/PE2/PIG2/RGE/WEI2 | 特殊函数、正实 support、尾部稳定 |
| P1-F | EXGAUS/EGB2/LOGSHASH/LQNO/SN/MN 及余项 | 长尾、矩、广播、异常、dtype |

每族必须通过固定点、边界、单调性/归一化、广播、upper-tail/log-p、抽样和实时 R 对照；无 R 结果只能标 `smoke-only`。

### P2：全 family 拟合 parity（6–10 周）

建立连续、计数、zero-inflated、边界、极端值数据集。每族至少一个 RS R golden；CG/MIXED 分层覆盖。记录系数、fitted、deviance、迭代、收敛/失败原因、warnings 和 dtype。必须修复 NaN-as-convergence、溢出伪成功、MIXED 路由不可达、ZAGA/GU 差异、链接尺度热启动污染和拟合/预测参数化不一致；失败不得用 skip 隐藏。

### P3：`gamlss` 核心 API 与 smoother（10–16 周）

顺序：family/link/check/构造；formula/`Formulae2data`/`predictAll`/update/refit/newdata；residuals/diagnostics/hatvalues/vcov/logLik/deviance；GAIC/stepwise/find.hyper/profile deviance；最后准入 `lo/random/re/ri/pcat/fp/bfp/cy/pvc/pbm/pbz/pbo/pbc/pbp/scs`。每个 smoother 必须通过 parsing、basis、penalty、lambda/EDF、fit、newdata、R golden、admission 八项检查。

### P4：周边 R 包（12–20 周）

- D1：`gamlss.tr`、`gamlss.cens`，打通 censor/truncation；
- D2：`gamlss.mx`、`gamlss.add`，先有限混合/NP，再 GA/BA/NN/FK/TR；
- D3：`gamlss.spatial`、`gamlss.foreach`、`gamlss.prepdata`；
- D4：`gamlss.ggplots` 数据层、`gamlss.data`、WeatherGermany；外观不要求像素一致；
- D5：`gamlss2` 单独规格/版本/证据，不阻塞 1.x/1.0。

### P5：ondil 在线层（可与 P1/P2 并行，独立发布）

先定义共享 distribution contract，再做离线等价初始化、chunk 顺序不变性、warm-start、漂移、缺失值、并发、性能/内存基线和流式 golden。L1 R golden 只证明 kernel，不证明在线 estimator。

## 6. 时间、依赖与并行安排

| 周期 | 主线 | 依赖 |
|---|---|---|
| W1–W2 | P0 审计/环境/收集 | 阻塞 P1/P2/P3 |
| W3–W6 | P1-A/B/C + P2 数据集/诊断 | P0.1–P0.4 |
| W7–W10 | P1-D/E + P2 RS | P1 contract、R runner |
| W11–W14 | P1-F + P2 CG/MIXED | DPQR parity 基线 |
| W15–W20 | P3 family/core/prediction/diagnostics | P1/P2 admission |
| W21–W28 | P3 smoother + P4-D1 | 核心 API golden |
| W29–W40 | P4 D2–D5 + 发布 | 核心 API 稳定 |
| 全程 | P5 ondil、CI、文档、benchmark | contract 和独立门禁 |

时间为容量估计，不是完成承诺；R artifact、失败分类或 admission 不齐只能保持 `experimental`。

## 7. 里程碑与发布门槛

| 里程碑 | 门槛 | 建议版本 |
|---|---|---|
| M0 | P0 完成；真实 coverage 可重生成；分层 collection 无错误；x64/R manifest 完整 | 0.4.x |
| M1 | 目标 R family 全部 Surface+Kernel+DPQR parity；无 placeholder 进入目标范围 | 0.5.0 |
| M2 | RS 全覆盖；CG/MIXED 分层 parity；失败可解释；fit/newdata 证据齐全 | 0.6.0 |
| M3 | 核心 API、诊断、模型选择和准入 smoother 闭环 | 0.9.0 |
| M4 | D1–D4 周边包、服务、文档、教程、发布 artifact 同步 | 1.0.0 |

每个里程碑必须有 `--require-r`、固定 manifest、无隐性 skip、coverage diff、目标/相关 suite 和 collection 日志、中文任务提交、可复现命令。性能只能在一致性 gate 后运行，并区分 cold/warm、CPU/GPU 和 R 可用性。

## 8. 每个任务交付模板

1. R source map；2. gamlss-python/ondil reference map；3. JAX dtype/可微性/特殊函数/异常策略；4. kernel/registry/API/admission；5. unit/property/R comparison/fit/newdata/service tests；6. JSON/manifest/环境/hash/seed/命令/commit；7. 四层证据满足后更新状态；8. 目标 suite、相关 suite、分层 collection 和 `git diff --check`。

## 9. 立即执行的前十五项任务

- [ ] P0.1 将 `generate_r_export_coverage.py` 拆为真实 coverage generator 与 registry inventory generator；
- [ ] P0.2 解析 12 个 NAMESPACE、R 函数定义和 S3/S4 exports，生成逐符号 schema；
- [ ] P0.3 重新生成 146 family 的 JSON/Markdown/placeholder queue，并校验运行时一致；
- [ ] P0.4 建立 omnilss/ondil/gamlss-python 分层 collection 命令和零错误门禁；
- [ ] P0.5 固定 Windows CPU、WSL GPU、CI 的 R/Python/JAX/NumPy/SciPy/x64 manifest；
- [ ] P0.6 修正 `BEo/PARETO2o`、全部 `o` 变体、alias/extra/missing 的自动差异报告；
- [ ] P1.1 为 registry 项映射真实 R family、参数、support、link 和函数集合；
- [ ] P1.2 完成 ST/SEP/SST 的 R golden 与独立 JAX kernel；
- [ ] P1.3 完成 Box-Cox-o 与 BEINF/zero-adjusted 批次；
- [ ] P1.4 完成离散递推与 zero-inflated 批次；
- [ ] P1.5 完成 GIG/NO2/NOF/PARETO/PE2/RGE/WEI2 连续 placeholder 批次；
- [ ] P2.1 建立全族最小拟合数据集和统一 convergence diagnostic schema；
- [ ] P2.2 修复 NaN/overflow/MIXED 路由并加入回归；
- [ ] P3.1 冻结 family/link/formula/prediction contract；
- [ ] P5.1 建立 OmniLSS↔ondil distribution contract 与独立流式 golden。

## 10. 风险、决策与当前状态

- **范围风险**：146 registry 项不等于 R family；采用逐符号/逐函数/逐场景证据，gamlss2 不阻塞 1.0。
- **数值风险**：x64、JAX 版本、特殊函数、R 与 Python 优化器差异；所有失败必须分类。
- **证据风险**：inventory 冒充 coverage、R 不可用时 skip、静态 null golden；P0 为硬门禁。
- **工程风险**：收集超时不能等同于收集通过；采用分层日志和超时定位。
- **参考项目风险**：gamlss-python 只作转录 oracle，ondil 只作在线设计 oracle，不改变 R 优先级。
- **发布决策**：服务生产路由继续 fail-closed；placeholder/alias/experimental 默认不 admission；benchmark 不能先于 parity。

当前专项状态：**P0 进行中，P1/P2/P3/P4/P5 待依赖门禁后执行**。根 `DEVELOPMENT_PLAN.md` 的 Phase 0–4“完成”只表示 Release 1 阶段目标，不代表全量 R GAMLSS 迁移完成。
