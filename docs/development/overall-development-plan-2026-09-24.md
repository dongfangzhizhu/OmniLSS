# OmniLSS 整体项目进度审查与完备开发计划

> **审查日期**：2026-09-24  
> **审查范围**：`omnilss/` 主实现、`R/` 中 GAMLSS 源码、`gamlss-python/`、`ondil/`、测试/CI/文档/基准、Windows `.venv`；`debianjaxgpu/` 仅审查其定位和相关配置，未在本轮启动 WSL GPU 基准。  
> **文档用途**：作为当前工作树的事实基线、优先级队列和阶段验收计划。历史迁移清单仅用于追踪，不应替代本文件中的能力证据口径。

## 1. 执行摘要

OmniLSS 已不是概念验证：主包已有可运行的公式 API、NumPy RS 拟合器、实验性 CG/MIXED/JOINT/L-BFGS/JAX RS 路径、146 个注册 family、`pb/ps/cs` 平滑链、schema-safe newdata 预测、诊断、模型工件以及 HTTP/gRPC 服务。然而，项目当前仍应定性为 **Beta 阶段的“一个可信生产基线 + 大范围实验迁移”**，尚不能宣称完整迁移 R GAMLSS。

关键结论：

1. **主路径真实可用，但生产支持面很窄**：NumPy RS 是当前最成熟算法；能力矩阵中仅 `NO` 为 `implemented`/`production_safe`，仅 `NO`、`ZIP` 有 RS 拟合和端到端预测的 validated 证据。
2. **分布广度远高于验证深度**：运行时实测 146 个注册 family，其中 1 个 implemented、100 个 experimental、25 个 compat alias、20 个 placeholder。注册成功不能等价为数值正确、R 一致或可生产。
3. **R 兼容存在基础语义缺口**：当前 `MIXED` 是自动选择 RS 或 L-BFGS，而 R `mixed()` 是严格的 `RS(n1) → CG(n2)`；主公式层未形成 R 等价的逐参数 `offset()`；输入 NA、weights 合同、完整 links 和复杂 formula 语义也需统一。
4. **平滑器的代码准入与历史报告冲突**：代码把 `pb/ps/cs` 标为唯一完整准入项，但最新提交的 live-R 报告中三者一致性为 0/3。这些能力可以保留 experimental 路由，但在重新生成全绿证据前不能描述为 R 等价或 production-safe。
5. **JAX/GPU 是技术储备，不是当前优势结论**：RS_JAX 只覆盖 `NO/GA/PO/BI/WEI/TF`，不支持 smooth，默认 GPU crossover 为无限阈值，已有说明称目标硬件上尚未证实加速。应先保证统计正确性，再优化设备路径。
6. **工程框架完整，但发布门禁仍有高风险缺口**：常规 CI 只运行测试子集；GPU job 手动、允许失败；mypy 不阻断；server extra 中存在疑似错误依赖 `httpx2`；序列化默认 `save_model` 与 `load_model` 使用不同格式。
7. **项目下一阶段不应继续扩充名称数量**。首要工作是修复 P0 合同和发布问题，建立可生成的能力真相源，然后按高频 family 扩大 DPQR→拟合→预测→工件→R parity 的端到端交集。

## 2. 审查方法与本轮证据

### 2.1 审查方法

- 追踪 `omnilss.gamlss()` 到 RS、CG、MIXED、JOINT/L-BFGS 和 RS_JAX 的路由与模型构造。
- 检查 family registry、运行时 capability matrix、placeholder/alias 判定和平滑 admission。
- 对照 `R/gamlss/R/gamlss-5.R` 的 RS/CG/mixed、weights、offset、model matrix 和结果对象语义。
- 对照 `gamlss-python` 的 R 行为翻译，以及 `ondil` 的 sklearn/online/regularization 设计。
- 检查 pytest 组织、GitHub Actions、发布、文档、devcontainer 和已提交基准报告。
- 使用当前 `.venv` 做能力矩阵和三组关键测试实测；未把历史报告或文件存在本身视作当前成功证据。

### 2.2 实测环境与结果

| 项目 | 本轮结果 |
|---|---|
| Windows Python | 3.10.16（`.venv`） |
| JAX | 0.6.2，默认 backend 为 CPU |
| OmniLSS | 0.4.0 |
| R | 4.6.0 |
| family 总数 | 146 |
| implementation 状态 | 1 implemented / 100 experimental / 25 compat_alias / 20 placeholder |
| production-safe | `NO` |
| RS validated | `NO`, `ZIP` |
| prediction validated | `NO`, `ZIP` |
| 定向测试 | `test_family_capabilities.py`、`test_serialization.py`、`test_prediction.py` 共 41 项通过 |

定向测试出现 40 次 `_parse_formula` 弃用警告和 1 次 Hypothesis 收集配置警告；本轮未执行完整 pytest、完整 live-R parity、文档构建或 GPU 测试，因此不能据此宣称全库通过、全 R 一致或 GPU 可用。

### 2.3 历史实证的解释

`benchmarks/results/reports/consistency_report_20260726_121056.md` 是仓库中最新可见的 live-R 综合报告：42 项中 30 项通过、12 项失败；DPQR 19/27、拟合 11/12、平滑 0/3。失败包含 `BI` d/p/q、`GU` p/q、`NBI` p/q、`PO` q 和 RS fit，以及 `pb/ps/cs`。该报告是重要缺陷证据，但不是当前 commit 的重新运行结果；下一里程碑必须生成带 commit、环境和 R 包来源的新报告。

## 3. 当前架构与成熟度

### 3.1 核心执行路径

`omnilss/src/omnilss/fitting.py` 中的 `gamlss()` 是统一入口：解析 family 和公式，进行方法/能力检查，然后路由至：

- `RS`：`algorithms/rs_algorithm.py`，NumPy 交替参数 IRLS；支持 fixed parameter、weights、四参数设计、step-halving、平滑 lambda 更新和收敛遥测。
- `RS_JAX`：fully traced JAX 路径，仅六个核心 family，不支持 smooth，当前为 experimental。
- `CG`：full Hessian 或 IRLS-cross；默认可先做短 RS warm start，尚无普遍 parity 证据。
- `MIXED`：调用 `mixed_fit(..., algorithm="auto")`，并非 R mixed 语义。
- `JOINT/LBFGS`：先以 `gamlss_ml()` 生成初值，再进行联合优化；属于实验扩展。

统一返回 `GAMLSSModel`，由 prediction、diagnostics、serialization 和 API 层复用。HTTP/gRPC 端点调用真实核心拟合，不是 mock；但服务层的成熟度不能超过它所准入的 family/method 证据。

### 3.2 能力成熟度表

| 轨道 | 当前成熟度 | 已有可信部分 | 主要缺口 |
|---|---|---|---|
| NumPy RS | 中高 | 完整迭代骨架、数值保护、诊断元数据；NO/ZIP 有验证证据 | 大多数 family 缺端到端 R parity；输入合同、offset、复杂公式仍缺 |
| CG | 中低/实验 | 两类真实实现、cross derivative/Hessian 路径 | 全族证据不足；warm-start 改变纯 R CG 轨迹；不支持 smooth |
| MIXED | 低/语义不兼容 | 能路由并返回收敛模型 | 不是 R 的 RS→CG；能力矩阵错误地借用 `cg_fit` 粒度 |
| JOINT/L-BFGS | 实验 | 真实优化器与 warm start | smooth/族支持、收敛合同、能力证据不足 |
| RS_JAX/GPU | 实验 | 六族 JAX traced 路径、设备路由框架 | 无 smooth；无稳定 crossover；GPU job 非强制；未证实性能优势 |
| 分布族 | 广而浅 | 146 个注册名，R consistency 测试框架存在 | 20 placeholder、25 alias；仅 NO production-safe；DPQR 历史失败 |
| Links | 部分 | identity/log/logit/probit/inverse/cloglog/sqrt 等 | 缺 cauchit、mu²、shifted/bounded、power、custom own 等 R 语义 |
| Formula/design | 中等 | 多参数 formula、基础交互、factor、schema 保存 | 复杂 transform/contrast/cbind/offset 和训练-预测一致性未达 R/patsy 水平 |
| Smoother | 部分 | `pb/ps/cs` 形成 parse→fit→predict 状态链 | 最新 live-R 三者全失败；random/re/ri 等尚未准入；部分代码不可达 |
| Prediction | 中等 | 参数/分位数/centile、因子和平滑 schema fail-closed | validated 仅 NO/ZIP；`terms` 未实现；固定构造参数使用不统一 |
| Diagnostics/selection | 中等偏低 | 大量 API 和结构化诊断数据 | 很多为 staged port；依赖 DPQR 正确性；工件加载后残差/设计不完整 |
| Serialization | 中等 | safe ZIP+JSON/NPZ、schema/zip bomb 检查 | 默认 save/load 格式不配对；版本 0.3 vs 包 0.4；构造参数恢复不通用 |
| sklearn/API | 中等 | `GAMLSSRegressor` 和 HTTP/gRPC 基本链路 | sample_weight/partial_fit、严格 production admission、API 契约需完善 |
| Online/regularization | 参考或独立实验 | 有 standalone regularization；ondil 可参考在线更新 | 尚未接入每参数 GAMLSS IWLS；无主包 online contract |

## 4. 与 R 及参考项目的差距边界

### 4.1 必须以 R 为基准的能力

以下是迁移正确性的硬合同，必须直接对照 `R/` 源码和固定 R artifact：

1. family 参数化、support、links、初值、d/p/q/r、score/Hessian/cross derivative；
2. RS、纯 CG、`mixed(n1,n2)=RS(n1)→CG(n2)` 的可观察语义；
3. likelihood weights、每参数 offset、NA 策略、固定参数、binomial denominator；
4. 四参数公式、factor/contrast/interaction、model matrix 和 newdata 列稳定性；
5. smooth backfitting、lambda/EDF、预测外推；
6. deviance、rqres、vcov、GAIC 和模型选择的数值定义。

### 4.2 `gamlss-python` 的用途

可借鉴：基于 patsy 的 R 风格公式、`offset()`、输入 NA/weights 校验、RS/CG/mixed 翻译、结果对象槽位和 golden 流程。  
不可作为最终证据：它是 NumPy/SciPy 参考实现，不能替代 R 源码或 OmniLSS 的 JAX 可微/设备要求。

### 4.3 `ondil` 的用途

可借鉴：sklearn estimator 合同、`sample_weight`、按参数 equation/method、lasso/elastic-net 路径、在线 `update()`、forget factor、缩放和调试遥测。  
不属于 R parity：在线更新、正则路径、流式状态是 Python 扩展，应建立独立验收，不混入 R 迁移完成率。

## 5. P0 阻断问题

这些问题应在继续扩大 family 或 GPU 工作前解决。

| 编号 | 问题 | 影响 | 完成条件 |
|---|---|---|---|
| P0-01 | `save_model = save_model_pickle`，`load_model = load_model_json` | 自然配对调用必然失败，且安全语义混乱 | 默认 API 使用同一安全格式；pickle 改为显式 trusted API；round-trip 契约通过 |
| P0-02 | 工件版本常量 0.3.0，包版本 0.4.0 | 兼容治理漂移 | 区分 package/schema 版本，定义迁移矩阵并验证旧工件 |
| P0-03 | `MIXED` 不等于 R mixed | 核心算法名称误导 | 实现 `RS(n1)→CG(n2)` 兼容模式；自动混合策略另命名 |
| P0-04 | 缺统一 NA/weights/offset 合同 | 基本 R 行为和数值安全不可靠 | 所有路由统一 fail-fast；逐参数 offset 保存并参与 fit/predict |
| P0-05 | production route 可允许普通 experimental | 服务外观超出统计保证 | production 默认要求 `production_safe=validated`；实验入口显式隔离/标记 |
| P0-06 | server extra 为疑似错误包 `httpx2` | 依赖供应链/安装风险 | 核实并改为正确 pinned/约束依赖；安装 smoke 通过 |
| P0-07 | CI 引用疑似不存在的 `test_smooth_outer_loop.py` | 常规 CI shard 可能直接失败 | 修正测试清单并为 PR 建立稳定全量/分层 suite |
| P0-08 | devcontainer 引用疑似缺失 `.codex/setup.sh` | quick-R parity 容器可能无法创建 | 恢复脚本或调整 postCreate；CPU 容器从零构建成功 |
| P0-09 | capability 状态依赖运行时 AST 标记 | 状态可审计性和可重复性不足 | 生成并版本化静态能力快照；CI 检查与代码一致 |
| P0-10 | 历史文档宣称“八核心族/三算法与 R 等价” | 与能力矩阵及 live-R 报告冲突 | 全局校准 README、商业化和迁移文档，仅按 validated 证据声明 |

## 6. 完备开发路线图

计划按“正确性基座 → R 核心闭环 → family 扩展 → smoother/API → JAX/GPU → 扩展生态 → 稳定发布”推进。周期是 2–4 人团队的相对估算；里程碑以验收门禁结束，不以日期自动结束。

### Phase 0：事实基线与发布止血（1–2 周）

**目标**：消除 P0 语义和工程风险，建立唯一可重生成的进度真相源。

任务：

1. 修复序列化默认配对、工件版本和 family constructor state。
2. 修复 `httpx2`、CI 失效路径、devcontainer setup、docs 依赖/仓库 URL 漂移。
3. 定义 `CapabilityRecord` 的五层状态：registered、kernel-complete、R-validated、route-supported、production-safe；生成 JSON 和 Markdown。
4. 统一输入 validation，在 RS/CG/MIXED/JOINT/LBFGS/RS_JAX 前执行。
5. 实现 R-compatible mixed；保留现有自动算法时改名为 `HYBRID` 或 `AUTO_OPTIMIZER`。
6. 修订用户可见声明，明确 NO production baseline、其他 experimental。

**验收门禁**：

- 定向核心测试、完整 CPU pytest、R required suite 均无意外 skip；
- 默认模型保存→加载→newdata 预测等价；不可信 pickle 不走默认入口；
- capability snapshot 可重复生成且工作树无漂移；
- CI/devcontainer 从干净环境可执行；
- production API 拒绝非 production-safe family。

### Phase 1：R 公式与拟合核心合同（3–5 周）

**目标**：先让核心算法和数据语义正确，再扩大 family。

任务：

1. 引入或对齐 patsy/formulaic 级 design-info：factor、contrast、transform、interaction、`cbind()`、每参数 `offset()`。
2. 实现统一 DataContract：长度、shape、dtype、finite、NA、weights、support、binomial denominator。
3. 建立 pure RS、pure CG、R mixed 三条明确模式；把 warm-start/stabilized CG 作为显式扩展选项。
4. 对齐 step、初值、fixed parameter、deviance、迭代/收敛原因、noObs/df/penalty。
5. 将模型结果拆成稳定的 FitState、DesignState、FamilyState、DiagnosticsState，避免旧 R 端口模块各自解释状态。

**验收门禁**：NO 在无权重/权重/offset/factor/多参数公式/新数据场景下与 R golden 一致；pure CG 与 mixed 的算法轨迹可解释；非法输入在所有路由产生同类结构化错误。

### Phase 2：高频 family 端到端晋级（6–10 周）

**目标**：把高频分布从 experimental 提升为真实可发布能力，而不是继续增加注册名。

按业务价值和技术依赖分批：

- **2A 基础族**：`GA/PO/BI/NBI/NBII/BE/WEI/TF/IG/LOGNO`；
- **2B 零模型/混合族**：`ZIP/ZAGA/BEINF` 及常用零调整族；
- **2C 生长曲线族**：`BCCG/BCT/BCPE`；
- **2D 偏态厚尾族**：`SHASH/GT/SN/SEP/ST` 中高频且可验证者；
- **2E 剩余正实数、离散、截断和特殊函数族**。

每个 family 必须走同一交付流水线：

1. R source map 和参数/support/link 规范；
2. 无 placeholder/fallback 的 d/p/q/r kernel；
3. broadcasting、log/upper-tail、边界、单调性/归一性、随机抽样；
4. score/Hessian 与有限差分或 AD 验证；
5. RS fit、预测、rqres、工件 round-trip；适用时 CG/mixed；
6. 固定 R golden + live-R 门禁；
7. capability 晋级及用户文档。

**验收门禁**：每个晋级 family 的 DPQR、拟合、预测和工件均有独立证据；placeholder 和 alias 不计完成数；R 不收敛/不适用须记录 rationale，不以 skip 掩盖。

### Phase 3：Smoother、随机效应与正则化（6–10 周）

**目标**：建立可组合的 penalized term 协议，而非继续堆积专用分支。

顺序：

1. 重新验证 `pb/ps/cs`，首先解决最新报告的 0/3 parity；
2. `random/re`：随机截距→随机斜率、方差/EDF、newdata 未见 level 策略；
3. `ri`：把 ridge 作为参数模型的一等惩罚项；
4. `pbz/pbm/pbc/pvc/cy/lo/pcat`；
5. tensor/空间/时间等高级 smoother，按产品范围取舍。

统一 smoother admission 七项：parsing、basis、penalty、lambda/EDF、fitting、prediction、R golden。任一缺失则 fail-closed。

正则化借鉴 ondil，但接入 OmniLSS 每个分布参数的 IWLS：ridge→lasso→elastic-net，明确 intercept 惩罚、路径选择和 GAIC/CV 规则。

**验收门禁**：`pb/ps/cs` 新 live-R 全绿；至少一个 random/re 闭环；序列化后平滑预测一致；每种准入 smoother 有独立 capability，不因 parser 识别自动放行。

### Phase 4：预测、推断、诊断与模型选择（4–7 周）

**目标**：从“函数名已迁移”提升到结果语义可验证。

任务：

- `predict(type=response/link/terms)`、`predictAll`、SE/区间和固定构造参数；
- randomized quantile residual、hat/vcov/confint、worm/Q-Q/calibration；
- GAIC、stepGAIC、fitDist/chooseDist、profile/hyperparameter refit；
- artifact 恢复足够的残差/设计状态，或明确禁止无法可靠执行的 loaded-model 诊断；
- 对 staged 的旧 R 文件名模块进行归并、弃用或完成，减少重复入口。

**验收门禁**：训练后与工件加载后的受支持操作结果一致；模型选择真的执行 refit 而非仅汇总预计算结果；每项 API 有 R 行为对照或明确的 Python 差异说明。

### Phase 5：JAX 内核和 GPU 可用性（4–8 周，可与 Phase 3/4 后半并行）

**目标**：在统计结果冻结后证明 JAX 路径的正确性和收益。

任务：

1. 统一 NumPy RS 与 JAX RS 的算法合同和 convergence telemetry；
2. 扩展 JAX family 前先保证现有六族 CPU/JAX/R 三方 parity；
3. 减少 host-device sync、重复 tracing 和 Python outer loop；
4. 在 `debianjaxgpu` 中记录 Python/JAX/jaxlib/CUDA/cuDNN/GPU/driver/x64；
5. 分离 compile、cold、warm，做 n×p×family×method 扫描及显存峰值；
6. 只有在固定硬件上跨越稳定阈值后才配置 auto route；否则保持 CPU-first。

**验收门禁**：GPU 测试断言实际设备和 solver backend；GPU job 不再 `continue-on-error`（至少在发布候选流程中）；精度门禁先于性能门禁；生成可复现 crossover 表和退化回退策略。

### Phase 6：Python 生态扩展（4–8 周，独立轨道）

**目标**：在不污染 R parity 的情况下形成 Python-native 价值。

- 完善 sklearn：`sample_weight`、pipeline/CV、metadata、clone、统一 score；
- 基于 ondil 设计可选 online adapter：`partial_fit/update`、forget、chunk、缩放、正则路径；
- 将 deep/AutoML/服务层限制在 capability-safe family；
- statsmodels 风格 Results 仅在有明确用户需求时实施，不作为 1.0 阻断项。

**验收门禁**：sklearn estimator checks 通过；online batch 初值等价、chunk/顺序/漂移有独立测试；扩展 API 不被计入 R 迁移完成率。

### Phase 7：工程治理、文档和 1.0 发布（3–5 周）

**目标**：形成可重复构建、可审计、支持范围清晰的稳定版本。

任务：

- PR 运行分层完整 CPU suite；mypy 从非阻断逐步变为阻断；统一 Ruff/Black/Mypy 配置；
- docs extra 与 MkDocs 对齐，PR 严格构建文档并验证 examples；
- 使用受控 lock/constraints，固定关键 CI 工具和 Actions SHA；
- 校验 tag、pyproject、工件 schema、CHANGELOG 和 GitHub/PyPI 版本一致；
- wheel+sdist、Python 3.10–3.12、Windows/Linux/macOS 安装和最小 fit/predict；
- 发布支持矩阵、已知限制、迁移指南、安全和回滚说明。

**1.0 门禁**：目标 family 列表全部 production-safe；默认路由无 experimental；完整 CPU/R 门禁全绿；目标 GPU 能力有证据或明确排除；文档不含超出能力矩阵的声明。

## 7. 建议里程碑

| 里程碑 | 建议窗口 | 可交付结果 |
|---|---:|---|
| M0 可信基线 | 第 1–2 周 | P0 清零、能力快照、序列化/CI/devcontainer 修复 |
| M1 R 核心闭环 | 第 3–6 周 | NO 全合同；R-compatible RS/CG/mixed；offset/weights/formula 基线 |
| M2 高频族 Release Candidate | 第 7–14 周 | 2A/2B 中冻结的一组 family 端到端 validated |
| M3 Additive Release Candidate | 第 15–22 周 | pb/ps/cs parity，全链随机效应最小版本 |
| M4 Statistical API Complete | 第 20–26 周 | 预测/诊断/选择/工件合同闭环 |
| M5 Accelerator Preview | 第 23–30 周 | 可复现 GPU parity 和 crossover；不满足则明确延期 |
| M6 1.0 Scope Freeze | 第 28–34 周 | 支持矩阵冻结、全门禁、发布候选 |

日期仅用于资源规划；任一里程碑未满足门禁时不得靠版本号或日历晋级。

## 8. 前四周可直接执行的 backlog

| 顺序 | 工作项 | 依赖 | 交付证据 |
|---:|---|---|---|
| 1 | 修复默认 serialization 配对和版本策略 | 无 | 安全 round-trip + 旧工件兼容报告 |
| 2 | 核实并修复 `httpx2`、CI 失效 glob、devcontainer setup | 无 | clean install/build/CI smoke |
| 3 | 生成 146-family 静态能力快照和 placeholder 清单 | 1 | JSON/Markdown + CI drift check |
| 4 | production route 改为 strict production-safe | 3 | Python/HTTP/gRPC admission 契约 |
| 5 | 建立统一 DataContract（NA/weights/shape/support） | 无 | 所有方法参数化测试 |
| 6 | 实现逐参数 offset 和 design-info 保存 | 5 | R golden：mu/sigma offset + newdata |
| 7 | 实现 R-compatible mixed，重命名现有自动混合 | 5 | RS(n1)→CG(n2) 轨迹和结果对照 |
| 8 | 重跑 live-R 基线并修复 BI/GU/NBI/PO 已知 DPQR | 3、5 | 新报告，含环境/commit/seed |
| 9 | 修复 pb/ps/cs parity 或降级其公开声明 | 8 | 3/3 全绿或明确 experimental |
| 10 | 校准 README、migration、commercialization 文档 | 3、8、9 | 声明与 capability matrix 自动核对 |

## 9. 测试与证据策略

### 9.1 测试分层

- **L0 Contract**：import、registry、shape/dtype/support、schema、错误类型；每次 PR。
- **L1 Numerical**：DPQR、梯度/Hessian、RS/CG 单元、property；每次 PR。
- **L2 Integration**：fit→predict→serialize→load、smoother、服务；每次 PR 或分片必过。
- **L3 R parity**：固定 golden 每次 PR；live R 在核心 PR/夜间/发布候选 fail-closed。
- **L4 Platform**：Windows/Linux/macOS、Python 版本、CPU；发布候选。
- **L5 Accelerator/performance**：WSL/Linux GPU、cold/warm/memory；夜间和发布候选。

### 9.2 每个证据 artifact 必备字段

Git commit、工作树状态、Python/R/JAX/jaxlib/NumPy/SciPy 版本、R 包源码/hash、x64、设备、seed、family constructor 参数、formula、method、容差、通过/失败分类和生成命令。

### 9.3 完成定义

“已实现”必须同时满足：真实 kernel、无静默 fallback、R golden、目标方法拟合、newdata、序列化、失败合同、文档和 capability 更新。只有构造器、函数名、smoke 或某一项 R consistency 不足以晋级。

## 10. 风险登记

| 风险 | 等级 | 缓解措施 |
|---|---|---|
| 能力声明大于统计证据 | 极高 | capability 驱动文档和路由；production fail-closed |
| R 包/源码版本漂移 | 高 | 本地 `R/` hash、manifest、固定环境；live 与 golden 双轨 |
| 特殊函数/尾部/离散 q 数值不稳 | 高 | 高精参考、分区算法、property+R 边界测试 |
| 公式和预测 schema 分叉 | 高 | 单一 DesignState；训练/预测共用编译后的 term graph |
| 算法命名不等价 | 高 | pure/stabilized/hybrid 显式命名及遥测 |
| GPU 优化改变统计结果 | 高 | parity gate 前置，性能 gate 后置 |
| 旧迁移模块重复且行为不一致 | 中高 | 分批归并到核心 API，建立 deprecation map |
| 依赖和 CI 漂移 | 中高 | lock/constraints、固定版本、干净构建 |
| 范围无限扩张 | 中 | 1.0 冻结 family/smoother/API 列表，其他进入 post-1.0 |
| GPL 与商业边界 | 中 | 持续法律审查，技术计划不替代法律意见 |

## 11. 项目管理建议

1. 每两周更新 capability snapshot 和本文件“实测结果”，不要手工维护多个互相矛盾的完成百分比。
2. 每个 PR 只允许一个主统计能力晋级；代码、测试、R artifact、文档和 capability 必须同 PR 完成。
3. 维护三个独立看板：R parity、Python-native 扩展、工程/发布；避免 GPU、online 或服务工作掩盖核心迁移状态。
4. 缺陷优先级采用“错误成功 > 错误结果 > 不可用 > 性能慢”：静默 fallback 和过度准入优先于新增功能。
5. 1.0 支持范围应按用户高频场景冻结，而不是等待 146 个 registry 名全部 production-safe。

## 12. 推荐验证命令（Windows）

```powershell
# 基础环境
& ".\.venv\Scripts\python.exe" -c "import sys, jax, omnilss; print(sys.version); print(jax.__version__, jax.default_backend()); print(omnilss.__version__)"
Rscript --version

# 当前关键快速门禁
Set-Location omnilss
& "..\.venv\Scripts\python.exe" -m pytest tests/test_family_capabilities.py tests/test_serialization.py tests/test_prediction.py -q

# 完整 CPU 测试（可能耗时，应在 CI 或明确的验证窗口运行）
$env:JAX_PLATFORMS = "cpu"
$env:JAX_ENABLE_X64 = "true"
& "..\.venv\Scripts\python.exe" -m pytest tests -q

# Live-R 一致性由现有项目 runner 执行；报告必须记录当前 commit 和环境
Set-Location ..
& ".\.venv\Scripts\python.exe" benchmarks/run_local_validation.py --quick
```

WSL GPU 命令和虚拟环境入口应先在 `debianjaxgpu/` 中确认实际布局；本文件不假设该目录可以直接当作 Windows 可执行环境。

## 13. 最终判断

项目已经完成了大量迁移和工程基础建设，最有价值的资产是：真实的 NumPy RS 主路径、广泛的 family 代码、R 验证框架、能力治理、schema-safe 预测和服务/工件骨架。当前最大问题不是“功能太少”，而是“公开表面、历史文档和真实 validated 证据之间存在明显距离”。

建议立即执行 Phase 0，并把 **NO 的完整可信闭环复制到一组冻结的高频 family**；随后解决 `pb/ps/cs` parity 和随机效应，再进入 JAX/GPU 优化。按此顺序推进，OmniLSS 才能从功能广泛的 Beta 转变为支持范围明确、与 R 可复现对齐、可在 Python/JAX 生态稳定交付的 GAMLSS 实现。
