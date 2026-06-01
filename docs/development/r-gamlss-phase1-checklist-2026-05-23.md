# OmniLSS Phase-1 (`gamlss` / `gamlss.dist`) 全量对齐 Checklist（2026-05-23）

基线文档：`docs/development/r-gamlss-coverage-and-plan-2026-05-22.md`。  
本清单用于把“第一期目标（覆盖 R `gamlss` + `gamlss.dist` 功能）”拆解成可打勾项，按 **先开发后验证** 的顺序推进。

## 0. 统计口径

- ✅ 已完成：仓库中存在可调用实现，且 API/行为路径已经接入主流程。
- 🟡 部分完成：存在实现但缺 R parity 证据、或功能仅别名/兼容层。
- ⬜ 未完成：缺实现或仅有占位。

---

## A. 基础设施与可追溯性（Phase A）

- ✅ A1 CI 快速 R parity gate（devcontainer 内 `run_local_validation.py --quick`）。
- ✅ A2 R coverage matrix 生成脚本与 CI 漂移检查。
- ✅ A3 JSU 重复工厂收敛（`distributions_b11` legacy alias 已移除）。
- ✅ A4 关键迁移模块补充 `R reference:` 注释（含 smoothers）。
- ✅ A5 capability matrix v4 实现状态层：`FamilyCapability` 现已公开
  `implementation_status`，并能从 registry/compat factory 位置标记
  `implemented` / `experimental` / `compat_alias` / `placeholder` / `missing`
  / `unsupported_with_rationale`；`ST1-4`、`DBI`、`DBURR12`、Box-Cox untr、
  `MULTIN` 与 `gen.Family` skeleton compatibility shells 不再只靠
  feature evidence 混入正常覆盖统计；同时已通过静态源码扫描识别
  `build_ad_family(...)` 缺失显式 `p/q/r` 的 AD fallback 占位路径（如
  `GB2`、`GIG`、`SEP1`），避免触发昂贵的 family 实例化。

---

## B. 分布覆盖（Phase B）

### B1. 已注册 family 的 R-backed d/p/q 证据补齐

- 🟡 已推进：新增 `GIG` 的 R consistency（d/p/q）测试文件，并已纳入 consistency suites 的 basic/all/phase0 gate 组。
- 🟡 持续推进：新增 `YULE`、`ZINBF` 的 R consistency（d/p/q）测试，并已纳入 consistency suites 的 basic/all（含 phase0 gate）组，且已并入项目 `all` 与 `quick` 套件；其余 `experimental` family 的 d/p/q parity 仍待批量补齐（约 60+）。
- ✅ **2026-06-02 更新**：已完成 42 个缺失 family 的完整 R consistency 测试生成：
  - 零调整/膨胀变体：ZAPIG, ZIPIG, ZASICHEL, ZISICHEL, ZAZIPF, ZABB, ZIBB, ZABNB, ZIBNB（9个）
  - 连续正值分布：GAF, NOF, RGE, WEI2, WEI3, GB1, PARETO1, PARETO1O, PIG2, PE2（10个）
  - 偏斜/形状分布：EGB2, LQNO, SST, ST5, SEP, SEP1, SEP2, SEP3, SEP4, LOGSHASH, LOGSHASHO, SHASHO, SHASHO2, JSUO（14个）
  - 计数分布：LG, ZIPF, ZALG, GPO, DELAPORT, DPO1, GP, GENERALISEDPOISSON（8个）
  - 有界分布：LOGITNO（1个）
  - 测试基于参数感知框架，支持 1-4 参数 family，使用正确的参数范围配置。
  - 创建了 `family_param_ranges.py` 配置文件，定义每个 family 的参数约束。
  - 创建了 `generate_all_r_tests.py` 统一测试生成器。

### B2. 计划中“缺失分布/接口”

> 进度说明（2026-05-23）：以上分布缺口已全部具备“兼容层可调用入口 + registry 可解析”，
> 当前剩余重点转向“专用数值内核替换 + R 一致性证据从脚手架升级为稳定覆盖”。

- ✅ `MULTIN` 构造接口（`MULTIN(3|4|5)`），并补齐 `distribution_registry.resolve("MULTIN")` 默认解析路径；新增独立 `MULTINFamily` compatibility shell（保留当前基于 `MN3/4/5` 数值骨架）。
- ✅ Batch-16 家族 registry 入口补齐（`DPO1`/`PARETO1`/`PARETO1o`/`SEP`/`BCCGo`/`LOGSHASHo` 已可通过 `distribution_registry.resolve()` 解析）。
- ✅ Batch-9/10/13/15 多家族 registry 入口补齐（`LG`/`ZIPF`/`ZALG`/`IGA`/`PIG2`/`PE2`/`LOGITNO`/`DELAPORT`/`GP`/`GeneralisedPoisson`/`ZASICHEL`/`ZISICHEL`/`ZAPIG`/`ZIPIG`/`ZABB`/`ZIBB`/`ZAZIPF`/`ZABNB`/`ZIBNB`/`ZINBF` 已可通过 `distribution_registry.resolve()` 解析）。
- ✅ `GP` alias 兼容语义修正（新增独立 `GPFamily` compatibility shell，避免 `resolve("GP")` 名称/类型回落到 `GENERALISEDPOISSON`）。
- ✅ `GENERALISEDPOISSON` alias 兼容语义修正（新增独立 `GENERALISEDPOISSONFamily` compatibility shell，避免 registry 解析后名称/类型回落到 `GeneralisedPoisson` 原始构造器）。
- ✅ `*O` 后缀 alias 兼容语义修正（`SHASHO`/`JSUO`/`SHASHO2`/`BCPEO`/`BCTO`/`PARETO1O`/`BCCGO`/`LOGSHASHO` 新增独立 compatibility shell，避免名称/类型回落到原始小写-o构造器）。
- ✅ Batch-11/12/15 与零调整族 registry 入口继续补齐（`GIG`/`GB1`/`WEI2`/`WEI3`/`NOF`/`GAF`/`RGE`/`SEP1-4`/`SST`/`ST3C`/`EGB2`/`LQNO`/`NBF`/`LOGSHASH`/`GPO`/`ZABI`/`ZANBI` 已可通过 `distribution_registry.resolve()` 解析）。
- ✅ `SEP2/SEP3/SEP4` 兼容构造器语义修正（保持共用 SEP 数值骨架，但分别使用独立 family class + 对应名称返回 `SEP2`/`SEP3`/`SEP4`，避免 registry 解析后名称/类型回落为 `SEP1`）。
- 🟡 `TF2` 接口：已提供构造器与注册；新增独立 `TF2Family` compatibility class（与 `TF` 解耦 family 类型语义）并保留当前共用数值内核；`nu` 链接已切换为 `logshiftto2`（显式约束 `nu > 2`，与 TF2 语义对齐），并新增构造器/链接单测；同时在 TF2 family d/p/q/r 与 score/hessian 路径加入 `nu>2` 钳制防护（即使直接调用 family 接口也保持 TF2 语义）；d/p/q R 一致性已扩展到 `central/tail/ultra_tail_prob` 三组参数剖面（含极端概率区间 `1e-4 ~ 1-1e-4`），并纳入 consistency suites（basic/all/phase0 gate）；已改为直接对 R `TF2` 执行 d/p/q 对齐（不再借道 R `TF`）；后续仍需继续扩展 TF2 专用参数化的边界验收。
- 🟡 ST1（已提供兼容构造器与 registry，当前共用 ST3C 数值内核；已将 d/p/q R 一致性用例扩展到 `central/tail/ultra_tail` 三组参数场景与极端概率区间，并纳入 consistency suites 的 basic/all 组）
- 🟡 ST2（已提供兼容构造器与 registry，当前共用 ST3C 数值内核；已将 d/p/q R 一致性用例扩展到 `central/tail/ultra_tail` 三组参数场景与极端概率区间，并纳入 consistency suites 的 basic/all 组）
- 🟡 ST3（已提供兼容构造器与 registry，当前共用 ST3C 数值内核；已将 d/p/q R 一致性用例扩展到 `central/tail/ultra_tail` 三组参数场景与极端概率区间，并纳入 consistency suites 的 basic/all 组）
- 🟡 ST4（已提供兼容构造器与 registry，当前共用 ST3C 数值内核；已将 d/p/q R 一致性用例扩展到 `central/tail/ultra_tail` 三组参数场景与极端概率区间，并纳入 consistency suites 的 basic/all 组）
- ✅ `ST1/ST2/ST3/ST4` compatibility class 语义修正（在共用 ST3C 数值骨架前提下，分别返回独立 `ST1Family`/`ST2Family`/`ST3Family`/`ST4Family`，避免类型语义回落到通用 `FamilyDefinition`）。
- 🟡 DBI（已提供兼容构造器与 registry，当前共用 BI 数值内核；已将 d/p/q R 一致性用例扩展到 `mid_range/edge_heavy/ultra_tail` 三组参数与极端概率区间，并纳入 consistency suites 的 basic/all 组）
- 🟡 DBURR12（已提供兼容构造器与 registry，当前共用 SICHEL 数值内核；已将 d/p/q R 一致性用例扩展到 `central/tail_edge/ultra_tail_prob` 场景并覆盖极端概率区间，并纳入 consistency suites 的 basic/all 组）
- 🟡 BCCGuntr（已提供兼容构造器与 registry，当前共用 BCCG 数值内核；已将 d/p/q R 一致性用例扩展到 `central/tail/ultra_tail` 三组参数场景并覆盖极端概率区间，并纳入 consistency suites 的 basic/all/phase0 gate 组）
- 🟡 BCPEuntr（已提供兼容构造器与 registry，当前共用 BCPE 数值内核；已将 d/p/q R 一致性用例扩展到 `central/tail/ultra_tail` 三组参数场景并覆盖极端概率区间，并纳入 consistency suites 的 basic/all/phase0 gate 组）
- 🟡 BCTuntr（已提供兼容构造器与 registry，当前共用 BCT 数值内核；已将 d/p/q R 一致性用例扩展到 `central/tail/ultra_tail` 三组参数场景并覆盖极端概率区间，并纳入 consistency suites 的 basic/all/phase0 gate 组）
- ✅ `DBI`/`DBURR12`/`BCCGuntr`/`BCPEuntr`/`BCTuntr` compatibility class 语义修正（保持当前共用数值骨架，同时分别返回独立 compatibility family class，避免类型语义回落到通用 `FamilyDefinition`）。
- ✅ `ZIBI`/`ZINBI` alias 兼容语义修正（新增独立 `ZIBIFamily`/`ZINBIFamily` compatibility shell，避免回落到原始 zero-variant 构造器类型语义）。
- ✅ `ZAGA`/`ZAIG`/`ZAP` alias 兼容语义修正（新增独立 `ZAGAFamily`/`ZAIGFamily`/`ZAPFamily` compatibility shell，避免回落到原始 mixed/zero-altered 构造器类型语义）。
- ✅ `ZABI`/`ZANBI` alias 兼容语义修正（新增独立 `ZABIFamily`/`ZANBIFamily` compatibility shell，避免回落到原始 zero-adjusted 构造器类型语义）。
- ✅ `ZABB`/`ZIBB`/`ZAZIPF`/`ZABNB`/`ZIBNB` alias 兼容语义修正（新增独立 compatibility shell，避免回落到原始 zero-variant 构造器类型语义）。
- ✅ `ZASICHEL`/`ZISICHEL`/`ZAPIG`/`ZIPIG` alias 兼容语义修正（新增独立 compatibility shell，避免回落到原始 batch-10 zero-variant 构造器类型语义）。
- ✅ `ZINBF` alias 兼容语义修正（新增独立 `ZINBFFamily` compatibility shell，避免回落到原始 `ZINBFFamily` 构造器类型语义）。
- 🟡 `centileSK`：已有实现与 R 一致性测试；已将经验分位对齐扩展到 `linear/right_skew/mixed_sign` 三组样本剖面，并补充 `theoCentileSK` 极端概率（0.1%~99.9%）对齐；已纳入 consistency suites 的 basic/all/phase0 gate 组，后续继续补充更多边界语义验收。
- ✅ `flexDist`, `momentSK`, `hazardFun`, `gen.hazard`：已具备实现与基础单测覆盖。

### B3. `gen.Family` skeleton builder

- 🟡 已新增统一兼容 builder 入口（`build_gen_family_skeleton`）并将 19 个常见 skeleton 名称注册到 registry；现已补齐 `GenSkeletonCompatFamily` compatibility shell 以稳定 skeleton 类型语义，并为代表性 count/continuous skeleton 新增 d/p/q R 一致性测试脚手架；仍待逐个替换为专用数值实现。

### B4. `distributions3` 风格接口

- 🟡 已完成第一版 helper：`pdf/log_pdf/cdf/quantile/random/is_continuous/is_discrete/support`。
- 🟡 已补：`mean/variance/skewness/kurtosis` 支持统一参数与 sample 回退策略；并新增 LOGNO/NO2/GA/EXP/PO/BI 的解析矩实现与单测（含 GA 偏度/峰度解析式），仍待扩展更多 family 的解析解与精度基准。

---

## C. 拟合与诊断缺口（Phase C）

- 🟡 已有模块雏形/数据接口：`Formulae2data`, `getSmo/edfAll`, `dtop`, `par_plot`, `prof_dev`, `find_hyper`, `ECDF`, `loglogplot`, `gen_likelihood`, `lms`, `quantSheets` 等；其中 `gen_likelihood` 已补权重加权总量（weighted totals）与权重形状校验，`dtop` 已补权重加权偏差贡献排序与权重形状校验，`par_plot` 已补 response residual / weighted residual 诊断载荷与权重形状校验，`find_hyper` 已补有限值校验与候选排序索引（rank order / best index）输出，`ECDF/loglogplot` 已补权重路径与权重语义校验，`quantSheets` 已补加权分位路径与概率序约束校验，`findPower/z_scoresQS` 已补加权路径与权重语义校验，`lms` 已补权重输入与 weighted z 诊断载荷，`prof_dev` 已补加权偏差曲线/排序输出与有限值校验，`getSmo/edfAll` 已补 finite 校验与 `edf_summary` 汇总载荷（含 total/share），`Formulae2data` 已补权重列读取与权重语义校验。
- ⬜ 待完成：上述函数的 R 对齐验收（输入语义、列名、边界行为、数值容差）。
- ⬜ 待完成：`gamlssML` 相关 `predict/summary/vcov` 的 R parity 完整验证。

---

## D. 平滑器覆盖（Phase D）

- ✅ 已实现主干：`pb`, `ps`, `cs` + `te/ti`, `tps` 扩展。
- 🟡 待实现：平滑器项已全部具备轻量数据接口（含 `pbm/pbz/pbo/pbc`）；后续重点为参数级行为对齐、R parity 与可视化输出一致性。
- 🟡 parser 层已识别 `pb/ps/cs/s/lo/random/re/ri/pcat/fp/bfp/cy/pvc/pbm/pbz/pbo/pbc` 与部分 tensor smooth term（含 `s(..., smoother='random'|'re'|'ri'|'lo'|'pcat'|'fp'|'bfp'|'cy'|'pvc'|'pbm'|'pbz'|'pbo'|'pbc')`）；并已在 smooth metadata 增加 `staged_compat` 标识（staged=true、`pb/ps/cs` 为 false）以避免误判为完整实现。当前仍需“完整参数级解析 + R parity”补齐。

---

## E. 服务层与持久化（Phase E）

- 🟡 REST / gRPC 对新分布与诊断入口的系统化暴露仍待补齐；gRPC
  `Fit()` 已修正失败响应类型（返回 `FitResponse` 而不是
  `BatchFitResponse`），并在进入 `gamlss()` 前接入 method-route
  preflight，默认阻断 `compat_alias` / `placeholder` family。
- ✅ gRPC 本地模型仓默认路径已从硬编码 `/tmp/omnilss-grpc-models`
  改为跨平台 `tempfile.gettempdir()/omnilss-grpc-models`，避免 Windows
  环境导入服务模块时因 `\tmp` 权限失败。
- ✅ SQLite 模型仓 capability snapshot schema 升级与回放验证已完成：`models` 表现在迁移/创建 `schema_version`、`family`、`capability_matrix_version`、`implementation_status`、`capability_json`、`capability_report_json`、`capability_ok` 字段；保存、加载与进程重启恢复索引时会从模型 artifact 回放 capability snapshot 并持久化当前 runtime 对齐报告。

---

## F. 发布门禁与文档（Phase F）

- 🟡 CI 已具备部分门禁（quick parity + coverage matrix drift + phase0 gate 扩展覆盖 ST1-4 / Box-Cox untr / centileSK / GIG 一致性模块；`tests/run_suite.py` 已增强为可容忍可选依赖触发的 skip-like 模块加载异常，并补齐脚本直跑场景（无预设 PYTHONPATH）下的 suites 导入回退，避免门禁脚本因环境差异/单模块缺依赖直接中断）。
- 🟡 capability matrix v4 已生成到 `docs/development/family-capability-matrix-2026-05-18.json`，
  并新增运行时/生成工具测试；method route preflight 现在默认拒绝
  `compat_alias` 与 `placeholder` family，除非调用方显式传入
  `allow_compat_alias=True` / `allow_placeholder=True` 用于迁移或
  smoke-test 场景；本轮已同步刷新 matrix artifact，以纳入
  `dpqr_functions.py` placeholder/近似静态识别结果。
- ⬜ “所有暴露 family 均有 R-backed d/p/q 或明确 experimental 标注”尚未达成。
- ⬜ README / 中英文文档 / capability 页面全量对齐待完成。

---

## 下一批开发优先级（从高到低）

1. **兼容层 family 的 R consistency 证据深化**：已将 ST1-4、DBI、DBURR12、BCCGuntr、BCPEuntr、BCTuntr 的脚手架扩展到多参数区间与边界场景（central/tail、edge-heavy μ、`ultra_tail` 概率区间）；其余兼容层 family 持续补齐中。  
2. **将兼容层逐步替换为专用数值实现**：优先 ST1/ST2/ST3/ST4、DBI、DBURR12、Box-Cox untr 系列；替换完成前必须保持 `implementation_status=compat_alias`，不得计入实现覆盖。  
3. ✅ placeholder 自动检测范围继续扩展：已覆盖 AD `build_ad_family` fallback，且已补 `dpqr_functions.py` 中 NaN / not-implemented / 正态近似 / generic fallback random draw 的静态识别；同时修正“先用 `build_ad_family` 再显式接入专用 d/p/q/r”的 family（如 `BE`、`WEI`、`IG`）不再被误判为 placeholder。
4. `distributions3` 统计矩接口从“部分 family 支持”升级为“可配置回退 + 全家族可用性声明”（已新增 `moment_availability` 声明接口与单测，后续继续扩展解析矩覆盖）。  
5. 为当前已注册但无证据的 family 批量补 d/p/q R consistency（按家族分组推进）。

---

## 人工检查点（本轮）

- 请优先确认该清单状态标注是否符合你对“一期目标”的验收口径。  
- 若认可该口径，下一轮将直接按“下一批开发优先级”第 1 项开始实现。


---

## 2026-06-02 开发进展更新（第二轮）

### 已完成工作

1. **验证 benchmarks 测试框架**
   - ✅ 运行 `comprehensive_r_consistency_test.py --quick` 测试
   - ✅ Python-only 模式：42/42 测试通过 (100%)
   - 🟡 R 对比模式：14/42 测试通过 (33.3%)
     - Fitting 测试：11/12 通过 (92%)
     - DPQR 测试：3/27 通过 (11%) - 主要是容差问题
     - Smoothing 测试：0/3 通过 (0%) - 需要调查

2. **测试失败分析**
   - DPQR 测试失败主要是数值精度问题：
     - Max absolute error: 4.95e-05 到 4.90e-04
     - 默认容差：abs=1.0e-05, rel=1.0e-05（可能过严）
   - Smoothing 测试失败误差较大：
     - pb: 6.87e+00
     - ps: 1.49e+02
     - cs: 3.63e+02
   - 需要调查 smoothing 实现差异

3. **测试基础设施状态**
   - ✅ benchmarks 目录已有完善的测试框架
   - ✅ 自动生成 JSON 结果和 Markdown 报告
   - ✅ 支持多种测试模式（quick, no-r, large 等）
   - ✅ 包含性能测试（comprehensive_performance_test.py）

4. **Smoother 问题诊断（2026-06-02 下午）**
   - ✅ 创建了 `diagnose_smoother_differences.py` 诊断工具
   - ✅ 运行诊断发现了根本问题：
     - **pb smoother**: Python 使用 35 个系数 vs R 使用 2 个系数（EDF=12）
       - 误差：max_abs=5.66e-02, RMSE=2.22e-02, correlation=0.999
       - 问题：默认参数不匹配，Python 可能使用了过多的基函数
     - **ps smoother**: Python 使用 24 个系数 vs R 使用 2 个系数（EDF=5）
       - 误差：max_abs=3.99e-01, RMSE=2.26e-01, correlation=0.964
       - 问题：默认参数不匹配，拟合形状差异较大
     - **cs smoother**: Python 返回常数值（完全失败）
       - 所有拟合值都是 0.1807（数据均值）
       - 误差：max_abs=8.01e-01, RMSE=4.44e-01
       - 问题：cubic spline 实现有严重 bug
   - ⚠️ **JAX 64位精度未启用**：所有测试都在 float32 模式下运行
     - 这可能导致数值精度问题
     - 需要确保测试环境正确设置 `JAX_ENABLE_X64=true`

### 发现的问题

1. **DPQR 数值精度**
   - 大部分 family 的 d/p/q 函数与 R 有小的数值差异
   - 误差在 1e-5 到 1e-4 量级，可能是浮点精度或算法实现差异
   - 需要评估是否需要调整容差或改进实现

2. **Smoothing 实现差异（已诊断）**
   - ❌ **cs smoother 完全失败**：返回常数值，需要紧急修复
   - ⚠️ **pb/ps smoother 参数不匹配**：
     - Python 使用的基函数数量远多于 R
     - 默认参数（n_knots, ps_intervals 等）与 R 不一致
     - 需要对齐默认参数或在 gamlss() 调用时传递正确参数
   - ⚠️ **JAX 64位模式未启用**：可能影响数值精度

3. **测试覆盖**
   - 当前 benchmarks 测试覆盖了主要的 family
   - 但还有 67 个 family 缺少专门的 R consistency 测试
   - 需要继续补充测试覆盖

### 下一步计划

1. **紧急修复（最高优先级）**
   - 🔴 修复 cs smoother 返回常数值的 bug
   - 🔴 确保测试环境启用 JAX 64位模式（`JAX_ENABLE_X64=true`）
   - 🟡 对齐 pb/ps smoother 的默认参数与 R gamlss

2. **短期（本次会话）**
   - 调查 cs smoother 为什么返回常数值
   - 对比 R gamlss 的 pb/ps 默认参数设置
   - 修复发现的问题并重新运行测试
   - 评估 DPQR 容差设置是否合理

3. **中期（后续开发）**
   - 验证 smoother 修复后的测试结果
   - 为剩余 67 个 family 添加测试
   - 改进数值精度（如果需要）

4. **长期（Phase 1 完成目标）**
   - 所有 family 都有 R consistency 测试
   - 所有测试通过或有明确的差异说明
   - 完成文档更新
