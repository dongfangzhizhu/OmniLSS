# OmniLSS 全量 GAMLSS 迁移：二次全面审计与执行计划

> 文档日期：2026-09-11  
> 审计对象：仓库当前 `main` 工作树；对照 `R/` 原始源码、`omnilss/`、`ondil/` 以及仓库内 `gamlss-python/`。  
> 目的：修正 2026-08-22 路线图中的过期事实，并给出可逐项验收的后续开发计划。  
> 本文是“全量迁移”专项执行文件；项目阶段总纲仍以 `DEVELOPMENT_PLAN.md` 为准。

## 1. 结论摘要

当前项目已具备可信的工程地基，但尚未达到“R gamlss 全量迁移”定义。当前可确认基线为：

| 维度 | 当前事实 | 结论 |
|---|---:|---|
| `omnilss` 版本 | 0.4.0 | 根 `.venv` 可导入，旧文档中的 0.3.0 已过期 |
| Python 注册家族 | 145 | 比 2026-08-22 文档一致，但不等于 145 个 R 家族完成 |
| implementation status | implemented 1 / experimental 84 / compat_alias 25 / placeholder 35 | 只有 `NO` 达到生产级实现口径 |
| R 包源码 | 12 个包目录，`gamlss` 79 个 R 文件、`gamlss.dist` 124 个 R 文件 | 原始范围仍为 1,115 个导出符号、约 1,700 个函数定义 |
| 测试收集 | 2,774 项，43 个收集错误 | 测试基座仍未闭环；不得宣称全库回归通过 |
| `gamlss-python` | 52 个 Python 源文件 | 已有较完整的 R 转录与 reference 流程，应作为公式/参数化/边界行为交叉 oracle |
| `ondil` | 65 个 Python 源文件，版本 0.4.8 | 在线增量层，不等价于离线 `gamlss` 迁移；需共享规范、独立验收 |

### 1.1 必须纠正的旧口径

1. “`BEo/GEOMo/PARETO2o` 三个家族未注册”不能继续作为当前事实。当前 registry 已存在 `GEOMO`；`BEo/PARETO2o` 是否仍缺失必须由 R NAMESPACE 与当前 registry 生成器重新计算，而非引用旧附录。
2. 145 个注册项包含大量内部扩展、兼容别名和骨架项，不能与 R 的 117 个分布家族直接相除得完成率。
3. `family_capabilities` 中的 `dqr/fitting/smoothing` 字段是能力元数据，不是 R 数值等价证据。必须增加逐家族、逐函数、逐场景的证据记录。
4. `gamlss-python` 的“与 R 约 1e-9”是其项目声明；必须通过本仓库固定 R artifact 复核后才能转化为 OmniLSS 结论。

## 2. 审计模型：四层证据，而不是单一状态

今后每个分布家族和每个核心 API 使用以下四层状态：

1. **Surface**：名称可导入、registry 可解析、参数名和 link 元数据存在。
2. **Kernel**：d/p/q/r（或对应 API）是真实实现，无 placeholder/静默近似，支持域和边界明确。
3. **Parity**：固定 R 源码、固定参数点、固定随机种子下通过数值对照；分别记录 d、p、q、r 和异常/边界案例。
4. **Admission**：在指定方法（RS/CG/MIXED/JOINT/LBFGS）、预测、新数据、服务路由中通过门禁，才可称 production-safe。

`experimental` 只能表示 Surface/Kernel 部分完成；`compat_alias` 永远不能当作 Parity；`placeholder` 不得进入生产路由。

## 3. 对照项目的可复用成果与边界

### 3.1 R 原始实现：唯一语义基准

每个家族开发必须先定位 `R/gamlss.dist/R/` 对应 `d/p/q/r`、link、参数检查和边界分支；拟合与平滑必须同时查阅 `R/gamlss/R/`。R 源码优先级高于任何 Python 转录。

### 3.2 `gamlss-python`：数值转录参考，不是依赖

可复用：参数化说明、解析导数、递推/特殊函数策略、RS/CG 控制参数、`pb/pbz` 以及 R reference 生成方式。  
不可直接复用：NumPy/SciPy 实现不能直接替代 JAX kernel；其 API、随机数、dtype、自动微分和服务路由均不满足 OmniLSS 要求。每次借鉴必须记录 R 文件、参考项目文件和最终 JAX 文件的对应关系。

### 3.3 `ondil`：在线层单独建模

`ondil` 的目标是在线/增量分布回归，依赖 NumPy/SciPy/Numba/Scikit-learn。后续应共享“分布族规范（参数、support、link、score、信息矩阵）”，但不得把离线 R parity 结果直接当作在线估计器正确性。在线层需要另外验证 chunk 顺序、warm start、漂移、缺失值和并发语义。

## 4. 当前差距矩阵

### 4.1 `gamlss.dist`

当前优先差距不是“注册数量”，而是：35 个 placeholder、25 个 compat_alias、逐函数证据大量缺失，以及部分 experimental 家族的 NaN/收敛问题。重点风险包括：

- 偏态 t/SEP/ST 家族的独立参数化与尾部积分；不得用 ST3C 克隆替代 ST1/2/4。
- Box-Cox `o` 变体的截断归一化、方差标准化和参数边界。
- 离散递推及 zero-inflated/zero-adjusted 家族：`DEL/DPO/SI/WARING/NBF/GPO/NBII/ZALG/ZIP/ZIP2/ZIPF`。
- `GB1/GB2/GIG/NO2/NOF/PARETO/PE2/PIG2/RGE/WEI2/exGAUS/EGB2/LOGSHASH/LQNO/SN/MN` 等 placeholder 连续族。
- `ZASICHEL/ZISICHEL` NaN、CG-ZAGA/CG-GU/MIXED-GU 收敛和非核心族拟合差异。

### 4.2 `gamlss` 核心

优先补齐 `gamlss.family`/link 系统、`predictAll`/`Formulae2data` 的语义、完整诊断与模型选择；平滑器必须按 admission 链路逐一准入。`pb/ps/cs` 的已有能力不代表其它 smoother 已完成。

### 4.3 周边包

`gamlss.tr`、`gamlss.cens` 与核心拟合耦合最紧；`mx/add` 次之；spatial/foreach/prepdata/ggplots/data 属于后续交付；`gamlss2` 是独立 API，不应与 1.x 的兼容目标混在同一批次。

## 5. 新的分阶段开发计划

### P0：审计与验证基座（1–2 周，阻塞所有宣称）

**P0.1 生成真实覆盖矩阵**：解析每个 R 包 NAMESPACE、R 函数定义、Python导出/registry、测试模块和 golden manifest，生成 `r-export-coverage.json` 与逐符号 Markdown。字段必须包含 R symbol、Python symbol、surface/kernel/parity/admission、证据路径、最近验证 commit。

**P0.2 修复测试收集**：定位并修复 43 个 `ondil/tests`/临时脚本收集错误；将临时诊断脚本移出 pytest 搜索路径；最终要求 `pytest --collect-only` 零错误，并分层统计 L1/L2/L3。

**P0.3 固定环境**：记录 Windows CPU、WSL GPU、R 版本、JAX/NumPy/SciPy 版本；统一 x64 策略。每个 golden manifest 必须包含 R/Python/JAX 版本和源码 commit。

**P0.4 自动差异报告**：每次 registry 或 R 源码变化自动报告 missing/extra/alias/placeholder，不再手工维护家族列表。

验收：覆盖 JSON 可重生成且与运行时一致；collect-only 零错误；报告可指出每个未完成项及证据路径。

### P1：分布 kernel 与 d/p/q/r parity（预计 8–12 周）

每个批次执行：R 源码阅读 → 数学说明 → JAX 实现 → support/boundary/property → R golden d/p/q/r → 状态晋升。建议顺序：

| 批次 | 范围 | 主要验收 |
|---|---|---|
| P1-A | ST/SEP/SST | 独立参数化、尾部、q 的稳健反演 |
| P1-B | BEINF/BEZI/BEOI/LG/LNO/LOGITNO/NET 及 Box-Cox `o` | 混合质量、截断归一化、边界 |
| P1-C | DEL/DPO/SI/WARING/YULE/NBF/GPO/NBII | 离散递推、归一化、整数 support |
| P1-D | ZALG/ZIP/ZIP2/ZIPF 及相关零膨胀 | 零质量、尾部概率、极端参数 |
| P1-E | GB/GIG/NO2/NOF/PARETO/PE2/PIG2/RGE/WEI2 | 正实数 support 与特殊函数 |
| P1-F | EXGAUS/EGB2/LOGSHASH/LQNO/SN/MN/剩余族 | 长尾、矩、数值稳定性 |

“完成”要求每族四个函数通过固定点、边界、单调性/归一化和随机抽样；没有 R 执行结果只能标 smoke-only。

### P2：全家族拟合 parity（预计 6–10 周）

先建立每族最小数据集（连续、计数、零膨胀、边界、极端值），再覆盖 RS、CG、MIXED；记录系数、fitted values、deviance、迭代次数、收敛原因、失败原因。修复 NaN-as-convergence、溢出伪成功、MIXED 路由和已知 ZAGA/GU 缺陷。每族至少一个 RS R golden；CG/MIXED 按族型分层覆盖。

### P3：核心 171 符号与 smoother（预计 10–16 周）

顺序：

1. `gamlss.family`、link、参数检查和 family 构造协议；
2. prediction/update/refit/Formulae2data；
3. diagnostics/residuals/hatvalues/vcov/logLik；
4. GAIC/stepwise/find.hyper/prof.dev；
5. `lo/random/re/ri/pcat/fp/bfp/cy/pvc/pbm/pbz/pbo/pbc/pbp/scs`。

每个 smoother 必须通过 parsing、basis、penalty、lambda/EDF、fit、newdata、R golden 七项 admission；未通过则明确拒绝，不得用近似趋势替代。

### P4：周边 R 包（预计 12–20 周）

- D1：`gamlss.tr`、`gamlss.cens`，与 P1/P2 家族和 censor/truncation 拟合打通；
- D2：`gamlss.mx`、`gamlss.add`，先完成有限混合/NP，再完成 GA/BA/NN/FK/TR；
- D3：`gamlss.spatial`、`gamlss.foreach`、`gamlss.prepdata`；
- D4：`gamlss.ggplots` 数据层、`gamlss.data`、WeatherGermany；
- D5：`gamlss2` 单独规格、单独版本目标，默认不阻塞 1.0。

### P5：ondil 在线层（与 P1/P2 并行，但独立验收）

建立共享 distribution contract；为在线 estimator 增加离线等价初始化、chunk 顺序不变性、warm-start、漂移和并发测试。L1 的 R golden 只验证 kernel；ondil 还必须有流式 golden 和性能/内存基线。

## 6. 里程碑与发布门槛

| 里程碑 | 必须满足 | 建议版本 |
|---|---|---|
| M0 | P0 完成，零收集错误，覆盖矩阵可重生成 | 0.4.x |
| M1 | R 117 家族全部 Surface+Kernel+Parity（d/p/q/r），无 placeholder 进入目标范围 | 0.5.0 |
| M2 | 家族拟合 parity 与收敛失败可解释，RS 全覆盖、CG/MIXED 分层覆盖 | 0.6.0 |
| M3 | `gamlss` 核心 API 与准入 smoother 闭环 | 0.9.0 |
| M4 | D1–D4 周边包完成，服务/文档/示例同步 | 1.0.0 |

任何里程碑都必须满足：`--require-r`、golden manifest、无 skip、覆盖报告、中文任务提交和可复现命令齐全。

## 7. 每个开发任务的标准模板

1. `R source map`：列出 R 文件、函数、参数化和边界分支；
2. `Python design`：JAX dtype、可微性、特殊函数和异常策略；
3. `implementation`：kernel/registry/API；
4. `tests`：unit/property、R comparison、fit/newdata（如适用）；
5. `evidence`：golden JSON、manifest、环境和 commit；
6. `status update`：只在四层状态满足条件后修改能力矩阵；
7. `regression`：运行目标 suite、相关全量 suite 和 collect-only。

## 8. 立即执行的前十项任务

- [ ] P0.1 重新生成 R↔Python 逐符号/逐家族覆盖矩阵；
- [ ] P0.2 清零 43 个测试收集错误并拆分 L1/L2/L3 测试；
- [ ] P0.3 冻结三套 Python/R 环境并更新基线；
- [ ] P0.4 自动报告 missing/extra/alias/placeholder；
- [ ] P0.5 核查 `BEo/PARETO2o` 及所有 `o` 变体的真实 registry 差异；
- [ ] P1.1 为每族建立参数化和 support 规范；
- [ ] P1.2 完成 ST/SEP R golden 与独立 kernel；
- [ ] P1.3 完成 BEINF/zero-adjusted/Box-Cox-o 批次；
- [ ] P1.4 完成离散递推批次；
- [ ] P2.1 建立全族最小拟合数据集和收敛诊断格式。

## 9. 风险与决策

- **范围风险**：117 家族、1,115 导出符号和周边包不能按一个完成率管理；采用上述四层证据和分阶段发布。
- **数值风险**：JAX 版本差异、float32、特殊函数和 R 优化器差异；golden 必须记录环境，失败必须分类为公式、参数化、优化器或数值稳定性。
- **参考项目风险**：`gamlss-python` 只能交叉参考，不得复制其未验证声明；`ondil` 不得替代 L1 parity。
- **工程风险**：pytest 收集错误会掩盖真实回归；P0.2 是硬阻塞。
- **范围决策**：`gamlss2` 放在 1.0 后独立规划；ggplot 外观不追求像素一致，只要求数据层一致；服务生产路由继续 fail-closed。
