# OmniLSS 六个月执行计划（2026-05-17 至 2026-11-17）

[English version](six-month-execution-plan-2026-05-17.md)

> 计划依据：本计划基于 OmniLSS 的实现级代码审计制定，只把运行时代码、测试、打包元数据和服务边界作为事实来源。README 宣传、docstring 和营销表述不作为交付证据。
>
> 默认工作语言：英文。本文提供同步中文版本，便于双语执行与利益相关方沟通。

---

## 1. 执行目标

未来六个月，OmniLSS 应从一个覆盖面较广的研究型 Python/JAX GAMLSS 实现，升级为可信的 open-core 分布式统计建模平台。

计划包含三条同步主线：

1. **研发主线**：加固统计核心、预测 artifact、验证矩阵和服务运行时。
2. **市场与推广主线**：通过透明 benchmark、技术内容、双语教育和垂直 demo 建立信任。
3. **商业主线**：验证商业切入点，定义 Pro/Enterprise 边界，并准备首批付费 pilot。

### 六个月目标状态

到 **2026-11-17**，OmniLSS 应具备：

- 稳定的 Core release candidate，支持 schema-safe 的 fit/predict/serialize/load roundtrip；
- 面向核心分布族和拟合路径的验证证据包；
- 安全服务 MVP，包含异步 job、模型注册表、鉴权和使用日志；
- 一个具备可衡量业务价值的垂直 demo，优先保险风险、医疗生长/风险曲线或金融尾部风险建模；
- Pro/Enterprise 产品定义，包括 pilot 定价、支持边界和部署 playbook。

---

## 2. 运营原则

1. **实现优先于宣传。** 只有经过代码、测试和可重复 artifact 验证的能力才能对外承诺。
2. **禁止静默统计降级。** 不支持的公式、平滑项、分布能力或 artifact schema 必须返回结构化错误。
3. **Core 建立可信度，Pro 创造商业价值。** 开源 Core 最大化信任；Pro/Enterprise 销售治理、自动化、运维和垂直工作流。
4. **学术可信度服务商业转化。** benchmark、R 一致性报告和校准指标不是副任务，而是市场教育和企业风控资产。
5. **文档双语默认。** 任何战略性公开计划、发布说明、教程或 benchmark 摘要都应提供英文和中文版本，并互相链接。

---

## 3. 六个月路线图总览

| 月份 | 主题 | 研发里程碑 | 市场里程碑 | 商业里程碑 |
|---|---|---|---|---|
| 第 1 月 | 可信 Core | Artifact schema v2、安全预测、公式安全、测试门禁 | 发布“可信分布建模”审计摘要 | 选择首个垂直方向和 pilot 用户画像 |
| 第 2 月 | 统计验证 | Family capability registry、R/Python 一致性报告、AD/Hessian 验证 | 发布 benchmark 与验证看板 | 定义 Pro/Enterprise 包装边界 |
| 第 3 月 | 生产服务 MVP | 安全 API、异步拟合 job、持久模型注册表、资源限制 | 发布双语教程和 demo notebook | 开始 design partner 外联 |
| 第 4 月 | 垂直产品化 | 垂直工作流模板和校准报告 | 发布行业 demo 与 case study 内容 | 运行 2-3 个 design partner pilot |
| 第 5 月 | 规模与可靠性 | 性能 backend 改进、可观测性、部署自动化 | 发布性能报告与对比系列 | 将 pilot 转化为付费 beta 报价 |
| 第 6 月 | 商业就绪 | Release candidate、文档冻结、支持 playbook | 公开发布 campaign 与 webinar | 拿下首个付费 pilot 或企业 LOI |

### 日历窗口

| 窗口 | 日期 | 主要 checkpoint |
|---|---|---|
| 第 1 月 | 2026-05-17 至 2026-06-16 | Core 可信底座 checkpoint |
| 第 2 月 | 2026-06-17 至 2026-07-16 | 验证可信度 checkpoint |
| 第 3 月 | 2026-07-17 至 2026-08-16 | 内部服务 MVP checkpoint |
| 第 4 月 | 2026-08-17 至 2026-09-16 | 垂直 pilot-ready checkpoint |
| 第 5 月 | 2026-09-17 至 2026-10-16 | 付费 beta readiness checkpoint |
| 第 6 月 | 2026-10-17 至 2026-11-17 | Release candidate 与商业就绪 checkpoint |

---

## 4. 研发计划

### 4.1 第 1 月：可信 Core 底座

**目标：** 让 fit → predict → serialize → load → predict 安全且可重复。

#### 工作流 D1：模型 Artifact 与设计矩阵 Schema v2

**任务**

- 定义版本化 `DesignMatrixSchema`，包含：
  - 参数名；
  - 原始公式；
  - 解析后的 term 顺序；
  - 截距状态；
  - 列名；
  - 系数数量；
  - factor levels；
  - 数值变换 AST metadata；
  - smooth basis metadata；
  - family capability metadata。
- 将 schema 持久化进模型 artifact。
- load 时恢复 schema，不再伪造不完整的设计矩阵。
- prediction 依赖已保存 schema，而不是重新解释松散公式。
- 对缺变量、未见 factor level、缺 smooth metadata、列数不匹配给出硬错误。

**验收标准**

- 线性 roundtrip prediction 与原模型在 `rtol <= 1e-10` 内一致。
- smooth roundtrip prediction 在 `rtol <= 1e-7` 内一致，或返回明确的不支持特性错误。
- 序列化 artifact 默认不再保存完整训练数据。
- 所有 prediction schema 失败都使用结构化异常类型。

#### 工作流 D2：公式安全与解析器加固

**任务**

- 确保所有数值公式表达式都通过 AST 白名单 evaluator。
- 拒绝属性访问、import、comprehension、lambda、索引、不安全调用和过深表达式。
- 用稳健 parser 或受限 grammar 替换脆弱的 smooth/tensor 参数字符串 split。
- 增加恶意公式测试。

**验收标准**

- 公式注入测试通过。
- 公式解析错误包含 term 名称和拒绝原因。
- 生产预测路径不执行任意 Python 代码。

#### 工作流 D3：RS/CG/JAX 能力矩阵

**任务**

- 为每个 family 和 method route 建立能力表：
  - 可拟合；
  - 可预测；
  - 可采样；
  - 支持 smooth；
  - 已通过 R 一致性验证；
  - 已通过 AD/Hessian 验证；
  - 生产安全；
  - 实验性。
- 运行时使用该矩阵阻止不支持的 method/family 组合。
- 明确记录 fallback 行为。

**验收标准**

- 不支持路径返回结构化错误，不再静默 fallback。
- 公开文档与运行时 capability 数据一致。

### 4.2 第 2 月：统计验证与研究证据

**目标：** 把正确性转化为可复用的科学证据包。

#### 工作流 D4：Family 验证矩阵

**任务**

- 优先 10 个核心 family：`NO`、`GA`、`PO`、`BI`、`NBI`、`BE`、`WEI`、`TF`、`LOGNO`、`ZAGA`。
- 对每个核心 family 验证：
  - density/PMF 数值；
  - CDF 单调性；
  - quantile/CDF 互逆一致性；
  - 随机样本矩（适用时）；
  - score vs finite difference；
  - Hessian vs finite difference 或 AD；
  - 边界条件和非法参数处理。

**验收标准**

- 每个核心 family 都有机器可读 validation JSON 和人类可读报告。
- 失败被分类为实现 bug、数值容差问题、不支持定义域或 reference mismatch。

#### 工作流 D5：R/Python 一致性 Harness

**任务**

- 将 R bridge 稳定为可选但可重复的验证环境。
- 增加 snapshot dataset，保证比较可重复。
- 报告 coefficients、fitted values、deviance、AIC/BIC、残差摘要和收敛标志。
- 把依赖 R 的测试与默认 CI 分离，但保留定期验证 job。

**验收标准**

- 每个优先 family 都有可重复一致性报告。
- CI 清晰区分“unit pass”“R unavailable skip”和“R consistency failure”。

#### 工作流 D6：学术指标

**任务**

- 增加 PIT、CRPS、quantile loss、interval coverage、calibration slope 和残差诊断。
- 为 synthetic 和 real-like 数据集建立可重复 benchmark runner。
- 起草论文大纲，主题聚焦 Python/JAX GAMLSS 验证、能力边界和可微拟合路线。

**验收标准**

- 至少三个公开 benchmark 场景可由一个命令复现。
- 验证输出可直接进入技术论文或 whitepaper。

### 4.3 第 3 月：生产服务 MVP

**目标：** 将服务原型升级为安全的内部平台 MVP。

#### 工作流 D7：安全 API 与 Job Runtime

**任务**

- 将同步长时间 fit 调用替换为异步 job。
- 增加持久化 model registry。
- 增加鉴权、request ID、audit log、timeout、payload limit 和 rate limit。
- 为 fit、predict、sample、artifact 操作增加结构化错误码。
- 为 gRPC 和 HTTP 增加 TLS/mTLS 部署选项。

**验收标准**

- 模型在服务重启后仍可从新进程预测。
- 超大或非法 payload 能安全失败。
- 每个请求都有可追踪的 job/model/user metadata。

#### 工作流 D8：部署与可观测性

**任务**

- 提供 Core API、registry、对象存储和 metrics 的 Docker Compose 部署。
- 增加 Prometheus-compatible metrics：
  - fit duration；
  - predict latency；
  - failed jobs；
  - model count；
  - artifact size；
  - memory usage；
  - numerical warning count。
- 增加带脱敏的结构化日志。

**验收标准**

- 内部 demo 部署可以通过一个文档化命令启动。
- 运维 dashboard 能展示 job health 和 model health。

### 4.4 第 4 月：垂直产品化

**目标：** 围绕领域用例包装平台，而不是只销售通用建模能力。

#### 推荐首个垂直方向：保险风险建模

保险是最佳第一切入点，因为 GAMLSS 天然适合赔付频率、赔付强度、零膨胀、尾部风险、分位数和监管可解释性。

**任务**

- 建立保险工作流模板：
  - claim frequency model；
  - claim severity model；
  - zero-inflated 或 heavy-tail alternative；
  - quantile/risk interval report；
  - model comparison report；
  - calibration report。
- 增加带已知 ground truth 的合成保险数据生成器。
- 增加 notebook 和 API demo。
- 增加可导出的 PDF/HTML report。

**验收标准**

- Demo 能回答：“哪些客户同时具有高期望赔付和高尾部风险？”
- Demo 包含 distributional metrics，而不仅是 point prediction metrics。
- Demo 可本地运行，也可通过服务 API 运行。

### 4.5 第 5 月：规模、可靠性与企业适配

**目标：** 让平台具备 beta 客户可信度。

#### 工作流 D9：性能与 Backend 改进

**任务**

- 优化 WLS backend 选择：dense NumPy、sparse SciPy、JAX dense 和 batch prediction。
- 向量化慢速离散 residual/CDF 路径。
- 增加硬件感知 benchmark 报告。
- 只在 benchmark 证据充分的地方定义 GPU 加速。

**验收标准**

- 性能报告包含硬件、backend、数据形状、family 和置信区间。
- 至少一个瓶颈 workload 获得 2x 改进。

#### 工作流 D10：企业可靠性

**任务**

- 为 model registry 增加 backup/restore。
- 增加模型版本和 rollback。
- 增加 artifact version 兼容性测试。
- 为 Pro/Enterprise API 增加角色权限边界。
- 增加支持 runbook。

**验收标准**

- 失败部署可 rollback 且不丢失已注册模型。
- 支持人员可通过 job metadata 和日志诊断失败 fit。

### 4.6 第 6 月：Release Candidate 与商业就绪

**目标：** 冻结范围、包装报价并转化 pilot。

**任务**

- 发布 Core release candidate。
- 在发布窗口冻结 public API 变更。
- 发布 capability matrix、validation report、benchmark report 和 deployment guide。
- 最终确定 Pro/Enterprise packaging 和 pilot 合同。
- 举办 launch webinar 和技术 deep-dive。

**验收标准**

- Release candidate 通过所有必要 unit、integration、artifact、service 和 validation gates。
- 至少两个 design partner 完成技术评估。
- 至少一个 paid pilot、LOI 或 procurement process 处于活跃状态。

---

## 5. 市场与推广计划

### 5.1 定位

**核心信息：** OmniLSS 面向需要校准不确定性、分位数、尾部风险和可解释统计结构的团队，而不是只提供均值预测。

**差异化点**

- GAMLSS 风格分布回归的 Python/JAX 实现。
- 覆盖较广的分布族。
- R 一致性和数学验证作为可信资产。
- Open-source Core 加企业工作流层。
- 面向英语和中文用户的双语技术教育。

### 5.2 目标受众

| 受众 | 痛点 | 信息 | 转化目标 |
|---|---|---|---|
| 统计学家与研究者 | 需要 Python-native distributional regression | 可重复的 GAMLSS 风格建模与验证报告 | GitHub stars、引用、反馈 |
| ML 工程师 | 需要超越 point prediction 的不确定性 | 可部署 API 中的分位数、区间和校准 | 试用服务部署 |
| 保险/金融分析师 | 需要尾部风险和赔付建模 | 建模完整条件分布和风险区间 | Design-partner pilot |
| 医疗分析团队 | 需要生长/风险曲线和校准区间 | 带可解释参数的分布曲线 | 应用 demo 会议 |
| 企业平台团队 | 需要治理和模型运维 | Registry、audit log、monitoring 和 support | Paid beta |

### 5.3 内容日历

| 月份 | 内容 | 渠道 | 成功指标 |
|---|---|---|---|
| 第 1 月 | 实现级审计摘要和 roadmap | GitHub docs、blog、LinkedIn/X | 500+ 浏览，20+ GitHub 互动 |
| 第 1 月 | “为什么分布回归重要”科普 | Blog + 中文翻译 | 10 个合格 inbound 对话 |
| 第 2 月 | R/Python 一致性报告 | GitHub release notes、技术博客 | 5 个外部技术 review |
| 第 2 月 | Family capability matrix | Docs + issue templates | 减少支持沟通混乱 |
| 第 3 月 | 安全服务 MVP demo | 视频 + demo repo | 5 个试用部署 |
| 第 3 月 | 教程系列：fit、predict、quantiles、calibration | Docs + notebooks | 100 次 notebook 启动/下载 |
| 第 4 月 | 保险风险建模垂直 demo | Webinar + case study | 3 个 design-partner call |
| 第 5 月 | 性能和 benchmark 报告 | Blog + benchmark docs | 2 个外部 benchmark 复现 |
| 第 6 月 | Launch webinar 和 release candidate 公告 | Webinar、GitHub、newsletter | 1 个 paid pilot 或 LOI，10 个严肃 lead |

### 5.4 社区与可信度动作

- 诚实发布 validation failure，并使用状态标签。
- 增加 “good first validation issue” 任务，吸引社区参与 family testing。
- 邀请 R `gamlss` 用户比较输出并报告 mismatch case。
- 为保险、医疗、金融创建双语 examples。
- 第 6 月提交 JOSS pre-submission checklist。

---

## 6. 商业计划

### 6.1 商业假设

核心统计建模应保持开源以最大化信任和采用率。收入应来自企业级运维、自动化、治理、垂直工作流和支持。

### 6.2 产品包装

| Package | 目标用户 | 内容 | 定价假设 |
|---|---|---|---|
| Core OSS | 研究者、开发者 | Python package、本地拟合、验证文档 | 免费 |
| Pro Developer | 小团队 | Hosted/self-hosted API、model registry、报告、batch jobs | $299-$999/月 |
| Enterprise | 监管行业团队 | SSO、RBAC、audit log、私有部署、支持 SLA、自定义验证 | $25k-$150k/年 |
| Vertical Solution | 保险/医疗/金融团队 | 模板、报告、dashboard、咨询 onboarding | Pilot fee + 年费 |

### 6.3 第一切入点：保险

**为什么优先保险**

- 对 frequency/severity modeling 有明确需求。
- 天然需要 quantiles、intervals 和 tail-risk estimates。
- 保险行业统计文化成熟，更容易解释 GAMLSS。
- 报告和校准能支撑更高企业定价。

**Pilot offer**

- 4-6 周技术 pilot。
- 范围：一个数据集、一个工作流，与当前 GLM/GAM/baseline model 对比。
- 交付物：
  - fitted distributional models；
  - quantile 和 tail-risk report；
  - calibration report；
  - deployment/API demo；
  - recommendation memo。
- Pilot fee 目标：$5k-$15k，可抵扣年度合同。

### 6.4 销售动作

1. **识别 design partner**
   - 精算分析团队；
   - 风险建模团队；
   - 需要生长/风险曲线的健康分析团队。
2. **技术 discovery**
   - 当前模型和痛点；
   - 所需分布与指标；
   - 部署约束；
   - 合规和审计需求。
3. **Pilot proposal**
   - 固定范围；
   - 明确成功指标；
   - 数据处理条款；
   - 付费 pilot 或 LOI。
4. **转化**
   - 将 pilot report 转化为生产 roadmap；
   - 提供年度 Pro/Enterprise license 和 onboarding。

### 6.5 商业 KPI

| 时间 | KPI |
|---|---|
| 第 1 月末 | 选择 1 个垂直方向，建立 20 个目标账号列表，安排 5 个 discovery call |
| 第 2 月末 | 完成 product one-pager、定价假设、2 个 warm design-partner conversation |
| 第 3 月末 | 5 个试用用户或部署，3 个严肃 pilot lead |
| 第 4 月末 | 2-3 个 active design-partner pilot |
| 第 5 月末 | 1 个 paid beta proposal 或 LOI 进入谈判 |
| 第 6 月末 | 1 个 paid pilot/LOI/procurement process 活跃；形成可重复 pilot playbook |

---

## 7. 团队与治理计划

### 7.1 最小团队形态

| 角色 | 职责 | 第 1-2 月 | 第 3-4 月 | 第 5-6 月 |
|---|---|---:|---:|---:|
| Core statistical engineer | Families、RS/CG、validation | 全职 | 全职 | 全职 |
| Platform/backend engineer | API、jobs、registry、deployment | 兼职 | 全职 | 全职 |
| Research lead | R consistency、metrics、paper package | 全职 | 兼职 | 兼职 |
| Developer advocate | Tutorials、demos、双语 docs | 兼职 | 全职 | 全职 |
| Commercial lead | Pilots、pricing、partnerships | 兼职 | 全职 | 全职 |

### 7.2 每周节奏

- 周一：计划与 blocker triage。
- 周三：validation/build review。
- 周五：demo、metrics review 和文档更新。
- 每月：根据 go/no-go 标准进行发布 checkpoint。

### 7.3 必需 Dashboard

- 工程 burndown 和 release gates。
- Family capability 和 validation matrix。
- Benchmark history。
- Service health 和 job metrics。
- Marketing funnel。
- Pilot pipeline。

---

## 8. 风险登记表

| 风险 | 概率 | 影响 | 缓解方式 |
|---|---:|---:|---|
| 验证暴露重大 family bug | 高 | 高 | 优先前 10 个 family；其他标为 experimental；公开已知限制 |
| Service MVP 消耗过多时间 | 中 | 高 | 第 3 月范围限定为内部 MVP，避免企业特性蔓延 |
| GPL 限制嵌入式商业采用 | 中 | 高 | 销售 hosted/self-hosted service 和企业运维，不销售闭源嵌入 |
| JAX 加速不明显 | 中 | 中 | 优先营销验证与分布指标；只在 benchmark 充分处启用 GPU |
| Design partner 未转化 | 中 | 高 | 第 1 月开始外联；聚焦保险；使用付费诊断 pilot |
| 双语文档漂移 | 中 | 中 | 每份战略文档要求 cross-link 和更新 checklist |

---

## 9. Go/No-Go Gates

### 第 2 月末：Core 可信度 Gate

只有满足以下条件才继续扩大投入：

- artifact roundtrip 对核心公式可靠；
- 前 10 个 family validation matrix 已启动；
- 不支持能力已被标注；
- 至少一个 validation report 可发布。

### 第 4 月末：产品切入点 Gate

只有满足以下条件才继续推进商业化：

- service MVP 可运行垂直 demo；
- 至少两个 design partner 深度参与；
- 一个垂直 workflow 能产出清晰商业报告；
- Pro/Enterprise 范围已明确。

### 第 6 月末：商业就绪 Gate

只有满足以下条件才进入正式商业推广：

- release candidate 通过必要 gates；
- validation 和 benchmark 证据可公开或可分享给客户；
- 至少一个 paid pilot、LOI 或 procurement process 存在；
- support 和 deployment playbook 可被核心作者以外的人使用。

---

## 10. 立即执行的 10 个动作

1. 创建 family capability registry，并按 feature 标记所有 family 为 validated、experimental 或 unsupported。
2. 实现 artifact schema v2 和 roundtrip tests。
3. 从默认 JSON model artifact 中移除完整训练数据。
4. 增加恶意公式 parser tests。
5. 优先为 `NO`、`GA`、`PO` 生成 validation report。
6. 起草保险风险 demo 数据生成器和 notebook。
7. 定义 Pro/Enterprise one-page product offer。
8. 建立 20 个保险或风险分析团队目标账号列表。
9. 发布双语 roadmap announcement。
10. 安排第一次月度 release checkpoint，并设置明确 go/no-go 标准。
