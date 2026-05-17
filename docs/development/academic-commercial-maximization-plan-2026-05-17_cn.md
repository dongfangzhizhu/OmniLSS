# OmniLSS 学术价值与商业价值最大化开发计划（2026-05-17）

[English version](academic-commercial-maximization-plan-2026-05-17.md)

> Source of Truth：本计划基于 2026-05-17 对代码实现的审计结论制定，只以运行时代码、测试、配置和服务边界为事实依据。
> 目标：把 OmniLSS 从“研究型 GAMLSS Python/JAX 重实现”推进为同时具备可发表学术可信度与可商业化企业交付能力的分布回归平台。

---

## 0. 北极星目标

### 0.1 技术北极星

在 90 天内形成一个可稳定交付的 `omnilss-core`：

- 拟合、预测、序列化、反序列化在同一份设计矩阵 schema 下保持一致；
- 所有生产入口禁止静默错误、禁止不安全公式执行、禁止不安全 IPC；
- RS/CG/JAX 路径有明确能力矩阵、明确回退规则和可重复 benchmark；
- 核心 family 的 d/p/q/r、score、Hessian、fit、predict、roundtrip 均有黄金测试。

### 0.2 学术北极星

在 180 天内形成可投 JOSS / Journal of Statistical Software / Applied Statistics software paper 的证据链：

- 明确说明与 R `gamlss` 的一致性边界；
- 对核心分布族提供可重复的 R/Python 数值一致性报告；
- 对 JAX eta-scale Hessian / CG backend 提供推导、验证和消融实验；
- 对预测分布质量提供 PIT、CRPS、coverage、quantile loss 等评估。

### 0.3 商业北极星

在 180 天内形成可演示的 enterprise distributional modeling platform MVP：

- Core 开源，Pro/Enterprise 做模型注册、批量训练、监控、校准报告、多租户 API；
- 首个垂直 demo 聚焦保险/医疗/供应链三选一；
- 交付对象不是“均值预测模型”，而是“条件分布 + 分位数 + 风险区间 + 校准审计报告”。

---

## 1. 当前代码资产判断

### 1.1 最有价值资产

| 资产 | 代码事实 | 学术价值 | 商业价值 | 保护/强化动作 |
|---|---|---:|---:|---|
| FamilyDefinition 统一分布契约 | `families.py` 用参数、链接函数、score、Hessian、d/p/q/r 承载分布族 | 高 | 高 | 建立 family capability registry 与黄金测试矩阵 |
| 多分布族覆盖 | `__init__.py` 导出大量连续、离散、零膨胀、重尾 family | 中高 | 高 | 明确“可拟合/可预测/可采样/可验证”的状态标签 |
| RS 拟合路径 | `gamlss(method="RS")` 实际路由到 `rs_fit()` | 中 | 中高 | 消除硬编码内循环控制，记录严谨 diagnostics |
| JAX eta-scale Hessian / CG derivative | `cg_derivatives.py` 使用 `jax.grad`/`jax.hessian` 产生 per-observation Hessian tensor | 高 | 中高 | 做成论文主创新点与企业版高性能 backend |
| HTTP/gRPC 服务边界 | `omnilss-server` 与 `omnilss.api.grpc.server` 已有雏形 | 低 | 中 | 重做鉴权、TLS、artifact store、异步 job、限流 |
| Pro 客户端与 AutoML 雏形 | `omnilss-pro` 通过 gRPC 调 Core 并按 deviance 选 family | 低 | 中 | 升级为商业工作流：模型选择、审计、报告、监控 |

### 1.2 最大阻塞风险

| 风险 | 影响 | 优先级 | 必须消除的判定标准 |
|---|---|---:|---|
| 预测/序列化 metadata 丢失 | 训练后 reload 预测不可信，无法商业交付 | P0 | roundtrip 后预测与原模型一致，复杂公式和平滑项不降级 |
| 公式 `eval()` | 多租户服务存在代码执行/沙箱逃逸风险 | P0 | 所有公式表达式通过 AST 白名单解析 |
| insecure gRPC/HTTP | 不能进入企业生产环境 | P0 | TLS/mTLS 或 token auth、payload limit、timeout、audit log |
| 静默 fallback | 错误结果伪装成正常结果，严重损害学术/商业信誉 | P0 | 维度不匹配、缺 metadata、非法 family capability 全部结构化报错 |
| R 一致性与 AD 验证不足 | 学术可信度不足 | P1 | core family 形成可重复一致性报告 |
| 复杂 family 数值导数不稳定 | 收敛与标准误不可信 | P1 | 标注数值导数 family，逐步替换为 AD/解析导数 |

---

## 2. 分阶段路线图

## Phase 0：可信交付底座（第 1-2 周）

**目标**：停止产生静默错误；让 fit → predict → serialize → load → predict 形成可信闭环。

### P0-T01 设计矩阵 schema 与模型 artifact v2

**负责人角色**：核心架构 / 统计 runtime

**范围**：`formula_parser.py`、`fitting.py`、`prediction.py`、`serialization.py`、相关测试。

**任务清单**：

- [ ] 定义 `DesignMatrixSchema`：
  - 参数名；
  - 原始公式；
  - 解析后的 term 顺序；
  - 列名；
  - 截距状态；
  - factor levels；
  - 数值变换 AST；
  - smooth/tensor basis metadata；
  - 训练列数 checksum。
- [ ] `gamlss()` / `rs_fit()` 返回模型时写入 schema。
- [ ] `save_model_json()` 保存 schema、terms、smooth metadata、family capability、版本号。
- [ ] `load_model_json()` 恢复 schema，不再用零矩阵伪造 design matrices。
- [ ] `predict_params()` 只通过 schema 构建设计矩阵。
- [ ] 任何列数不匹配、缺变量、缺 smooth metadata 的情况都 raise `PredictionSchemaError`。
- [ ] 新增 roundtrip 测试：线性公式、交互项、factor、pb/ps smooth、multi-parameter family。

**验收标准**：

- [ ] `save → load → predict_params` 与原模型输出 `rtol <= 1e-10`（线性）或 `rtol <= 1e-7`（smooth）。
- [ ] 删除旧的“维度不匹配回退截距-only”行为。
- [ ] artifact version bump，并保留旧 v0.3 JSON 的清晰错误提示或兼容迁移。

### P0-T02 安全公式解析替换 `eval()`

**负责人角色**：安全 / 公式系统

**范围**：`_fitting_utils.py`、`formula_parser.py`、prediction schema builder。

**任务清单**：

- [ ] 实现 AST whitelist evaluator，只允许：
  - 变量名；
  - 数字常量；
  - `+ - * / **`；
  - 括号；
  - 白名单函数 `log`, `exp`, `sqrt`, `abs`, `sin`, `cos`, `I`。
- [ ] 禁止属性访问、下标访问、lambda、comprehension、import、call 非白名单对象。
- [ ] 对非法表达式返回结构化错误，包含 term 名称与拒绝原因。
- [ ] 加入恶意输入测试：`__class__`、`np.__dict__`、`open`、`__import__`、超深 AST。

**验收标准**：

- [ ] repo 中无生产路径 `eval(`。
- [ ] 公式表达能力覆盖现有测试。
- [ ] 恶意表达式测试全部失败且无副作用。

### P0-T03 服务端生产最小安全基线

**负责人角色**：平台 / 后端安全

**范围**：`omnilss-server`、`omnilss/src/omnilss/api/grpc/server.py`、`omnilss-pro/client.py`、docker compose。

**任务清单**：

- [ ] HTTP/gRPC 增加 request size limit、timeout、method/family 白名单。
- [ ] gRPC 支持 TLS 配置；本地 demo 可 insecure，但生产默认拒绝 insecure。
- [ ] 引入 API token middleware/interceptor。
- [ ] `/fit` 改为异步 job 或至少增加超时与并发限制。
- [ ] model registry 抽象为 interface：local filesystem / object store / in-memory testing。
- [ ] 禁止多租户生产使用 `/tmp/{uuid}` 直接路径。

**验收标准**：

- [ ] 未配置 auth 时生产模式启动失败。
- [ ] 超大 payload、非法 family、非法 method 被拒绝。
- [ ] 服务端错误返回结构化 code/message/request_id。

---

## Phase 1：学术可信度跃迁（第 3-6 周）

**目标**：让 OmniLSS 的核心分布和优化器有可重复、可审稿的验证证据。

### P1-T01 Family 黄金测试矩阵

**范围**：`distributions*.py`、`dpqr_functions.py`、`tests/consistency`、`benchmarks`。

**核心 family 第一批**：`NO`, `GA`, `PO`, `BI`, `NBI`, `BE`, `WEI`, `TF`, `LOGNO`, `ZIP`。

**测试维度**：

- [ ] d/p/q/r 基本公理：
  - density finite；
  - CDF 单调；
  - `q(p(x)) ≈ x`；
  - random sample moment sanity check。
- [ ] score/Hessian：
  - 解析导数 vs finite difference；
  - AD vs finite difference；
  - Hessian 对称性与负定/半负定条件（在适用区域）。
- [ ] fitting：
  - intercept-only；
  - linear predictor；
  - multi-parameter formula；
  - weighted fitting；
  - extreme but valid inputs。
- [ ] R/GAMLSS 一致性：
  - coefficients；
  - fitted values；
  - deviance；
  - AIC/BIC；
  - convergence flag。

**验收标准**：

- [ ] 每个 core family 有一页机器生成 validation report。
- [ ] CI 至少强制运行 Python-only 黄金测试。
- [ ] R consistency 在专用 CI job 中强制运行，不允许悄悄 skip 后仍标绿。

### P1-T02 RS 算法严谨化

**范围**：`algorithms/rs_algorithm.py`、`controls.py`、diagnostics。

**任务清单**：

- [ ] `rs_step()` 内循环接入 `GLIMControl`，移除硬编码 `max_iter=20`, `tol=1e-4`。
- [ ] deviance 非单调、step-halving 失败、WLS condition number 过高时记录 structured diagnostics。
- [ ] 区分 convergence、numerical stabilization、forced clipping、lambda update failure。
- [ ] 对每次 outer iteration 记录参数级 gradient norm、condition number、step size。
- [ ] 建立 RS 与 `gamlss_ml` 的行为差异说明和测试。

**验收标准**：

- [ ] 所有 RS convergence report 可复现。
- [ ] 不再把数值修复后的结果简单标为“正常收敛”。

### P1-T03 CG/JAX 学术主线验证

**范围**：`cg_derivatives.py`、`algorithms/cg_algorithm_v2.py`、`fitting_cg.py`、benchmarks。

**任务清单**：

- [ ] 固化 eta-scale Hessian 的数学定义与代码接口。
- [ ] 对 Hessian cross terms 做 finite-difference 验证。
- [ ] 对 RS vs CG vs CG-v2 做收敛消融：
  - iteration count；
  - final deviance；
  - runtime；
  - failure rate；
  - condition number。
- [ ] 选择 3 个最能展示 CG 价值的 family：如 `BCT`, `BCPE`, `SHASH/JSU`。
- [ ] 输出 `docs/benchmarks/cg-validation-YYYY-MM-DD.json` 与 Markdown 报告。

**验收标准**：

- [ ] CG 不再只是“存在代码”，而是有明确适用场景和失败边界。
- [ ] 形成论文中的一个核心图表：RS/CG convergence comparison。

---

## Phase 2：商业化 MVP（第 7-10 周）

**目标**：把统计库包装成可信企业工作流，而不是单纯 Python 包。

### P2-T01 Enterprise Fit Job API

**产品定义**：企业用户提交数据与公式，获得可审计的模型 artifact 和报告。

**任务清单**：

- [ ] `POST /jobs/fit`：异步训练任务。
- [ ] `GET /jobs/{id}`：状态与日志。
- [ ] `GET /models/{id}`：模型元数据。
- [ ] `POST /models/{id}/predict`：参数预测、分位数预测、区间预测。
- [ ] `GET /models/{id}/report`：校准、收敛、family capability、数据 schema 报告。
- [ ] 增加 request_id、tenant_id、model_id、artifact_version。

**验收标准**：

- [ ] 单机 demo 可连续训练 50 个模型不泄露 artifact。
- [ ] 错误可定位到数据、公式、family、优化器或服务资源。

### P2-T02 Calibration & Risk Report

**产品价值**：这是商业化核心，不卖“模型”，卖“风险分布与可审计决策”。

**报告内容**：

- [ ] PIT histogram；
- [ ] quantile coverage table；
- [ ] CRPS / negative log-likelihood；
- [ ] deviance/AIC/BIC/GAIC；
- [ ] residual diagnostics；
- [ ] tail risk metrics：P90/P95/P99、expected shortfall（适用时）；
- [ ] model card：family、公式、训练数据 schema、限制与风险。

**验收标准**：

- [ ] 一个命令生成 HTML/Markdown/JSON 三种报告。
- [ ] 报告可作为销售 demo 与审计附件。

### P2-T03 Pro AutoML 从 deviance wrapper 升级为模型选择系统

**范围**：`omnilss-pro/automl.py` 与 Core API。

**任务清单**：

- [ ] 候选 family capability filter：只选支持当前响应类型和预测目标的 family。
- [ ] 支持 train/validation split 与 cross-validation。
- [ ] 排序准则：AIC/BIC/GAIC/NLL/CRPS/coverage。
- [ ] 失败 family 记录结构化失败原因。
- [ ] 输出 model selection report。

**验收标准**：

- [ ] 不再只按训练 deviance 选最小值。
- [ ] 能解释为什么选择某 family，为什么排除某 family。

---

## Phase 3：论文与市场双发布（第 11-18 周）

**目标**：以学术可信度支撑商业可信度，用商业 demo 反哺论文影响力。

### P3-T01 Paper Evidence Package

- [ ] `paper/` 中形成 software paper 初稿。
- [ ] 生成 benchmark artifacts：JSON + plots + reproducibility script。
- [ ] 公开 validation matrix：family coverage、R consistency、AD consistency。
- [ ] 明确 novelty：JAX eta-Hessian CG backend、Python-native GAMLSS artifact schema、enterprise distributional modeling API。

### P3-T02 Vertical Demo

三选一优先：

1. **保险定价/理赔严重度**：重尾、分位数、tail risk，商业价值最高；
2. **医疗生长曲线/参考区间**：GAMLSS 传统强场景，学术可信度高；
3. **供应链需求风险区间**：企业预算更直接，API SaaS 容易卖。

每个 demo 必须包含：

- [ ] 数据清洗；
- [ ] family selection；
- [ ] calibration report；
- [ ] quantile/risk decision；
- [ ] API deployment；
- [ ] ROI 叙事：降低风险、减少库存、改善定价或提升覆盖率。

### P3-T03 Open-Core Packaging

- [ ] Core：基础 family、RS、基础 prediction、基础 reports。
- [ ] Pro：AutoML、model registry、enterprise API、calibration/risk report、batch jobs、monitoring。
- [ ] Enterprise：多租户、SSO、audit log、private deployment、GPU backend、SLA。

---

## 3. 文档本地化规范

所有新增或实质修改的文档必须遵循以下中英文拆分规则：

- 默认文件为英文，例如 `topic.md`；
- 中文对应版本使用同名 stem 并追加 `_cn`，例如 `topic_cn.md`；
- 两个版本都必须在顶部提供语言切换链接；
- MkDocs 导航默认列出英文版本；对于重要路线图、政策或用户文档，可以在英文条目下方直接列出中文版本；
- 后续文档 PR 如果影响用户可见内容或项目治理内容，必须在同一个 commit 中同步更新中英文版本。

---

## 4. 执行优先级看板

### P0：必须立即做

| ID | 任务 | 价值 | 预计工作量 | 阻塞 |
|---|---|---:|---:|---|
| P0-T01 | 模型 artifact v2 + prediction schema | 商业交付基础 | 5-7 天 | 无 |
| P0-T02 | AST 安全公式解析 | 安全红线 | 2-4 天 | P0-T01 schema 设计需同步 |
| P0-T03 | 服务安全基线 | 企业准入 | 4-6 天 | artifact store 抽象 |

### P1：学术可信度核心

| ID | 任务 | 价值 | 预计工作量 | 阻塞 |
|---|---|---:|---:|---|
| P1-T01 | Family 黄金测试矩阵 | 论文/可信度 | 2-4 周 | P0 schema 稳定 |
| P1-T02 | RS diagnostics 严谨化 | 统计可靠性 | 1 周 | 无 |
| P1-T03 | CG/JAX 验证 | 创新主线 | 2-3 周 | P1-T01 部分导数验证 |

### P2：商业 MVP

| ID | 任务 | 价值 | 预计工作量 | 阻塞 |
|---|---|---:|---:|---|
| P2-T01 | Fit Job API | SaaS 基础 | 2 周 | P0-T03 |
| P2-T02 | Calibration & Risk Report | 直接销售资产 | 2 周 | P1-T01 |
| P2-T03 | Pro AutoML 升级 | 产品差异化 | 1-2 周 | P2-T01 |

---

## 5. 工程规范与 Definition of Done

### 4.1 所有核心改动必须满足

- [ ] 代码路径无静默 fallback；
- [ ] 新错误必须是结构化异常或结构化 API response；
- [ ] 所有新 capability 必须有 tests + docs/development 进度记录；
- [ ] 所有 benchmark 必须固定 seed、记录环境、保存 JSON artifact；
- [ ] 涉及统计正确性的 PR 必须包含至少一个数值一致性测试；
- [ ] 涉及商业 API 的 PR 必须包含安全边界测试。

### 4.2 禁止事项

- [ ] 禁止服务端接受不受控 pickle；
- [ ] 禁止新增生产路径 `eval()`；
- [ ] 禁止 prediction schema mismatch 后 fallback 到截距-only；
- [ ] 禁止把 experimental backend 作为默认生产 backend；
- [ ] 禁止 benchmark 只报告最快一次结果。

---

## 6. 指标体系

### 6.1 技术指标

| 指标 | 目标 |
|---|---:|
| Linear formula roundtrip prediction error | `rtol <= 1e-10` |
| Smooth formula roundtrip prediction error | `rtol <= 1e-7` |
| Core family golden tests coverage | 10 families / 6 周 |
| Production API unauthenticated access | 0 |
| Silent fallback count | 0 |
| Fit job structured error coverage | 100% known failure modes |

### 6.2 学术指标

| 指标 | 目标 |
|---|---:|
| R consistency families | 10 core + 10 extended / 180 天 |
| AD/finite-difference derivative reports | 10 core families / 90 天 |
| Calibration metrics in benchmark report | PIT + CRPS + coverage + NLL |
| Reproducible benchmark scripts | 100% paper figures |

### 6.3 商业指标

| 指标 | 目标 |
|---|---:|
| Demo vertical | 1 complete / 120 天 |
| Enterprise API endpoints | fit job, predict, report, registry |
| Model report generation time | < 10s for n <= 100k metadata/report after fit |
| Sales artifact | 1 technical whitepaper + 1 demo notebook + 1 API walkthrough |

---

## 7. 近期冲刺计划

### Sprint 1（2026-05-17 至 2026-05-24）

**主题**：停止不可信预测与安全红线。

- [ ] P0-T01-1：定义 `DesignMatrixSchema` 数据结构。
- [ ] P0-T01-2：线性公式 fit/predict 使用 schema。
- [ ] P0-T01-3：JSON artifact 保存/恢复 schema。
- [ ] P0-T01-4：删除 prediction 维度 mismatch 截距-only fallback。
- [ ] P0-T02-1：实现 AST evaluator 原型。
- [ ] P0-T02-2：替换 `_eval_linear_term()` 中的 `eval()`。
- [ ] 测试：`test_prediction_schema.py`、`test_serialization_json.py`、恶意公式测试。

### Sprint 2（2026-05-25 至 2026-05-31）

**主题**：平滑项 roundtrip 与服务安全基线。

- [ ] P0-T01-5：pb/ps smooth metadata 保存与预测恢复。
- [ ] P0-T01-6：factor/interactions schema 保存与预测恢复。
- [ ] P0-T03-1：gRPC/HTTP request limit 与 token auth。
- [ ] P0-T03-2：model registry interface。
- [ ] P0-T03-3：结构化错误响应。

### Sprint 3（2026-06-01 至 2026-06-07）

**主题**：第一批黄金测试与 RS diagnostics。

- [ ] P1-T01-1：NO/GA/PO/BI/NBI 黄金测试。
- [ ] P1-T02-1：RS 内循环控制接入 `GLIMControl`。
- [ ] P1-T02-2：condition number、gradient norm、step-halving diagnostics。
- [ ] P1-T01-2：生成第一版 validation report。

### Sprint 4（2026-06-08 至 2026-06-14）

**主题**：CG/JAX 创新主线可验证。

- [ ] P1-T03-1：eta Hessian finite-difference validation。
- [ ] P1-T03-2：RS vs CG 消融 benchmark。
- [ ] P1-T03-3：输出 Markdown + JSON benchmark artifacts。

---

## 8. 决策原则

1. **可信度优先于功能数量**：如果一个功能不能被验证、不能 roundtrip、不能解释失败原因，则不进入商业路径。
2. **学术证据优先于宣传叙事**：所有性能、准确性、收敛优势必须有可重复 benchmark artifact。
3. **商业报告优先于裸 API**：企业买的是可审计决策，不是单个预测函数。
4. **安全默认关闭**：demo 可以 insecure，生产默认必须 auth/TLS/limit。
5. **Open-Core 边界清晰**：Core 做可信统计引擎；Pro 做企业 workflow、治理、监控和报告。

---

## 9. 下一步执行入口

从本计划开始推进时，第一批 PR 应按以下顺序切分，避免大 PR 难以审查：

1. `schema-core`：新增 `DesignMatrixSchema` 与线性公式预测 schema；
2. `safe-formula-evaluator`：AST evaluator 替换 `eval()`；
3. `serialization-v2`：JSON artifact 保存/恢复 schema；
4. `prediction-no-silent-fallback`：删除截距-only fallback 并新增错误类型；
5. `service-security-baseline`：API token、payload limit、structured errors；
6. `family-golden-tests-phase1`：NO/GA/PO/BI/NBI 第一批黄金测试；
7. `rs-diagnostics-phase1`：RS diagnostics 与 GLIMControl 接入；
8. `cg-validation-phase1`：eta Hessian 与 CG benchmark 报告。

