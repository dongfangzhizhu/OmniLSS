# OmniLSS 代码驱动重规划（面向 AI Coder 执行）

## 🎯 一、项目核心技术本质（代码事实一句话）
OmniLSS 当前的核心是：围绕 `fitting.py` 的 GAMLSS/RS 迭代框架，以 NumPy/JAX 混合实现 IWLS/WLS 与多分布 score/hessian 更新，并通过 `fitting_jit.py` 与 `rs_jax.py` 尝试将热点步骤 JIT 化，但主路径仍存在 CPU/NumPy 与 JAX 之间的切换开销和算法级收敛稳定性权衡问题。

---

## 0) 范围边界（必须遵守）
- **当前唯一主战场：`omnilss/` 核心库。** 所有功能完善、重构、测试与性能优化必须优先落在 `omnilss/src/omnilss` 与 `omnilss/tests`。
- `omnilss-pro/` 视为后续拆分项目：当前阶段仅允许“兼容性不破坏”检查，**不作为架构演进主路径**。
- `omnilss-server/` 仅做最小接口适配：以消费 `omnilss` 稳定 API 为目标，避免反向驱动核心算法设计。
- 里程碑验收标准从今天起改为：
  1) `omnilss` 公共 API 稳定（版本化、契约测试）
  2) `omnilss` 核心 family 拟合稳定
  3) `omnilss` 基准与 R 一致性可复现

- 验证与回归必须显式覆盖两个基准脚本：
  - `benchmarks/comprehensive_performance_test.py`（性能对比主脚本）
  - `benchmarks/comprehensive_r_consistency_test.py`（与 R 功能一致性主脚本）

---

## 🛠️ 二、技术视角：深度诊断与重构建议

## A. 架构与设计模式诊断

### 现状（代码证据）
- `omnilss/src/omnilss/fitting.py` 承担公式解析、设计矩阵构建、参数初始化、固定参数处理、平滑项入口等多重职责，呈现“大核心编排器”结构。  
- `omnilss/src/omnilss/fitting_jit.py` 提供 `create_jit_iwls_step/create_jit_wls_solver/create_jit_full_iteration` 等可组合 JIT 构件，但与主流程并未形成强约束的一致执行后端。  
- `omnilss/src/omnilss/rs_jax.py` 当前仅暴露 `_jax_wls/_jax_penalized_wls` 原子能力，功能边界偏窄，未形成完整 JAX-only RS pipeline。  

### 问题结论
1. **后端抽象不完整**：主算法“流程层”和“线代求解层”耦合仍强，导致 warm-start + mixed backend 成本不可控。  
2. **路径分叉导致维护负担**：NumPy/TF/JAX 各有实现，缺乏统一性能诊断接口。  
3. **重构优先级应转为“执行图单一化”**：先统一执行边界，再谈局部 kernel 优化。

### 重构动作（AI Coder 任务）
1. 引入 `SolverBackend` 协议层（numpy/jax/tf），统一暴露：
   - `compute_score_hessian`
   - `iwls_step`
   - `wls_solve`
   - `run_outer_iteration`
2. 将 warm-start 抽象为可插拔策略：
   - `NoWarmStart`
   - `NumpyWarmStart`
   - `FamilyHeuristicInit`
3. 强制 benchmark 中记录 **端到端时间拆分**：
   - warm-start
   - core RS iterations
   - host-device sync

---

## B. 生产完备性与鲁棒性

### 现状（代码证据）
- `fitting_jit.py` 中 `iwls_step` 已对 `score/hess/dr` 做 `isfinite` 保护和 clipping，说明项目已识别数值爆炸风险。  
- `_jax_wls` 与 `create_jit_wls_solver` 均依赖 `jnp.linalg.lstsq`，在小特征维度场景（p≈2）会放大 GPU launch overhead。  

### 关键风险（结合你给出的新认知）
1. **IRLS 首步爆炸**：常数初始化下 `z` 范围过宽，触发极大 beta，eta 出界。  
2. **内层多步振荡**：warm-start 邻域下多次 inner IRLS 往返震荡，收敛收益小于扰动。  
3. **GPU 性能错配**：当前问题规模属于“小 p、顺序迭代、频繁线代调用”，天然不利于 GPU。  

### 立即工程化修复
1. 将 `max_inner=1` 设为默认稳态策略（与 R `glim.fit` 一致），并在控制参数中显式注释“防振荡”。
2. 增加 `eta`/`z` 动态裁剪策略：
   - family-specific eta bounds
   - z winsorization（按分位数）
3. 在 JAX 路径中减少 `block_until_ready` 触发点，仅在计时边界调用。

---

## C. 性能与安全

### 性能结论（基于现有代码结构 + 你的实测认知）
- CPU 上 NO/TF 轻微加速（1.2x/1.3x），其余 family 被 warm-start 的 NumPy RS 吃掉收益。  
- JAX vs R 在 NO/WEI/GA 的大样本出现领先，说明“算法正确 + 线代可用”，但不是“架构最优”。  
- RTX 3060 上即便 pure JAX core 仍慢于 NumPy，根因在 `while_loop + lstsq` 的小矩阵 GPU 不经济性。  

### 安全与可运维性补强
1. 在 `omnilss` 核心层先实现输入规模阈值、运行时预算与超时钩子，再由 `omnilss-server` 透传，避免服务层特化逻辑反向污染核心。  
2. 对 family/链接函数参数增加统一合法性校验（NaN/Inf/范围）并记录审计日志，校验代码应位于 `omnilss` 核心入口。

---

## 🎓 三、学术视角：理论可靠性报告

## A. 算法正确性
- 项目采用经典 RS/IWLS 主干，`score + hessian + link derivative` 形成 working response/weights 的路径符合广义加性分布回归的标准推导。  
- `max_inner=1` 并非“偷步”，而是避免局部振荡的稳定策略，在外层循环中推进同样可收敛到实用解（与 R 实践一致）。  

## B. 学术硬伤与边界
1. **缺乏“收敛类型标签”**：当前实践上在做 damped/fixed-point 风格迭代，但未显式定义收敛准则分类（单调 deviance、参数范数、KKT-like 条件）。  
2. **GPU 优势叙事需修正**：该实现并不对应“并行大算子优化”问题，而是“顺序小问题反复求解”，理论上就不应默认 GPU 更快。  
3. **TF AD 的代价模型未内建**：自动微分带来的通用性与 per-call 高开销尚未进行理论与工程边界划分。

## C. 学术化增强建议
1. 引入统一收敛判据报告：`Δdeviance`, `||Δβ||2`, `max|score|`。  
2. 以 `benchmarks/comprehensive_performance_test.py` 为主入口，增加“同等精度约束下速度比较”（固定误差上限，而非仅 wall-time）。  
3. 基于 `benchmarks/comprehensive_r_consistency_test.py` 给每个 family 形成“可证明稳定区间”经验图（n, p, init, link）。

---

## 💰 四、商业视角：变现矩阵与最高收益路线图

## A. 核心技术壁垒（Moat）
- **最有价值资产不是单个 kernel，而是“多 family 的可复现拟合稳定性 + R 对齐能力 + 跨后端执行框架”**。  
- 当你能证明在关键 family（NO/WEI/GA 等）上：
  - 与 R 结果一致
  - 大样本更快
  - 收敛异常可解释
  这本身就是可销售的“可信计算资产”。

## B. ROI 与规模化判断
- 继续投入 GPU-only JAX RS 在当前 p=2 小设计矩阵场景 ROI 低。  
- 更高 ROI 路径：
  1. CPU first（NumPy/BLAS 优化）
  2. 大规模批量拟合场景再上 GPU（并行多模型，而非单模型 IRLS）

## C. 变现策略优先级
1. **企业 API/SaaS（首选）**：提供“风险建模/定价/精算”场景的稳定分布回归服务。  
2. **Open-core**：开源核心拟合器 + 商业版（并行调参、审计追踪、模型治理）。  
3. **技术授权/并购准备**：把 benchmark 与一致性测试体系打包为可审计资产。

---

## ⚡ 五、Top 3 立即执行清单（明天就改，且仅围绕 `omnilss` 核心）

1. **重构执行架构：后端协议 + warm-start 策略解耦**  
   - 目标：消除“warm-start 把 NumPy 全成本带入 JAX 路径”的结构性问题。  
   - 产出：`BackendAdapter` + `WarmStartPolicy` 接口与基准测试开关。

2. **稳定性优先：默认 `max_inner=1` + eta/z 边界控制 + 统一收敛报告**  
   - 目标：先解决振荡与首步爆炸，确保生产可控。  
   - 产出：family 级边界表、收敛日志、失败可回放样本。

3. **商业化基准体系：从“快不快”改为“精度约束下吞吐”**  
   - 目标：把工程优化直接映射到销售话术与 SLA。  
   - 产出：`R一致性 + 精度阈值 + 吞吐成本` 三维看板（数据来源固定为 `benchmarks/comprehensive_performance_test.py` 与 `benchmarks/comprehensive_r_consistency_test.py`），支撑 API 定价。

---

## AI Coder 执行节奏（建议两周）

### Week 1（架构与稳定，目录限定：`omnilss/src/omnilss` + `omnilss/tests`）
- Day 1-2：抽象 backend/warm-start 接口并迁移 NO/WEI 两个 family。  
- Day 3-4：实现 `max_inner=1` 默认与 convergence telemetry。  
- Day 5：补齐失败 case 回归测试（首步爆炸、振荡、NaN）。

### Week 2（性能与产品化，先核心后外围）
- Day 1-2：在 `benchmarks/comprehensive_performance_test.py` 新增 benchmark 分解计时与精度约束报告。  
- Day 3：完成 CPU-first 优化策略（小 p 走 NumPy 快路径）。  
- Day 4-5：结合 `benchmarks/comprehensive_r_consistency_test.py` 结果生成面向商业汇报的性能-成本-可靠性矩阵。


---

## Week 1 Task Checklist (English Default)

- [x] **Day 1**: Introduce backend and warm-start abstraction foundations in `omnilss` core.
- [x] **Day 2**: Add baseline warm-start policies (`NoWarmStart`, `FamilyHeuristicInit`) and typed outputs.
- [x] **Day 3**: Set inner-loop default to one step (`glim_control().cyc == 1`) for anti-oscillation baseline.
- [x] **Day 4**: Add core unit tests to lock warm-start behavior and inner-loop defaults.
- [x] **Day 5**: Re-validate roadmap linkage to benchmark entry points:
  - `benchmarks/comprehensive_performance_test.py`
  - `benchmarks/comprehensive_r_consistency_test.py`

### Next Implementation Gate
1. Wire `SolverBackend` into `rs_algorithm` execution path.
2. Add convergence telemetry fields (`delta_deviance`, `beta_l2_delta`, `max_abs_score`) to model slots.
3. Add timing split reporters in benchmark scripts.

## Week 2 Task Checklist (English Default)

- [x] **Day 1**: Wire backend abstraction into RS execution (`rs_step` supports `solver_backend`, default NumPy backend).
- [x] **Day 2**: Add convergence telemetry fields to model slots (`delta_deviance`, `beta_l2_delta`, `max_abs_score`).
- [x] **Day 3**: Add benchmark timing-split plumb lines in `benchmarks/comprehensive_performance_test.py`.
- [x] **Day 4**: Add/extend unit tests for backend protocol baseline behavior.
- [x] **Day 5**: Re-check checklist sequencing and confirm benchmark script linkage remains explicit.

## Week 3 Task Checklist (English Default)

- [x] **Day 1**: Add backend selection helper in RS (`numpy`/`jax`) using routing metadata.
- [x] **Day 2**: Implement a baseline `JaxSolverBackend` conforming to `SolverBackend` protocol.
- [x] **Day 3**: Wire selected backend into `rs_fit -> rs_step` update path.
- [x] **Day 4**: Expose selected backend in model `additional_slots` for observability.
- [x] **Day 5**: Add tests for JAX backend operations and backend-selection metadata persistence.

## Week 4 Task Checklist (English Default)

- [x] **Day 1**: Add backend-aware host-device sync timing capture in RS for JAX paths.
- [x] **Day 2**: Ensure `timing_split` schema is stable and always emitted.
- [x] **Day 3**: Add tests for timing split schema and non-negative values.
- [x] **Day 4**: Add tests for warm-start metadata persistence in model slots.
- [x] **Day 5**: Re-check roadmap sequencing and confirm benchmark compatibility remains intact.

## Week 5 Task Checklist (English Default)

- [x] **Day 1**: Persist timing split fields into benchmark result dataclass for structured reporting.
- [x] **Day 2**: Populate timing split fields from Python run outputs in benchmark loop.
- [x] **Day 3**: Add RS timing split summary table to benchmark markdown report.
- [x] **Day 4**: Re-check compatibility with existing report generation paths.
- [x] **Day 5**: Re-validate sequence completion and keep benchmark linkage explicit.

## Week 6 Task Checklist (English Default)

- [x] **Day 1**: Consume warm-start fitted values in RS initialization (mu path).
- [x] **Day 2**: Consume warm-start hints for non-mu parameter initial values.
- [x] **Day 3**: Keep fallback behavior safe when warm-start vectors are unavailable/mismatched.
- [x] **Day 4**: Add tests that explicit warm-start policy selection is persisted.
- [x] **Day 5**: Re-check checklist sequencing and maintain benchmark linkage continuity.

## Week 7 Task Checklist (English Default)

- [x] **Day 1**: Extend benchmark result schema with solver backend and warm-start policy fields.
- [x] **Day 2**: Populate backend/policy fields from model `additional_slots` in Python benchmark path.
- [x] **Day 3**: Add backend/policy usage summary section to markdown benchmark report.
- [x] **Day 4**: Preserve backward compatibility for missing metadata (`unknown` fallback).
- [x] **Day 5**: Re-check sequencing discipline and keep roadmap/benchmark alignment explicit.

## Week 8 Task Checklist (English Default)

- [x] **Day 1**: Extend R-consistency fitting notes with solver backend metadata.
- [x] **Day 2**: Extend R-consistency fitting notes with warm-start policy metadata.
- [x] **Day 3**: Keep metadata fallback safe (`unknown`) when slots are absent.
- [x] **Day 4**: Preserve existing timing split note format while appending backend/policy context.
- [x] **Day 5**: Re-check checklist sequence and ensure benchmark parity across performance/consistency scripts.

## Week 9 Task Checklist (English Default)

- [x] **Day 1**: Add parsing helper for backend/warm-start attribution from consistency notes.
- [x] **Day 2**: Add backend/warm-start columns in fitting report table (R-available mode).
- [x] **Day 3**: Add backend/warm-start columns in fitting report table (no-R mode).
- [x] **Day 4**: Keep fallback safety through `unknown` extraction for legacy notes.
- [x] **Day 5**: Re-check sequence adherence and keep performance/consistency reporting parity.

## Week 10 Task Checklist (English Default)

- [x] **Day 1**: Add consistency-report backend/policy usage summary section.
- [x] **Day 2**: Ensure summary is based on fitting-category rows only.
- [x] **Day 3**: Reuse parser helper for compatibility with legacy note strings.
- [x] **Day 4**: Keep graceful behavior when no fitting rows are available.
- [x] **Day 5**: Re-check sequence discipline and maintain parity with performance report attribution summaries.

## Week 11 Task Checklist (English Default)

- [x] **Day 1**: Extend performance benchmark result schema with convergence telemetry fields.
- [x] **Day 2**: Populate convergence telemetry from model `additional_slots` in Python benchmark path.
- [x] **Day 3**: Extend consistency fitting notes with convergence telemetry tags.
- [x] **Day 4**: Add a fitting convergence telemetry snapshot section in consistency markdown report.
- [x] **Day 5**: Re-check sequence discipline and keep report fallback behavior safe when fitting rows are absent.
