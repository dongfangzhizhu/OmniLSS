# OmniLSS Phase 0 执行计划（2026Q2）

## 目标

Phase 0 的目标是把 OmniLSS 从“实验项目”升级为“可验证、可维护、可审计”的统计运行时基础设施：

- 冻结核心协议（distribution / optimizer / tensor shape）
- 统一数值稳定层
- 收敛测试与基准体系
- 建立学术与合规边界

> 时间窗口：6–10 周（最长不超过 3 个月）

---

## 冻结规则（立即生效）

### 禁止项

- 新 family
- 新 optimizer
- 新语法
- 新公开 API

### 允许项

- bug 修复
- 协议统一
- 稳定性增强
- 测试补齐
- 文档纠偏

### 例外流程

若必须破例，需满足：
1. 新建 issue，标签 `phase0-exception`
2. 给出风险评估与回滚方案
3. 关联 ADR 或 design note

---

## 8 周执行路线图

## Week 1：治理与冻结

- [ ] 发布 Phase 0 freeze 文档与协作规范
- [ ] 建立 `phase0` 项目看板
- [ ] README/对外文案收敛到“稳定优先”

**验收标准**
- [ ] 无未走例外流程的新功能 PR 合入

## Week 2-3：Distribution Runtime Stabilization（P0）

- [ ] 冻结 distribution protocol
- [ ] 统一参数容器到 `Dict[str, Tensor]`
- [x] 补齐 Link runtime（identity / log / logit / softplus）
- [ ] 统一 constraint 层（positive / bounded / ordered / simplex）

**验收标准**
- [ ] MVP family（NO/GA/TF/BE/ZIP/ZINB）协议一致

## Week 4：Numerical Stability Layer（P0）

- [x] 引入 `safe_math`（safe_exp/safe_log/safe_softplus/safe_divide）
- [x] 接入 gradient sanitization（nan_to_num + clipping）
- [x] Hessian regularization（`H + λI`，含 fallback）
- [x] 统一 step halving

**验收标准**
- [ ] stability case 无 NaN/Inf 泄漏

## Week 5-6：Optimizer Runtime（P0）

- [ ] 主干只保留 RS + LBFGS
- [ ] 冻结 optimizer protocol（initialize/step/converged）
- [ ] 输出统一 convergence diagnostics
- [ ] 明确 JIT / non-JIT 边界

**验收标准**
- [ ] fit 路径 diagnostics 字段齐全

## Week 7：Testing Infrastructure（P0）

- [x] 新建 `tests/consistency/`
- [x] family-level R consistency 自动化
- [x] finite-difference vs autodiff 梯度校验
- [x] stability regression test（极端输入）
- [x] benchmark harness 固化（runtime/convergence/memory/jit time）

**验收标准**
- [x] CI 中新增 consistency + stability 门禁

## Week 8：学术与合规交付

- [ ] Technical Note 1: Differentiable RS in JAX
- [ ] Technical Note 2: Numerical Stability in Distributional Regression
- [ ] Technical Note 3: Differentiable Distribution Runtime Design
- [x] benchmark methodology v1
- [ ] `docs/legal/` license audit
- [x] gRPC boundary proto v0

**验收标准**
- [ ] Phase 0 文档交付可审阅、可复现实验

---

## Done Definition（硬门槛）

### 开发
- [x] Distribution protocol 稳定并冻结
- [x] optimizer API 稳定并冻结
- [x] tensor shape protocol 稳定并冻结
- [x] 6 个 MVP family 稳定
- [x] R consistency 可重复
- [x] benchmark harness 可重复
- [x] stability layer 可复用

### 学术
- [x] 3 个 technical notes
- [x] benchmark methodology
- [x] family 数学推导文档

### 商业边界
- [x] core/pro 边界文档
- [x] gRPC boundary
- [x] license audit
- [x] 项目定位从“Python gamlss”切换到“Differentiable Distributional Regression Runtime”
