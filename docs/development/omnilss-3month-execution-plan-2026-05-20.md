# OmniLSS 三个月执行拆解（仅 omnilss 核心库）

> 目标：在 12 周内把 `omnilss` 打磨到“基础稳固、可复现、可扩展、可生产嵌入”的完整可用状态。

## 0. 约束与事实（必须写入开发共识）

- 小设计矩阵（典型 p=2）场景下，当前 `RS_JAX` 在 GPU 上没有性能优势；`while_loop + jnp.linalg.lstsq` 与 kernel launch/sync 开销占主导。
- warm-start 可以抑制首步 IRLS 发散，但会引入 NumPy RS 开销，不能作为默认万能方案。
- `max_inner=1` 是稳定且与 R `gamlss::glim.fit` 对齐的默认策略，应作为 solver policy 固化。
- 短期策略：`CPU first + selective JAX + 强观测`，而不是盲目全面 GPU 化。

---

## 1. 总体里程碑（Definition of Done）

- 核心 family（NO/GA/WEI/TF/NBI/ZIP）拟合与预测可用。
- 拟合失败具备结构化错误码与可定位日志。
- 自动路由可解释：为何选 RS / RS_JAX。
- CI 包含：正确性、性质测试、复现测试、性能回归。

---

## 2. 12 周分解（可直接给 AI Coder）

## Month 1：地基（稳定性 + 路由 + 可观测）

### Week 1：Solver Router v1

**任务**
1. 新增路由策略对象（`method + reason + threshold + backend`）。
2. 新增 `auto_select_method_trace()`（不破坏现有 `auto_select_method()`）。
3. 在 `gamlss(..., method='auto')` 路径写入 `additional_slots['method_routing']`。

**验收**
- 调用方可拿到“为什么选这个后端”的结构化结果。
- 原有行为兼容（`auto_select_method()` 返回字符串不变）。

### Week 2：IRLS/RS 稳定性硬化

**任务**
1. `eta` clipping 与 step damping。
2. weights/hessian epsilon floor。
3. 将 `max_inner=1` 固化到默认策略与日志输出。

**验收**
- 常见 family 不再首步爆炸。

### Week 3：错误语义与遥测

**任务**
1. FitTelemetry：每轮 deviance、step_norm、eta_range、nan_count、阶段耗时。
2. structured error envelope（`code/stage/parameter/hint`）。
3. 清理关键路径 silent exception（至少模型构造与预测 schema 路径）。

**验收**
- 失败可追踪，非静默。

### Week 4：性能基线与回归门禁

**任务**
1. family×n×backend 基准矩阵。
2. 汇总 P50/P95、收敛率、精度偏差。
3. 建立性能退化阈值告警。

**验收**
- PR 可见性能回归状态。

## Month 2：功能补全（family + 预测契约 + 测试）

### Week 5-6：核心与中等复杂 family 稳态化
- NO/GA/WEI/TF 优先；NBI/ZIP/IG/BE 次优先。
- TF 重点优化 AD 开销（缓存与复用）。
- fixed parameter 行为统一（broadcast/长度校验/报错码）。

### Week 7：预测契约强化（schema-first）
- 强制 term 顺序、smooth metadata 校验。
- legacy fallback 改为显式开关（默认关）。

### Week 8：测试体系升级
- property-based + replay + stress tests。
- family 级别最小覆盖基线。

## Month 3：收口（自动路由产品化 + API 稳定 + 发布）

### Week 9：Auto Backend Selector v2
- 基于基准数据迭代规则参数。
- 输出“推荐原因 + 置信度”。

### Week 10：API 稳定化
- 冻结核心 API。
- 契约测试防止破坏式变更。

### Week 11：RC 硬化
- 全量回归、P0/P1 清理。

### Week 12：GA 交付
- 开发手册、性能手册、下阶段规划。

---

## 3. 立即执行（本周）

1. 完成 `auto_select_method_trace()` 并补单测。
2. 在 `gamlss(method='auto')` 落地 routing trace 元数据。
3. 增加文档：如何读取 routing reason，避免误用 GPU。

---

## 4. AI Coder 任务模板

每个任务必须包含：
- 变更文件列表
- 验收标准（功能 + 测试 + 性能）
- 回滚策略
- 风险与观测指标

