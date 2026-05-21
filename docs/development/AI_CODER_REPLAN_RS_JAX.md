# OmniLSS RS_JAX 重规划（基于代码实现与最新实测认知）

> 适用对象：AI Coder 执行代理。
> 
> 事实来源约束：本计划仅基于仓库代码路径（尤其 `omnilss/src/omnilss/algorithms/jax_rs_core.py`、`jax_rs_integration.py`、`rs_algorithm.py`、benchmark 脚本）与你给出的最新实测结论，不依赖 README/宣传文本。

---

## 🎯 一、项目核心技术本质（一句话）
当前 `RS_JAX` 的真实本质不是“纯 GPU 加速 RS”，而是“**NumPy 初始化 + JAX while_loop IRLS 核心**”的混合路径，其瓶颈由 warm-start 成本与小矩阵 WLS (`p=2`) 的设备效率共同决定。

---

## 🛠️ 二、技术视角：深度诊断与重构建议

## 2.1 关键代码事实（必须先统一认知）
1. `gamlss_rs_jax` 在进入 JAX core 前做了大量 NumPy 初始化与 `lstsq`（含 family 分支冷启动）。
2. `jax_rs_fit_core` 使用 `jax.lax.while_loop + fori_loop`，内层 WLS 为 `jnp.linalg.lstsq`。
3. `max_inner` 默认值在 JAX core 与 integration 层均为 1（已与抗振荡策略一致）。
4. BI/WEI 等 family 在 integration 层含特化初始化逻辑，说明“初始化稳定性”是当前实现关键依赖。

## 2.2 新认知与工程结论映射
- CPU：NO/TF 仅小幅加速（约 1.2x/1.3x），其余 family 因 warm-start NumPy RS 成本被吞噬。
- 对 R：NO/WEI/GA 在大样本已出现明显优势，说明算法链路是可用的。
- 对 NumPy：CPU 上 JAX 基本持续落后，warm-start 成本主导。
- RTX 3060：即使排除 warm-start 的 pure JAX core 仍慢于 NumPy，说明问题不止 warm-start，而是 **while_loop + GPU lstsq 在小 p 场景的效率不经济**。

## 2.3 立即重构方向（按 ROI 排序）

### A. 把 `RS_JAX` 拆成可测三段（必须做）
- Stage 1: `init`（含 warm-start）
- Stage 2: `irls_core`（JAX while_loop）
- Stage 3: `materialize/sync`（含 `block_until_ready` 边界）

**交付要求**
- benchmark 输出三段耗时与占比。
- 每个 family、每个 n、每个 device 都要落盘（CSV/JSON）。

### B. 默认策略改为“设备感知路由”（必须做）
- 当 `p <= 4` 且单模型 IRLS：默认走 NumPy RS（CPU 快路径）。
- 仅在以下条件才走 JAX：
  - 批量拟合（多模型并行）或
  - `p` 足够大/矩阵足够宽，或
  - 用户强制指定 `method='RS_JAX'`。

### C. 保持 `max_inner=1`（锁定，不回退）
- 明确写入控制参数注释：多内层步在 warm-start 邻域易振荡。
- 若未来要支持 `max_inner>1`，必须先引入阻尼/线搜索（否则禁用）。

### D. 收敛稳定性护栏升级（必须做）
- `eta` family-specific clipping 表继续使用，但要在日志中记录 clipping 比例。
- 引入 `z` 分位数 winsorization 开关（默认关，诊断模式开）。
- 首步 beta 爆炸检测：若 `||beta||2` 超阈值，触发 fallback 初始化策略。

### E. JAX 路径现实化优化（可做，不承诺 GPU 提速）
- 尝试替换 `jnp.linalg.lstsq` 为法方程+Cholesky（仅在数值条件允许时）。
- 减少不必要 host sync，计时统一在边界采样。
- 明确文档声明：`RS_JAX` 在小 `p` 单模型场景**不保证**优于 NumPy。

---

## 🎓 三、学术视角：理论可靠性报告

## 3.1 正确性判断
- 当前实现延续 RS/IWLS 标准结构（score、hessian、link 导数构建工作变量），理论主干正确。
- `max_inner=1` 与 R `glim.fit` 实践一致，可视为“外层主导的固定点推进”，并非算法错误。

## 3.2 学术硬伤（当前必须承认）
1. 缺少系统性“收敛类型证明/分类”输出（仅有 deviance 变化，不足以学术完备）。
2. 缺少“同精度约束下”跨后端公平比较（wall-time 单指标不够）。
3. GPU 优势叙事与算子规模不匹配：当前问题更像顺序小规模 WLS 序列，不是典型 GPU 强项。

## 3.3 学术级补强任务
- 在 benchmark 中固定误差阈值（如 `Δdeviance`, `max|score|`）后比较吞吐。
- 引入重复实验与方差报告（至少 5 次，给均值/标准差）。
- 输出 family × n × p × device 的稳定域热力图（可复现实验协议）。

---

## 💰 四、商业视角：变现矩阵与最高收益路线图

## 4.1 Moat 重定义
真正护城河不是“JAX GPU 更快”，而是：
- 对 R 的稳定一致性；
- 多 family 工业级可收敛性；
- 自动路由到最经济后端的执行策略。

## 4.2 ROI 判断
- 继续押注“单模型小 p 的 GPU RS 加速”ROI 低。
- 应转向“CPU-first + 批量并行场景 GPU 化”的产品策略。

## 4.3 变现优先级
1. **API/SaaS（首选）**：对外承诺“精度阈值 + 可解释收敛 + 成本最优路由”。
2. **Open-core**：开源核心拟合，商业版提供批量调度、审计报表、SLA。
3. **技术资产化**：将一致性测试体系 + 路由策略打包成可审计资产用于合作/并购。

---

## ⚡ 五、Top 3 明日立即执行清单

1. **实现设备感知路由默认策略**：`p<=4` 单模型默认 NumPy，JAX 改为条件触发。  
2. **Benchmark 三段计时落地**：warm-start / core / sync 三段拆分并输出到报告。  
3. **稳定性防线上线**：保留 `max_inner=1`，加入首步 beta 爆炸监控与 fallback 初始化。

---

## AI Coder 执行计划（两周可交付）

## Week 1：稳定性与可观测性
- Day 1: 在 `RS_JAX` 路径接入三段计时埋点。
- Day 2: 增加 `beta_norm_first_step`、`eta_clip_ratio`、`sync_time_ms` 指标。
- Day 3: 实装首步爆炸 fallback（常数初始化失败时改 heuristic）。
- Day 4: 增加测试：振荡 case、爆炸 case、fallback case。
- Day 5: 更新 benchmark 报表模板，确保指标可读。

## Week 2：路由与性能策略
- Day 1: 增加 `device-aware` 路由规则（按 p/n/batch size）。
- Day 2: 在 `benchmarks/comprehensive_performance_test.py` 增加路由对照组。
- Day 3: 增加“同精度约束”比较模式。
- Day 4: 产出 family × device 推荐矩阵（自动生成）。
- Day 5: 冻结 v1 策略：默认 CPU-first，小 p 禁止盲目 GPU。

---

## 给 AI Coder 的执行铁律
- 先做 **可观测性**，再做优化；没有分段计时就不要宣称提速。
- 优先优化端到端（E2E），而不是单 kernel 峰值。
- 任何“GPU 更快”结论必须附：`family, n, p, batch_size, warm-start 开销占比`。
