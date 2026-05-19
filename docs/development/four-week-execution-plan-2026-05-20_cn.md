# OmniLSS 四周开发执行计划

> 创建日期：2026-05-20  
> 核心原则：**先完成 CG 算法，再做架构重构，全程保持性能诚实**  
> English version: [four-week-execution-plan-2026-05-20.md](./four-week-execution-plan-2026-05-20.md)

---

## 执行检查清单（每次开始任务前都要先检查）

- [x] 在实现前确认当前里程碑与日级任务。
- [x] 确认文档双语同步要求（中英文互相链接）。
- [x] 从 Week 1 Day 1–2 数学建模工作开始。
- [ ] 完成 Week 1 Day 1–2 文献/推导产出。
- [ ] 完成 Week 1 Day 3–4 交叉导数基础设施。
- [ ] 完成 Week 1 Day 5 数值验证报告。
- [ ] 完成 Week 2 CG 完整外循环实现与验证。
- [ ] 完成 Week 3 warm-start 解耦与基准修复。
- [ ] 完成 Week 4 集成与发布准备。

---

## 背景与架构共识

在路线图展开前，先确认三个架构事实：

1. **warm-start 陷阱**：当前 RS_JAX warm-start 把 NumPy RS 全路径带入计时，导致 WLS 的 GPU 加速效果被掩盖。在 RTX 3060 上，即使不含 warm-start 的纯 JAX 核心仍比 NumPy RS 慢 1.7–4.3×。
2. **小矩阵 GPU 劣势**：对于 IRLS WLS 的小设计矩阵 `[n, p=2]`，CPU LAPACK (`np.linalg.lstsq`) 明显快于 `jnp.linalg.lstsq`，后者受 kernel launch 开销影响，且顺序 IRLS 无法有效利用 GPU batch 并行。
3. **`max_inner=1` 是正确设定**：多次 inner IRLS 会在 warm-start 邻域振荡，这与 R `glim.fit` 一致，不应视为缺陷。

因此，JAX 的核心价值不是“让 RS 更快”，而是：

- 支撑 **CG**（CG 需要基于 AD 的跨参数二阶导，这是 JAX 的独特优势），
- 支撑 **大设计矩阵 / 多参数分布族** 的批量 GPU 推断。

---

## 里程碑总览

| 周次 | 核心目标 | 交付物 | 验收标准 |
| --- | --- | --- | --- |
| Week 1 | CG 理论建模 + 交叉导数基础设施 | `cross_derivatives.py` + 数学验证报告 | 与 R 数值对齐，误差 < 1e-6 |
| Week 2 | CG 完整实现 + 外循环 | `cg_algorithm.py`（完整版） | NO/GA/WEI 收敛且偏差与 R 对齐 |
| Week 3 | 架构重构：warm-start 解耦 + 基准修复 | 重构后的 RS + 诚实基准套件 | 报告明确区分 cold/hot 场景 |
| Week 4 | 集成 / 测试 / 发布准备 | v0.3.0-rc、PyPI 草案 | CI 全绿，License 迁移完成 |

---

## 周计划（规范任务源）

Week/Day 的详细任务、验收阈值、风险与 DoD 以 2026-05-20 已批准的英文计划原文为唯一规范来源执行。

为避免偏离，进度管理方式如下：

1. 每次开发前先查看上方执行检查清单；
2. 在进度文档中更新任务勾选状态；
3. 在 `docs/math/` 与 `docs/reports/` 记录可追溯证据。

---

## 当前立即执行任务（已启动）

### Week 1 / Day 1–2

- [x] 创建本执行计划文档。
- [x] 创建中文版本并互相交叉引用。
- [x] 创建 `docs/math/cg_derivation.md` 初版推导文档。
- [x] 创建 `docs/math/cg_derivation_cn.md` 并与英文版互链。
- [ ] 持续补全 block-matrix 全推导与分布族细节说明。
