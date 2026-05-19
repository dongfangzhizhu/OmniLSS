# OmniLSS 四周开发执行计划

> 创建日期：2026-05-20  
> 核心原则：**先完成 CG 算法，再做架构重构，全程保持性能诚实**  
> English version: [four-week-execution-plan-2026-05-20.md](./four-week-execution-plan-2026-05-20.md)

---

## 执行检查清单（每次开始任务前都要先检查）

- [x] 在实现前确认当前里程碑与日级任务。
- [x] 确认文档双语同步要求（中英文互相链接）。
- [x] 从 Week 1 Day 1–2 数学建模工作开始。
- [x] 完成 Week 1 Day 1–2 文献/推导产出（草案骨架已完成）。
- [x] 完成 Week 1 Day 3–4 交叉导数基础设施（初版实现已完成）。
- [x] 完成 Week 1 Day 5 数值验证报告（已发布本地 AD 验证报告）。
- [ ] 完成 Week 2 CG 完整外循环实现与验证（进行中：已新增外循环收敛脚手架与验证报告，R 对齐待完成）。
- [ ] 完成 Week 3 warm-start 解耦与基准修复。
- [ ] 完成 Week 4 集成与发布准备。

---

## 进展日志（按顺序推进）

### 2026-05-19 更新

- 已新增中英文四周执行计划文档。
- 已新增中英文 CG 推导工作区文档。
- 已实现 Week 1 Day 3–4 基础设施：
  - `omnilss/src/omnilss/derivatives/cross_derivatives.py`
  - `omnilss/tests/test_cross_derivatives.py`
- 已补充 NO/GA/WEI 的结构性验证（shape/对称性/有限值）。
- 已在 `docs/reports/` 发布中英文 Week 1 Day 5 交叉导数验证报告。
- 已发布 Week 2 中英文进展报告：`docs/reports/CG_FULL_VERIFICATION_2026_05_19.md` / `_cn.md`。
- 已启动 Week 2 Day 6–7：新增 `omnilss/src/omnilss/algorithms/cg_algorithm_full.py`，包含 `build_joint_scoring_matrix(...)` 与 `solve_joint_system(...)`。
- 已新增 Week 2 测试：`omnilss/tests/test_cg_algorithm_full.py`，覆盖 block 组装、线性求解一致性，以及外循环 deviance 下降校验。
- 新增 `run_cg_outer_loop(...)` 脚手架：按 `c_crit` 做相对 global deviance 收敛判定，并记录每轮步长历史。

> 说明：当前路线图文档仅定义 Week 1–Week 4。现阶段按顺序从 Week 1 向后推进；Week 4 之后任务在该计划中尚未定义，因此不能标记完成。

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
