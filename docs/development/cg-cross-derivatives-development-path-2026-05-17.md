# CG 交叉导数完整实现开发路径（2026-05-17）

本文档用于解决 OmniLSS 中 CG（Cole-Green）算法“名实不符”的技术债：公开 API 与文案必须对应一个可验证的交叉导数实现，而不是仅把 RS 或普通 joint optimizer 包装成 CG。

## 1. 问题定义

### 1.1 什么才算“真正的 CG”

在 GAMLSS 中，RS 算法按参数块顺序更新，例如先更新 `mu`，再更新 `sigma`、`nu`、`tau`。这种方式隐含近似：当前参数块更新时忽略它与其他分布参数的二阶耦合。

CG 算法的核心差异是使用参数之间的交叉二阶导数。对每个观测 `i` 和两个线性预测子 `eta_k`、`eta_j`，需要显式计算：

```text
H_kj,i = d² l_i / (d eta_k,i d eta_j,i),  k != j
```

其中 `l_i` 是单个观测的 log-likelihood。只有当更新方向包含这些 `H_kj` 项时，算法才配得上 Cole-Green / cross-derivative correction 的名字。

### 1.2 当前代码中的两条路径

当前代码已经出现两种实现思路：

1. **coefficient-level full Hessian 路径**：`omnilss.fitting_cg.fit_cg()` 将所有 beta 系数展平后，直接对整体 log-likelihood 使用 `jax.grad` 与 `jax.hessian`，得到完整 observed-information matrix。这个路径天然包含 `beta_mu / beta_sigma / beta_nu / beta_tau` 的交叉块。
2. **eta-level CG v2 路径**：`omnilss/src/omnilss/algorithms/cg_algorithm.py` 中存在 `_compute_cross_derivatives()` 与 `_irls_step_with_adjustment()`，`cg_algorithm_v2.py` 中存在 `cg_outer_step()`，意图在 RS 风格的 IRLS 工作响应中加入交叉导数修正。

问题不是“完全没有交叉导数代码”，而是需要统一 API、数学定义、测试门禁和文案，让用户调用 `method="CG"` 时能明确知道走的是哪条可验证 CG 路径。

## 2. 推荐目标架构

### 2.1 保留两层实现，但明确职责

| 层级 | 推荐名称 | 用途 | 是否作为发布默认 |
|---|---|---|---:|
| coefficient-level Hessian | `CG_FULL_HESSIAN` | 小到中等规模、需要最严格数学一致性的 observed-information Newton / Fisher scoring 路径 | 是，作为 correctness reference |
| eta-level correction | `CG_IRLS_CROSS` | 与 RS 结构兼容、便于平滑项和大样本扩展的交叉导数修正路径 | 先实验，达标后再默认 |
| legacy damping fallback | `CG_LEGACY_DAMPED` | 仅用于向后兼容或紧急回退 | 否，必须在 warning / docs 中标注 |

### 2.2 API 约定

短期建议：

```python
model = gamlss("y ~ x", family=NO(), data=data, method="CG")
```

必须稳定路由到一个“包含交叉导数”的实现。如果存在回退到非交叉导数路径的情况，必须满足：

- 发出 `RuntimeWarning` 或写入模型诊断字段；
- `model.additional_slots["cg_backend"]` 记录实际 backend；
- `model.additional_slots["cg_cross_derivatives"]` 记录 `"full_hessian"`、`"eta_correction"` 或 `"disabled_fallback"`；
- 测试用例断言正常路径不会静默降级为 `"disabled_fallback"`。

## 3. 数学实现路线

### 3.1 coefficient-level full Hessian 路径

这是最适合作为 correctness oracle 的路径。

1. 将所有 estimable parameters 的 beta 系数按固定顺序展平：
   `theta = [beta_mu, beta_sigma, beta_nu, beta_tau]`。
2. 定义 scalar objective：
   `L(theta) = sum_i w_i * logpdf(y_i | params(theta)) - penalty(theta)`。
3. 使用 AD：
   - `score = grad(L)(theta)`；
   - `H = hessian(L)(theta)`；
   - `I = -H`。
4. 保留完整 block matrix，不允许把 off-diagonal blocks 置零。
5. 做数值稳定化：
   - symmetrize：`I = 0.5 * (I + I.T)`；
   - non-finite cleanup；
   - eigenvalue floor 或 Levenberg-Marquardt diagonal damping；
   - backtracking line search，要求 deviance finite 且不升高。

优点：实现清晰、交叉块可直接测试。缺点：大参数量时 Hessian 成本高，平滑项和稀疏结构需要额外优化。

### 3.2 eta-level cross-derivative correction 路径

这是更贴近传统 CG / RS 工作响应结构的路径。

对当前参数块 `k`，先计算标准 IRLS 工作响应，再加入其他参数块的修正项：

```text
cross_adjustment_k,i = sum_{j != k} H_kj,i * Delta eta_j,i
```

随后把 `cross_adjustment_k` 注入工作响应：

```text
z_k,adjusted = z_k,standard + cross_adjustment_k / W_k
```

其中 `W_k` 是与当前参数块对应的 working weight。开发时必须明确 `score`、`hessian_diag`、`link_derivative` 的定义域：

- 如果 `score` / `hessian_diag` 是对 distribution parameter `theta_k` 的导数，则需要链式法则转换到 eta 尺度；
- 如果直接用 AD 得到 `d l / d eta_k` 与 `d² l / d eta_k²`，则工作响应公式可以更简单，且更不容易发生 link derivative 方向错误；
- 推荐最终统一为 **eta 尺度的一阶和二阶导数**，减少 `d eta / d theta` 与 `d theta / d eta` 混淆。

## 4. 具体开发步骤

### Phase 0：冻结语义与诊断字段（0.3.x）

- [ ] 在模型结果中增加 `cg_backend`、`cg_cross_derivatives`、`cg_line_search_steps`、`cg_condition_number` 诊断字段。
- [ ] `method="CG"` 禁止静默调用 legacy damping fallback；如果必须回退，写入 warning 与诊断字段。
- [ ] 在 README / API docs 中把 CG 状态拆成 “full Hessian reference” 与 “IRLS cross correction experimental”。

### Phase 1：把 full Hessian 路径设为 correctness reference（0.3.x）

- [ ] 为 `fit_cg(..., return_fisher=True)` 增加 block extraction helper，例如 `extract_information_blocks(fisher, param_slices)`。
- [ ] 新增测试：异方差 Normal 模型中 `mu/sigma` cross block 必须非零。
- [ ] 新增测试：三参数或四参数 family 中至少一个 cross block 非零，并且 matrix symmetric。
- [ ] 新增测试：人为把 off-diagonal blocks 清零后，更新方向与完整 CG 更新方向不同。
- [ ] 记录小样本 benchmark，用 full Hessian 路径作为 eta-level correction 的数值参考。

### Phase 2：重写 eta-level derivative kernel（0.4.0 candidate）

- [ ] 新增 `cg_derivatives.py`，集中实现 `eta_score_hessian()` 与 `eta_cross_hessian()`。
- [ ] `eta_cross_hessian()` 使用 `jax.jacfwd(jax.grad(...))` 或 `jax.hessian(...)` 对单观测 logpdf 的 eta-vector 求 Hessian。
- [ ] 使用 `jax.vmap` 对观测维度批量化，返回形状 `(n, p_params, p_params)` 的 per-observation Hessian。
- [ ] 删除 `_compute_cross_derivatives()` 中的宽泛 `except Exception: return zeros`，改成显式失败或带诊断的受控 fallback。
- [ ] 增加 x64 gate：在统计验证测试中启用 `JAX_ENABLE_X64=1`，避免 float32 导致 Hessian 数值误判。

### Phase 3：统一 CG outer loop（0.4.0 candidate）

- [ ] 让 `cg_outer_step()` 接收统一 derivative bundle，而不是在每个参数块内重复调用 cross derivative。
- [ ] 固定参数更新顺序，并记录每轮 `Delta eta`。
- [ ] 对每轮 outer update 加 backtracking line search；如果 deviance 上升，缩小全局 step，而不是只缩小单参数 step。
- [ ] 支持 observation weights 与 offsets，测试覆盖 intercept-only、单协变量、多参数公式。
- [ ] 对平滑项先采用 dense penalty matrix reference，再评估 sparse / low-rank 优化。

### Phase 4：验证、benchmark 与文案切换

- [ ] 对 `NO`、`GA`、`BCCG`、`SHASH`、`BCT` 建立最小 CG 验证矩阵。
- [ ] 比较 `RS`、`CG_FULL_HESSIAN`、`CG_IRLS_CROSS` 的 deviance trajectory、迭代次数、最终参数差异。
- [ ] 生成 benchmark artifact，至少包含硬件、JAX backend、x64 状态、样本量、参数量、family、formula。
- [ ] 只有当 `CG_IRLS_CROSS` 与 `CG_FULL_HESSIAN` 在参考矩阵上达标后，才能把 README 中的默认 CG 文案切换为“production-ready”。

## 5. 测试门禁

最低测试集合如下：

```text
tests/test_cg_cross_derivatives.py
  - test_no_mu_sigma_cross_block_nonzero
  - test_full_hessian_keeps_off_diagonal_blocks
  - test_zeroing_cross_blocks_changes_update_direction
  - test_eta_cross_hessian_shape_and_symmetry
  - test_cg_method_records_backend_diagnostics
```

数值容忍建议：

- symmetry：`max_abs(H - H.T) < 1e-8`（x64）或 `< 1e-5`（float32 smoke）；
- non-zero cross block：`norm(block) > 1e-8`（x64）；
- deviance monotonicity：每次 accepted update 后 deviance 不增加，允许 `1e-8` 级别浮点误差。

## 6. 风险与规避

| 风险 | 影响 | 规避 |
|---|---|---|
| Hessian indefinite | Newton step 发散 | eigenvalue floor、LM damping、line search |
| link derivative 方向混淆 | 更新方向错误 | 内部统一使用 eta 尺度 derivative bundle |
| 交叉导数失败后静默置零 | 再次“名实不符” | 禁止 silent zero fallback，必须 warning + diagnostics |
| full Hessian 内存过大 | 大模型不可用 | full Hessian 作为 reference；大模型走 eta-level / block sparse |
| float32 Hessian 不稳定 | 测试误判 | release validation 使用 `JAX_ENABLE_X64=1` |

## 7. 完成定义

CG 交叉导数开发完成必须同时满足：

1. `method="CG"` 默认路径包含 cross derivatives，且模型对象可审计实际 backend。
2. 至少一个公开测试证明 cross block 非零且参与更新方向。
3. 正常路径不得静默退化为 RS / damping fallback。
4. README、API 文档、CHANGELOG 使用同一套术语。
5. benchmark artifact 证明 CG 在目标问题上至少具有正确性优势；性能优势如果不能稳定复现，不得对外宣称。
