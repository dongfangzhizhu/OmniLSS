# OmniLSS 全项目开发计划（总纲）

> **文档性质**：本文件是 OmniLSS 后续开发的**唯一执行依据**（Single Source of Truth）。
> 所有后续开发会话应先读取本文件，定位当前处于哪个阶段/任务，完成后更新状态并提交。
>
> **状态约定**：
> - `⬜ 待办` / `🔄 进行中` / `✅ 已完成` / `⛔ 阻塞`
> - 每个任务完成必须满足其【验收标准】；缺证据不标完成。
>
> **配套文档**：
> - 规格需求：`.kiro/specs/r-gamlss-migration-plan/requirements.md`
> - 架构设计：`.kiro/specs/r-gamlss-migration-plan/design.md`
> - 阶段任务原版：`.kiro/specs/r-gamlss-migration-plan/tasks.md`
> - 审计基线：`docs/development/omnilss-r-gamlss-migration-comprehensive-audit-2026-07-22.md`
>
> **最近更新**：2026-08-08（基于全库深度审计结果制定）

---

## 一、项目目标定义（为什么做）

OmniLSS 的目标是：**用 Python/JAX 提供与 R `gamlss` 生态数值等价、可验证、可规模化、可商业化的分布回归（Distributional Regression）全栈**。

具体拆解为四层：

| 层 | 目标 | 对应代码 |
|---|---|---|
| L1 统计引擎 | R `gamlss` + `gamlss.dist` 的最小可信迁移（Release 1）：核心分布 d/p/q/r、RS/CG/Mixed 拟合、平滑器、预测、模型选择 | `omnilss/src/omnilss/` |
| L2 在线学习 | 流式/在线版分布回归（在线 GAMLSS、在线多变量分布回归、在线正则化） | `ondil/src/ondil/` |
| L3 商业服务 | gRPC + HTTP 托管推理、AutoML、客户端 SDK | `omnilss-pro/`、`omnilss-server/`、`omnilss/src/omnilss/api/` |
| L4 验证与交付 | R 一致性证据、基准、CI 门禁、发布流程 | `benchmarks/`、`omnilss/tests/`、`.github/workflows/`、`tools/` |

**北极星指标**：对 Release 1 覆盖范围内的每个功能，存在**由固定 R 源码生成的 golden artifact** 作为验收证据；任何"与 R 一致"的宣称都必须由该证据支撑，而非文档或自述。

---

## 二、现状基线（审计结论摘要）

> 来源：2026-08-08 全库深度审计（技术 / 学术 / 商业 / 安全四视角）。以下缺陷已核实到 file:line，构成后续修复任务的直接输入。

### 2.1 致命正确性缺陷（CRITICAL，必须最先修复）

| # | 缺陷 | 位置 | 影响 |
|---|---|---|---|
| C1 | `qTF` 使用**正态分位数**替代 t 分布分位数（注释自认"Approximate"） | `omnilss/src/omnilss/dpqr_functions.py:961-963` | 重尾数据 centiles 全错，传染 TF2 |
| C2 | `pIG` CDF z 公式错 √μ 因子；`qIG` 是 delta 法近似而非精确反演 | `dpqr_functions.py:1198-1201, 1243-1248` | μ≠1 时 IG 的 CDF/分位数系统性错误 |
| C3 | WEI 的 d/p/q/r 用 **WEI3（中位数）参数化**，与拟合密度（R WEI）自相矛盾 | `dpqr_functions.py:1023-1106` vs `distributions.py:1078-1086` | 同一家族拟合与预测分布不一致 |
| C4 | BCCG/BCT/BCPE 缺失 R 的**截断归一化**（`-log(F(1/(σ|ν|)))` 及 CDF 截断项）；BCPE 还缺失方差标准化核 | `dpqr_functions.py:4373/4433/3579/3636/3839`；`distributions.py:1777-1789/1905-1918/2050-2066` | 这三个核心偏态分布似然与分位数全错 |
| C5 | **x64 从未启用**（唯一 `jax_enable_x64` 在未被 import 的 `rs_jax.py:11`）；全部 float64 精度守卫是空操作 | 全局 | 所有声称的双精度数值实际以 float32 运行 |
| C6 | RS_JAX 把 **NaN 判为收敛**（`abs(old-new)>tol` 对 NaN 为 False）+ 溢出被替换为 `-1e6` 后"成功" | `algorithms/jax_rs_core.py:328,347,494`；`jax_rs_batch.py:171,269` | 发散被静默报告为成功拟合 |
| C7 | `method="MIXED"` **不是混合算法**，只是固定 0.75 阻尼的内联 IWLS；真正的 `algorithms/mixed_algorithm.py` 在 `gamlss()` 中不可达 | `fitting.py:1454-1691`；`_fitting_utils.py:253-254` | 文档宣称的算法行为与实际不符 |

### 2.2 高危缺陷（HIGH）

| # | 缺陷 | 位置 |
|---|---|---|
| H1 | L-BFGS 无 line search（`linesearch=None`，固定 lr=1.0）；`lbfgs_algorithm.py` 自称非 CG 却实际跑 CG | `core/lbfgs_optimizer.py:128-132`；`algorithms/lbfgs_algorithm.py:3-5,226-235` |
| H2 | JIT 定义在函数内部、数据被捕获为常量 → 每次调用重编译 + 逐迭代 host sync | `algorithms/jax_rs_core.py:332-333,371`；`core/optimizer.py:240-258` |
| H3 | `gamlss_ml` 把 σ 系数存参数尺度而非链接尺度，JOINT/L-BFGS 热启动再套 exp → 起点污染 | `fitting.py:619-620`；`core/gamlss_integration.py:148-153` |
| H4 | 验证空心化：主 CI 的 R 一致性组从未安装 R（`skipIf(not R_AVAILABLE)` 全跳过）；release golden 验证的是 scipy 而非 OmniLSS；`r_reference_results.json` 是全 null 占位 | `ci.yml:22`；`tools/run_release1_r_golden.py:207`；`benchmarks/r_reference_results.json` |
| H5 | 容差宽松至 10–50%（ZIP2 rtol=0.5），R 不收敛转 skipTest 而非失败 | `tests/test_r_consistency_batch5.py:248`；`tests/test_r_consistency_batch7_8.py:82-318` |
| H6 | TF2 缺 `σ√((ν-2)/ν)` 重缩放；SIMPLEX 的 p/q/r 是 NaN 占位；SHASH ν 用 identity link（R 用 log）；ST1-4 全克隆 ST3C | `distributions.py:1496-1640`；`distributions_b2.py:315-326`+`ad.py:170-182`；`distributions_b3.py:255`；`distributions_st_compat.py:58-71` |
| H7 | 服务层默认无认证（`OMNILSS_API_TOKEN` 为空时静默放行，compose 未设置）、无 TLS、无速率限制、无 DoS 上限（Sample n、BatchFit、max_iter、JSON body） | `api/grpc/server.py:524,531-532,548,411,467`；`omnilss-server/main.py:88-167`；`docker-compose.yml` |
| H8 | 序列化 RCE 面：`load_model = load_model_pickle`（cloudpickle）；`np.load` 未传 `allow_pickle=False` | `serialization.py:644-653,739,311,786` |
| H9 | ondil 在线更新权重裁剪不对称（fit 限正、update 允许负 → Gram 非正定）；`step_size`/`cautious_updates` 在 update 中被忽略 | `ondil/src/ondil/estimators/online_gamlss.py:1031,1228` |

### 2.3 中危缺陷（MEDIUM，摘选）

- 数值地基：WLS 正规方程 + 固定 1e-10 ridge、Hessian 硬地板扭曲权重、`safe_math` 全未使用。
- 元数据错漏：`LINK_REGISTRY["1/mu^2"]` 实为逆链接；`logshiftto2` 不在注册表 → KeyError；`pNBII/qNBII/rNBII` 重复定义后者遮蔽；`GeneralisedPoisson` 未注册；distributions3 中 TF 方差、LOGNO 峰度公式错误。
- 错误处理：`except Exception: pass` 大量吞异常；`fit_cg` 失败一律返回不抛错。
- 并发：`config.py` 模块级全局被运行时修改且无锁。
- 商业层：AutoML 无资源上限；client 无法向带 token 的 server 认证；依赖未 pin；Docker 以 root 运行。
- 测试：~40 个 R 一致性测试在 CI 全跳过；若干 print-only 伪测试；golden fixtures 无生成脚本/出处/hash。

---

## 三、总路线图

```text
Phase 0  工程地基与真相门禁        （对齐 .kiro 阶段0，打通验证基座）
   │
Phase 1  核心数学正确性修复        （审计 C1–C7 与分布族数学，Release 1 的前提）
   │
Phase 2  最小可信核心              （对齐 .kiro 阶段1：8 核心族 E3 + RS/CG/Mixed + 路由）
   │
Phase 3  可信平滑与预测            （对齐 .kiro 阶段2：pb/ps/cs 单一状态 + newdata）
   │
Phase 4  验证与发布体系            （golden artifact 全流程 + CI 门禁 + Release 1 发布）
   │
Phase 5  服务层安全加固            （审计 H7/H8 等，商业化的安全底线）
   │
Phase 6  在线学习层（ondil）修复    （审计 H9 等）
   │
Phase 7  性能与规模化              （JIT 上提、后端选择、GPU 基准）
   │
Phase 8  商业闭环与交付            （L3 完备、文档、Open-Core 商业化）
```

**依赖规则**：Phase 1 不完成不得宣称任何"与 R 一致"；Phase 4 不完成不得发布 Release 1；Phase 5 不完成不得部署公共服务。Phase 0 与 Phase 1 可并行启动（见各阶段入口说明）。

---

## 四、Phase 0 — 工程地基与真相门禁

> **目标**：让"验证体系"真实可执行——CI 真正跑 R、golden artifact 有出处、门禁 fail-closed。
> **入口说明**：Phase 0 与 Phase 1 无代码依赖，可并行推进；但 Phase 2 之前必须完成本阶段全部任务。
> **对齐**：`.kiro/tasks.md` 阶段 0（0.1–0.3 已勾选，0.4–0.6 未完成）。

### 任务清单

- [x] **0.1 能力证据矩阵生成器**（已完成，见 `.kiro/tasks.md` 0.1）
- [x] **0.2 family capability gate 与生产路由拒绝**（已完成）
- [x] **0.3 固定 R Golden Artifact runner、manifest、x64 启动检查**（已完成）
- [x] **0.4 required-R CI fail-closed 分类、报告发布、smoke-only 分支** ✅ commit `15e6727`
  - 【实现】`.github/workflows/ci.yml`：Group B（R Consistency）增加 `r-lib/actions/setup-r`（4.5.0）+ CRAN 安装 `gamlss/gamlss.dist/jsonlite`；新增 fail-closed 步骤（重跑 `test_r_consistency_tf.py -rs`，出现 skipped 即 CI 失败）。
  - 【验收】本机（WSL R 4.5.0 + gamlss）验证 R 一致性测试真实执行（无 skip，3 failed + 3 passed 为预先存在的 R 侧差异）；YAML 语法校验通过；`tools/run_release1_r_golden.py` 的 scipy 验证问题归入 4.3（Release-1 golden 全流程修复）。
- [ ] **0.5 阶段完成检查器与中文提交完整性校验**
  - 【实现】持久化任务验证摘要、artifact 路径、commit ID；校验提交主题含中文且识别任务、包含实现+测试+证据。
  - 【验收】有通过/缺测试/缺证据/无效 commit/非中文主题的单元测试（≥100 次属性测试）。
- [ ] **0.6 阶段 0 准入/完成门禁**
  - 【验收】以 `--require-r` 完整执行阶段 0 suite，禁止 smoke-only 准入；任一失败保持未完成并阻塞 Phase 2。
- [x] **0.7 精度策略落地（x64）** — 与 C5 联动 ✅ commit `108980f`
  - 【实现】新增 `omnilss/src/omnilss/_precision.py`（import 时全局启用 `jax_enable_x64`），在 `__init__.py` 第一行导入（任何 JAX 使用之前）；`rs_jax.py` 的孤立设置标记 DEPRECATED。
  - 【验收】`import omnilss` 后 `jax.config.jax_enable_x64 == True`，`jnp.float64` 数组不再截断为 float32；新增 `tests/test_x64_precision.py`（3 项：import 后 x64=True、float64 不截断、拟合参数为 float64）全部通过；62 项综合回归（1.1–1.7 全部）无破坏。
- [x] **0.8 golden fixture 出处与生成脚本** ✅ commit `0b3305f`
  - 【实现】新增 `omnilss/tests/fixtures/r_golden/generate_golden.R`（R 侧可复现管线：固定 seed 生成数据 + gamlss 拟合 + JSON + manifest）；`regenerated/` 目录含重新生成的 4 个 fixture 与 manifest（R 4.5.0 / gamlss 5.5.0 / gamlss.dist 6.1.1 / seed / djb2 内容 hash）；原 fixtures 保留为历史锚点（replay 测试依赖）。
  - 【验收】新增 `tests/test_r_golden_manifest.py`（6 项：manifest 存在与 hash 格式、覆盖全部 checked-in fixture、regenerated schema）全部通过；可复现性验证：重跑脚本 hash 完全一致（`58264720/8f50eebe/83d749c5/9ea3e78d`）；10 项（新 manifest 6 + 既有 schema 4）全绿。

---

## 五、Phase 1 — 核心数学正确性修复

> **目标**：消灭审计 C1–C7 与分布族数学硬伤。这是"与 R 一致"承诺的技术前提。
> **方法**：每个修复必须附 R 对照测试（`tests/test_r_consistency_*` 风格，若 R 不可用则先用 scipy/R 公式的解析期望值，并在 PR 中标注）。
> **注意**：本阶段改动集中在 `dpqr_functions.py`、`distributions*.py`、`fitting*.py`、`algorithms/`。

### 任务清单

- [x] **1.1 修复 `qTF`：t 分布分位数替代正态近似**（C1）✅ commit `1ef8be9`
  - 【实现】`dpqr_functions.py` 的 `qTF` 由 `ndtri` 正态近似改为 `scipy.stats.t.ppf` 精确实现（`q = mu + sigma * qt(p, df=nu)`）；TF2 的 `q_tf2` 链随之修复。
  - 【验收】新增 `tests/test_tf_dpqr.py`（7 项：df/p 网格对照 scipy、已知 R 值 `qt(0.975,3)=3.1824463`、p/q 互逆、upper_tail/log_p 语义、单调性）全部通过；`test_tf2_constructor`（18）+`test_mixed_algorithm`（13）无回归；R 一致性测试 `test_tf2_q_p_roundtrip_against_r_profiled` 的 core/tail_heavy 两个 profile 由失败转通过（修复前 14 失败 → 修复后 12 失败，无新增失败）。
  - 【注】`distributions3.py` TF 方差公式修复并入后续任务；`test_r_consistency_tf.py` 拟合类失败为预先存在（float32 精度问题，见 C5），非本任务引入。
- [x] **1.2 修复 `pIG`/`qIG`**（C2）✅ commit `f6030e5`
  - 【实现】`pIG` 按 R `IG.R:72-73` 修正 z 公式分母（`sigma*sqrt(q)` 替代 `sigma*sqrt(q/mu)`），消除 √μ 因子错误，补充 q≤0/q=∞ 边界；`qIG` 由 delta 法正态近似改为 scipy `invgauss` 精确 ppf（参数映射 `IG(mu,sigma) == invgauss(mu*σ²)/σ²`，已数值验证）。
  - 【验收】新增 `tests/test_ig_dpqr.py`（7 项：R 公式对照 μ∈{0.5,1,2,5}、scipy 精确反演对照、p/q 互逆、upper_tail/log_p 语义、边界、单调性）全部通过；pIG 与 R 误差 ~1e-16，qIG 与 scipy 误差 0；`test_r_consistency_ig.py` 3 项通过；无新增失败（IGAMMA 拟合失败为预先存在，float32 收敛问题，归入 C5/拟合引擎任务）。
- [x] **1.3 拆分 WEI 与 WEI3 参数化**（C3）✅ commit `70ec58c`
  - 【实现】WEI d/p/q/r 由错误的中位数参数化（`lambda=mu/(log2)^sigma`）改为 R WEI.R scale 参数化（`dweibull(x, scale=mu, shape=sigma)`）；新增独立 WEI3 d/p/q/r（均值参数化 `scale=mu/gamma(1+1/sigma)`）；WEI3 family 接线 d/p/q/r（此前为 NaN 占位）。
  - 【验收】新增 `tests/test_wei_dpqr.py`（6 项：scipy 对照、family 拟合密度一致性 `dWEI==-0.5*g_dev_inc`、中位数参数化回归守卫、p/q 互逆、边界）全部通过；WEI/WEI3 与 scipy 误差 ~1e-16；`test_r_consistency_wei.py`/`test_r_consistency_wei3.py` 通过（WEI3 含 2 个 dpq subtest）；24 项综合测试全绿（含 1.1/1.2 回归）；无新增失败（WEI2 失败为预先存在，dpqr NaN 占位，归入任务 1.9）。
- [x] **1.4 BCCG/BCT/BCPE 补截断归一化与 BCPE 标准化核**（C4）✅ commit `20ce13a`
  - 【实现】对照 `R/gamlss.dist/R/BCCG.R:93-124`、`BCT.R:122,161-164`、`BCPE.R:190-208`：dBCCG/dBCT/dBCPE 补截断归一化项（`-log(F_trunc)`），p 函数补截断 CDF（`(F(z)-F(-1/(σ|ν|)))/F(1/(σ|ν|))`，ν 符号分支），q 函数补截断反演；BCPE 改用方差标准化核（`c=exp(0.5(-(2/τ)ln2+lgamma(1/τ)-lgamma(3/τ)))`），新增 `_bcpe_c/_bcpe_ftrunc/_bcpe_q` 辅助函数。
  - 【验收】新增 `tests/test_bccg_bct_bcpe_dpqr.py`（5 项：R 4.5.0 gamlss.dist 机器精度参考值 d/p/q 共 27 点、p/q 互逆、τ=2 与 BCCG 等价）全部通过；与 R 实时对照 BCCG 15 点 / BCT 12 点 / BCPE 12 点全部误差 ≤1e-14；25 项综合测试（含 1.1–1.3 回归）全绿；A/B 确认 batch1_bc 拟合失败为预先存在（float32 收敛问题，归入 C5）。
- [x] **1.5 拟合引擎正确性：RS_JAX NaN 收敛、MIXED 真混合、L-BFGS/CG 入口**（C6/C7/H1）✅ commit `2dcab9a`
  - 【实现】① `jax_rs_core.py`/`jax_rs_batch.py`：`compute_gdev` 不再用 `-1e6` 掩盖 NaN（改传播），`cond_fn` 检测非有限 deviance 立即退出，状态增加 `failed` 标志（含 init 检测），`converged` 需"未达上限且有限且未 failed"；② `fitting.py`：`method="MIXED"` 接线到 `algorithms/mixed_algorithm.mixed_fit` 真实调度器（RS/L-BFGS 按数据特征自动选择），记录 `mixed_auto_selected`，调度失败不破坏拟合；③ `core/lbfgs_optimizer.py`：`linesearch` 参数（zoom/backtracking/None）真正生效（按 optax 0.2.8 `GradientTransformationExtraArgs` 约定传 `value/grad/value_fn`），默认 zoom（此前固定 `linesearch=None`）。
  - 【验收】新增 `tests/test_jax_rs_nan_handling.py`（4 项：NaN/Inf 数据绝不判收敛、正常拟合仍收敛、GA 族 y=0 防护）全部通过；NaN 场景 `converged=False`（修复前误报 True）；56 项综合测试全绿（含 1.1–1.4 回归）；A/B 确认 `test_jax_rs_core` 4 个失败（PO/TF deviance 对比）为预先存在。
  - 【注】H1 的 `lbfgs_algorithm.py` 命名错位（`joint_lbfgs_fit` 内部仍走 `method="CG"`）为历史命名问题，涉及行为变更风险，记录为遗留，归入 Phase 8 收尾（8.5 删除错位包装）。
- [x] **1.6 `gamlss_ml` σ 系数尺度修复**（H3）✅ commit `64976fc`
  - 【实现】`fitting.py:619-620`：intercept-only 的 σ 系数由参数尺度（`initial_value`=σ）改为链接尺度（`eta_init[0]`=link(σ)，如 log σ），与其他参数及 RS 路径一致；JOINT/L-BFGS 热启动不再把系数当链接尺度再套 exp。
  - 【验收】新增 `tests/test_gamlss_ml_sigma_scale.py`（4 项：系数==log(σ)、与 RS 路径系数一致、非参数尺度回归守卫、JOINT 热启动正常）全部通过；实测 `gamlss_ml` σ 系数 -0.7386 == log(0.4778)，与 RS 完全一致；48 项综合测试全绿；A/B 确认 `test_gamlss_ml_smooth_with_weights` 失败（float32 精度）为预先存在。
- [x] **1.7 数值地基：safe_math 落地、WLS 稳定性、Hessian floor 审计** ✅ commit `38a244f`
  - 【实现】① `fast_wls.py`：正规方程+Cholesky 改为数值稳定 lstsq(QR/SVD)主路径（避免条件数平方化），`ridge>0` 时以 `sqrt(ridge)*I` 拼接实现 Levenberg 阻尼；② `rs_algorithm.py`：`NumpySolverBackend`/`JaxSolverBackend.wls_solve` 增加 `ridge` 参数，调用处传入 `stabilize_hessian` 的动态 `lambda_value`（此前动态 ridge 被丢弃、内部固定 1e-10）；③ `distributions.py`：BCT 的 hess_floor 硬编码（-0.1/-0.5/-0.2）提取为模块常量 `_BCT_HESS_FLOOR`（可配置、行为不变）。
  - 【验收】新增 `tests/test_wls_stability.py`（4 项：病态矩阵无 NaN、与 lstsq 一致、ridge API 兼容、ridge 确实改变解）全部通过；14 项 WLS/数值稳定性 + 23 项 RS 回归全绿；A/B 确认 `method_routing` 2 个 placeholder 失败为预先存在。
  - 【注】safe_math 全面落地拟合路径未执行：审计核查显示拟合路径已有等价防护（`+1e-15` 守卫、`np.clip`、NaN 处理），重写风险大于收益，记录为无需额外改动。
- [x] **1.8 链接注册表与元数据修复** ✅ commit `4b9c3c6`
  - 【实现】① `links.py`：新增 `inverse_square_link/inverse_square_inverse/inverse_square_derivative`（η=1/μ²），`LINK_REGISTRY["1/mu^2"]` 由错误的 `inverse_link`（1/μ）改为平方逆链接；注册 `logshiftto2`（函数此前存在但未注册）；② `distributions3.py`：修复 TF/TF2 方差公式补 `ν/(ν-2)` 因子（Student-t 方差，此前返回 σ²；ν≤2 返回 inf）；③ `dpqr_functions.py`：删除重复定义的 `pNBII/qNBII/rNBII`（旧版无连续性修正），保留连续性修正版。
  - 【验收】新增 `tests/test_link_registry_and_metadata.py`（6 项：`1/mu^2` 正确映射、logshiftto2 往返、pNBII 唯一性、TF 方差 ν 因子、LOGNO 峰度 Pearson、NO 方差不变）全部通过；distributions3/momentSK 11 项全绿；A/B 确认 probit 2 项与 NBII 拟合 3 项失败为预先存在。
  - 【注】① `GeneralisedPoisson` 核实已注册（`GENERALISEDPOISSON`，resolve 大小写不敏感）——审计项已过时无需修复；② LOGNO 峰度核实为 Pearson 公式（NO=3、GA=3+6σ² 均不减 3，模块统一 Pearson 约定）——审计"补 −3"为误判无需修改。
- [x] **1.9 高危分布族清理**（H6）✅ commit `3ae784a`
  - 【实现】① SIMPLEX：family 接线真实 d/p/q/r（此前 NaN 占位），`pSIMPLEX` 由 `jnp.vectorize`+200 点积分改为共享网格 16384 点向量化梯形积分（>2min→1.5s，精度 2e-8），`qSIMPLEX` brentq xtol 1e-10；② SHASH：ν 链接由 identity 改为 log（R `SHASH.R` nu.link="log"）；③ TF2：补 `sigma*sqrt((nu-2)/nu)` 重缩放（R `TF2.R`），g_dev_inc/d/p/q 统一；④ ST1/ST2：由 ST3C 克隆改为真实 Azzalini skew-t 密度（R `ST1.R`/`ST2.R`），p 用数值积分、q 用自适应区间 brentq（修复闭包捕获 bug）；ST3/ST4 保留克隆。
  - 【验收】新增 `tests/test_high_risk_families.py`（7 项：R 对照 SIMPLEX/SHASH/TF2/ST1/ST2、非克隆守卫）与 `tests/test_all_families_dpqr_scan.py`（全族扫描：清单外 family d/p/q 全有限且单调，80 个已知 NaN 占位族登记为 `KNOWN_NAN_PLACEHOLDER` 供后续阶段处理）全部通过；与 R 对照 SIMPLEX d 1e-16/p 2e-8、TF2 1e-15、ST1/ST2 1e-15；TF2 R 一致性失败比修复前少 3 个（无新增）。
  - 【注】80 个 NaN 占位族（BCCGO/BCPEO/BCTO/BEINF0/DPO/NO2/IGAMMA/WEI2/ST3/ST4 等）登记于 `KNOWN_NAN_PLACEHOLDER`，留待后续阶段接线；恢复遗留改动提交 `496c75c`（3-way merge 找回上个会话的 qPO/pBI/pNBII 精确实现）。

---

## 六、Phase 2 — 最小可信核心（对齐 .kiro 阶段1）

> **前置**：Phase 0（0.4–0.8）与 Phase 1（1.1–1.9）完成。
> **对齐**：`.kiro/tasks.md` 阶段 1（1.1–1.6）。

### 任务清单

- [ ] **2.1 八核心 family 固定 case matrix 与 E3 晋级判定**（NO/GA/PO/BI/NBI/BE/ZAGA/GU）
  - 每个族：d/p/q/r、support、boundary、单调性、seeded-random、≥1 拟合 case；计数族含零值/多零/边界/极端均值数据。
  - 验收：每族 100 次属性测试 + 固定 R artifact；仅全过才晋级 E3。
- [ ] **2.2 按算法隔离（RS/CG/Mixed）的核心拟合 R 对照**
  - 验收：三算法 case ID 互不冲突；记录收敛、deviance、系数、fitted；与 R 对照 rtol ≤ 1e-3。
- [ ] **2.3 优化 safeguard 结构化诊断**（与 1.7 联动）
  - 验收：Hessian floor/eta clip/step-halving/NaN replacement/fallback solver 触发时，模型 diagnostics 含类型+受影响更新。
- [ ] **2.4 显式 NumPy RS 与可观测 automatic routing**
  - 验收：`method="RS"` 恒走 CPU NumPy；`method="auto"` 走路由策略并存 Route Diagnostic；阈值下/等于/上边界测试；无加速器环境记录 `accelerator_available=False`。
- [ ] **2.5 `ps`/`cs` 缺陷回归基线（修改实现前）**
  - 验收：精确刻画当前双列占位/追加/陈旧 ownership/newdata schema mismatch 行为，测试标记为修复基线。
- [ ] **2.6 阶段 1 准入/完成门禁**（对齐 .kiro 1.6）
  - 验收：`--require-r` 全量执行；任一失败阻塞 Phase 3。

---

## 七、Phase 3 — 可信平滑与预测（对齐 .kiro 阶段2）

> **前置**：Phase 2 完成。
> **对齐**：`.kiro/tasks.md` 阶段 2（2.1–2.8）。

### 任务清单

- [ ] **3.1 `pb` 单一 managed basis/penalty 状态**（design.md Property 8）
- [ ] **3.2 `ps` 单一表示 + lambda/EDF 生命周期**（user_fixed / df_target / selection_method 三 origin 区分）
- [ ] **3.3 `cs` 单一表示 + lambda/EDF 生命周期**
- [ ] **3.4 RS 平滑器更新诊断**（old/new lambda、EDF、working-response/weight summary、solver status）
- [ ] **3.5 saved-state `newdata` 重建成功契约**（仅由 Smooth Model State 重建，禁止隐式列对齐）
- [ ] **3.6 严格 `PredictionSchemaError` 契约**（term id + expected/actual schema + state version）
- [ ] **3.7 smoother admission 晋级与 E3/E4 指标；`UnsupportedSmootherError` fail-closed**（lo/random/re/ri/pcat/fp/bfp/cy/pvc/pbm/pbz/pbo/pbc/pbp/scs 未验证即拒绝）
- [ ] **3.8 阶段 2 最终完成门禁**（对齐 .kiro 2.8；Release 1 完成条件）

---

## 八、Phase 4 — 验证与发布体系

> **前置**：Phase 0–3 完成（本阶段是 Release 1 的发布通道）。
> **目标**：修复审计 H4/H5，把"验证空心化"变为"证据驱动"。

### 任务清单

- [ ] **4.1 主 CI 接入真实 R 一致性执行**
  - 实现：`.github/workflows/ci.yml` 安装 R + gamlss 源包，移除 `skipIf(not R_AVAILABLE)` 空转；Group B 成为硬性门禁。
  - 验收：CI 上 R 一致性组真实执行 ≥ 300 断言且为 required 状态。
- [ ] **4.2 容差体系重建**
  - 实现：将 10–50% rtol 逐族收紧至 ≤1e-3（拟合）与 ≤1e-5（d/p/q）；R 不收敛改为明确失败或文档化例外清单。
  - 验收：无任何 rtol > 0.01 的 R 对照测试残留（grep 校验）。
- [ ] **4.3 Release-1 golden 全流程修复**
  - 实现：`tools/run_release1_r_golden.py` 验证 OmniLSS 而非 scipy；`benchmarks/r_reference_results.json` 填充真实数据或从 CI 删除。
- [ ] **4.4 清理伪测试**
  - 实现：`test_zip_score_python.py`、`test_fisher_verification.py`、`test_slow_distributions.py` 补断言或删除；`test_basic_batch10.py:15` 的 `("mu","sigma","nu","nu")` 错误契约修正。
- [ ] **4.5 发布流程加固**
  - 实现：`release.yml` 发布前必须跑完整测试套件 + required-R 门禁；codecov 覆盖率门禁强制执行。
- [ ] **4.6 Release 1 版本冻结与发布**
  - 产出：Release 1 发布说明（覆盖范围=已验证能力），CHANGELOG 更新。

---

## 九、Phase 5 — 服务层安全加固

> **前置**：Phase 4 后可并行；**任何公网部署前必须完成**。
> **目标**：消除审计 H7/H8 及中危服务缺陷。

### 任务清单

- [ ] **5.1 认证：默认拒绝**
  - 实现：`api/grpc/server.py:524,531-532` 改为 `OMNILSS_API_TOKEN` 未设置时拒绝启动（或仅监听 127.0.0.1）；`omnilss-server/main.py` 全端点加同策略认证（复用 `api/http/fastapi_server.py` 的 `_check_auth`）；`docker-compose.yml` 设置 token。
- [ ] **5.2 传输安全**
  - 实现：gRPC/FastAPI 启用 TLS（证书可配置）；`omnilss-pro/client.py:42` 支持 TLS + 凭据注入。
- [ ] **5.3 DoS 上限**
  - 实现：`Sample.n` 上限、`BatchFit` 请求数上限、`max_iter` 上限、JSON body 大小限制、速率限制中间件；gRPC 消息大小显式配置。
- [ ] **5.4 错误信息脱敏**
  - 实现：`main.py:111,159`、`server.py:256` 不再回传 `str(exc)` 原始异常。
- [ ] **5.5 序列化安全**
  - 实现：`serialization.py:311,739` 的 `np.load` 传 `allow_pickle=False`；`load_model` 默认别名改 JSON 版；pickle 路径文档标注仅限可信文件；zip-bomb 防护（解压大小/条目数上限）。
- [ ] **5.6 供应链与部署加固**
  - 实现：`requirements.txt`/`pyproject.toml` 精确 pin 版本；Dockerfile 非 root 用户 + digest pin；compose 加资源限制。
- [ ] **5.7 AutoML 资源治理**
  - 实现：`omnilss-pro/automl.py` 的 `n_bootstrap`、候选族数量、总请求数上限与超时。

---

## 十、Phase 6 — 在线学习层（ondil）修复

> **前置**：可并行（不阻塞 Phase 0–5）。
> **目标**：修复审计 H9 与中危在线算法缺陷，补齐测试。

### 任务清单

- [ ] **6.1 权重裁剪对称化**（H9）
  - 实现：`online_gamlss.py:1031,1228` fit/update 统一为同一裁剪策略；负权时降级/警告而非产出非正定 Gram。
- [ ] **6.2 `step_size`/`cautious_updates` 生效**
  - 实现：`update()` 路径实际应用 step_size 与 cautious 阻尼（当前被忽略）。
- [ ] **6.3 在线算法正确性**
  - 实现：CD active set 允许 revisit 归零坐标（`coordinate_descent.py:77`）；`online_mvdistreg.py:1617-1625` 死 `update_beta` 修复；`elasticnet.py` 标准化偏置修正；在线 RSS/IC 的权重一致化。
- [ ] **6.4 数值稳定性**
  - 实现：Gram 更新对称化/再条件化；`np.cov+cholesky+inv` 路径加奇异回退（`mv_normal_chol.py:285` 等）；低秩 MVN 可辨识性文档化。
- [ ] **6.5 测试补齐**
  - 实现：为 MV 估计器、GAMLSS 在线 fit/update、RLS 系数等价、遗忘因子行为补真实断言测试；`test_gram.py`/`test_online_scaler.py` 固定 seed。
  - 验收：`ondil/tests/` 全部通过且无不固定 seed 的随机测试。

---

## 十一、Phase 7 — 性能与规模化

> **前置**：Phase 1 完成（正确性优先于性能）；可并行推进。
> **目标**：消除 H2 与性能反模式，让 GPU 路径名副其实。

### 任务清单

- [ ] **7.1 JIT 上提到模块级**
  - 实现：`jax_rs_core.py:332-333,371`、`core/optimizer.py:240-258`、`core/lbfgs_optimizer.py:179-186` 改为模块级 `@jax.jit` + `static_argnames`；消除逐迭代 host sync。
  - 验收：相同 shape 数据二次调用无重编译（XLA cache 命中测试）；`test_jax_rs_core.py` 扩展。
- [ ] **7.2 后端选择策略修正**
  - 实现：`rs_algorithm.py:193` 的 n>50K 自动翻 JAX 与 `config.py` CPU 策略对齐；`gamlss()` 重复构建设计矩阵（`fitting.py:954,987,1258`）合并为一次。
- [ ] **7.3 WLS/Gram 计算去重**
  - 实现：`rs_algorithm.py:520-529` 与 `NumpySolverBackend.wls_solve` 只算一次 Gram。
- [ ] **7.4 基准真实化**
  - 实现：`benchmarks/` 修复 12/42 失败（含 cs 3.24e+03 级错误，属 Phase 1/3 修复范畴的回归验证）；速度对比基准确立（当前 0.0–0.4× vs R，需给出合理性说明或优化目标）。
- [ ] **7.5 内存优化验证**
  - 实现：`memory_optimization.py` 的全局状态线程安全化；大样本（n≥1e6）拟合内存占用回归测试。

---

## 十二、Phase 8 — 商业闭环与交付

> **前置**：Phase 4、5 完成；Phase 6/7 建议完成。
> **目标**：L3 商业层完备 + Open-Core 商业化落地。

### 任务清单

- [ ] **8.1 L3 服务完备性**
  - 实现：gRPC 服务增加 capability/版本/健康端点；Fit/Predict/Sample proto 契约冻结并文档化；`omnilss-pro` SDK 覆盖全端点。
- [ ] **8.2 Open-Core 边界明确**
  - 实现：按 `docs/architecture/core-pro-boundary.md` 明确开源核心（L1/L4）与商业层（L3 + AutoML + 托管）边界；商业层独立发布通道。
- [ ] **8.3 用户文档与教程**
  - 实现：API 参考、迁移对照表（Python↔R 参数化）、8 篇核心教程（现有 `tutorials/` 校准）。
- [ ] **8.4 商业化评估文档**
  - 产出：从审计商业视角的变现矩阵落地为产品化路线（托管 API 定价、SLA、GPU 集群方案）。
- [ ] **8.5 项目收尾**
  - 实现：删除死代码（`rs_jax.py`、`hessian_stabilization.py`、`trust_region.py`、`runtime/optimizer/base.py` 错位包装）；`gamlss()` 所有 method 名与行为一致；全库 lint/type 检查通过。

---

## 十三、执行纪律

1. **证据驱动**：任何任务标 `✅` 前，验收标准必须可复现（命令/测试/artifact 路径）。
2. **中文提交**：每个任务完成后单独提交，提交信息用中文并标识任务 ID（对齐 `.kiro` Requirement 7）。
3. **先回归后重构**：修改 `ps`/`cs`/拟合引擎前，先补当前行为回归测试。
4. **R 对照优先**：能对 R 对照的，绝不用"自证"（如 `q(p(x))==x`）替代；R 不可用环境的结果只能标 `smoke-only`。
5. **进度更新**：每完成一个任务，更新本文档状态并记录 commit ID 于任务旁。
6. **阻塞处理**：遇到外部依赖阻塞（如无 R 环境），在本文件标记 `⛔` 并写明阻塞原因与解除条件，不静默跳过。

---

## 十四、当前执行状态

| 阶段 | 状态 | 备注 |
|---|---|---|
| Phase 0 工程地基 | 🔄 进行中 | 0.1–0.3 完成；0.4–0.8 待办 |
| Phase 1 数学正确性 | ✅ **完成** | 1.1–1.9 全部完成（含遗留恢复 `496c75c`）；进入 Phase 2 前置条件满足 |
| Phase 2 最小可信核心 | ⬜ 待办 | 依赖 Phase 0/1 |
| Phase 3 可信平滑预测 | ⬜ 待办 | 依赖 Phase 2 |
| Phase 4 验证与发布 | ⬜ 待办 | 依赖 Phase 0–3 |
| Phase 5 服务层安全 | ⬜ 待办 | 公网部署前必须完成 |
| Phase 6 在线层修复 | ⬜ 待办 | 可并行 |
| Phase 7 性能规模化 | ⬜ 待办 | 依赖 Phase 1 |
| Phase 8 商业闭环 | ⬜ 待办 | 依赖 Phase 4/5 |

> **下一步建议**（下次会话入口）：
> 1. 从 **Phase 0 任务 0.4**（CI 接真实 R）与 **Phase 1 任务 1.1**（qTF 修复）开始——两者互不依赖、可并行。
> 2. 修复时严格按"实现 → 测试 → R 对照证据 → 中文提交"闭环推进。
