# OmniLSS 项目长期记忆

## 回归测试：必须分块跑

`omnilss/tests/` 共 **425** 个文件。`pytest tests/ -m "not slow"` 全量单次运行实测
**超过 1h38m 未结束**（`test_consistency_families.py` 单文件 9 分钟）。

- 一律按目录/主题分块，每块控制在数分钟内。
- 前台 2 分钟超时会把慢批次打成 SIGTERM —— **看起来像失败，其实是超时**。慢批次用 `run_in_background`。
- 无 `pytest-timeout` 插件，`--timeout` 不可用。
- 沙箱吞掉 `--collect-only` 与重定向到文件的输出；看到 "0 cases" 或空输出别当结论，换直连 stderr 或 `grep` 数 `def test_`。
- 后台跑 pytest 时管道给 `grep` 常抓不到输出；用 `| tail -N` 直接取尾部。

## 覆盖工件纪律

`test_suite_sha256` 内嵌于 JSON 工件 → **任何 `tests/` 变更必须在同一 commit 内再生工件**
（顺序不能反）：

```
tools/generate_coverage_artifacts.py --date YYYY-MM-DD
tools/coverage_diff_report.py        # 确认 result: OK
```

（`generate_capability_matrix.py` 在 **`omnilss/tools/`** 下。）
门禁测试 `tests/test_symbol_coverage_tool.py` 报 "stale coverage artifacts"。

判定"既有失败 vs 本次引入"：改动已提交用 `git checkout -q <commit>~1` 造对照；未提交用 `git stash push -u`。

## 四条通用认知纪律

1. **既有用例失败，先分清是代码错还是期望错。** `test_grpc_batch_fit_service_direct` 用了非生产族 `GA`，被 fail-closed 准入正确拒绝 —— 是期望过时。若"修服务让它放行"就拆掉了安全边界。
2. **"R 没说" ≠ "可以靠常识补"。** 族级响应域表初版给 19 个族凭分布常识加了限制，核对源码后发现这些族**根本没有 `y.valid`**。假拒绝比漏检更糟。
3. **写断言前先看实际数值、也要看对通道。** D2.21 翻了 4 次，D2.23 翻 2 次，D2.24 翻 1 次（`nan != nan`），D2.25 又翻 1 次（**`family.pdf` 把 `-inf` exp 成有限的 `0.0`**，断言 `isfinite` 因错误的原因通过）。
4. **活 R 对照是唯一可信的 parity 证据。** `converged=True` 只说明算法跑完。`BE`/`BEINF` 的 `g_dev` 为负一度被判为缺陷，活 R 给出**完全相同的值**（-3.836092659932398）。

bridge 用法：`from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge`，
`RBridge().call_r_gamlss(..., family="<NAME>")`。**`family` 必须传字符串名**。
R 的 `stop()` 被包成 **`RuntimeError`** 抛出，不是返回 `{"success": False}`。

## 收敛状态：flag 与最优性是两件事

`GAMLSSModel` 有 `converged` / `termination_reason` 一等字段 + 访问器 `convergence_status()`。
词汇表（D2.2 起，**封闭**）：`relative_deviance_converged` / `max_iterations_reached` /
`non_finite_deviance`（RS+JAX-RS）；`step_rejected` / `no_progress_step_rejected`（CG）；
`cg_stage_not_converged`（MIXED）。

**`converged=True` 不等于到达极小**（D2.22）—— 它报告**步长塌缩**。BI 的 8566.07 全程
`converged=True`，实际是两周期震荡。

**D2.24 装上了仪器**：`rs_algorithm.eta_gradient_norms()` 发布
`eta_gradient_norm` / `eta_gradient_norm_by_param` 槽位 =
`max_p ||X_p' W dlogL/deta_p||_inf`，**刻意不 gate `converged`**（理由见下节）。
需要链式法则的 `dmu/deta`：natural 尺度 `dld` **跨参数不可比** —— BCPE 的
`sigma`/`tau` 是 log 链而 `nu` 是 identity，`sum(tau*dldt)=1e-9` 可与 `sum(dldm)=-21.6` 并存。

## 量级分离不是判据，"走一步能否下降"才是（D2.24，最易重犯）

**与 D2.19 的参数地板判据完全同构。** eta 梯度阈值第一版取 `1e-3`（正常族
8.3e-14~9.4e-05 vs 卡住的族 1.0、1.6e+09，看着分离干净）。用"沿梯度走一步能不能
真降 deviance"一测，**分离是假的**：

| 族 | `eta_grad` | Fisher 步增益 | |
|---|---|---|---|
| `PO` | 1.6e-03 | 1.9e-08 | 假阳性 |
| `WEI2` | 1.2e-02 | 2.9e-06 | 命中 |
| `GG` | 2.6e-01 | 8.1e-05 | 命中 |
| `LOGITNO` | 2.5e+00 | 9.1e-13 | 假阳性 |
| `BCPE` | 1.16e+00 | 9.0e-03 | 命中 |
| `LNO` | 8.8e+12 | 4.3e+12 | 命中 |

`PO` 与 `WEI2` **重叠七倍** → 任何阈值都分不开。梯度量级量的是"离驻点多远
（score 单位）"，不是"还有多少目标函数可达"。gate 版本会把已收敛的 Poisson
拟合误标成停滞。**唯一可接受的检验是问目标函数在另一侧做什么。**

## 分轴扫描 / 切片：先确认基准点是驻点（D2.19 + D2.24）

**形状相同的两个地板，性质可以完全相反** —— 读代码区分不出来：

| | R 的边界 | 边界另一侧 | 地板 |
|---|---|---|---|
| `SST.tau` `2+eps` | `m2` 带 `(tau-2)` 分母 | R 自己 `sqrt` 得 `NaN` 并 **abort** | **正当** |
| `ST3C.tau` `1+eps` | `tau.valid = all(tau > 0)` | 有定义且随 tau 变 | **不正当** |

- **唯一判据**：拿活 R 在边界另一侧求值，看 R 自己是否真有定义。
- 症状很特征：**地板把一段参数压成常函数 → 导数恒 0 → 密度关与导数关同时报相对偏差**。
- **通过非驻点的一维切片量的是轨迹，不是地形。** BCPE 曾被判为"两个不连通盆地"
  （固定 `nu`/`tau` 时 `sigma < 1.1077` 返回 `Inf`）—— 错因是那个切片取在**我们自己
  仍未收敛的迭代点**上；`tau` 从 17631 爬到 473368 时同一阈值跟着变。

## `g_dev_inc` 已经是完整 deviance，别再乘 2

`rs_algorithm` 里是 `g_dev = sum(w * g_dev_incr)`，**没有 ×2**。乘 2（deviance 定义成
`-2*logL` 时的自然反射）会让 `7.6e-16` 的 Newton 步看起来把 deviance 移动了 168
—— D2.24 据此得出过"判定全错"的**反向**结论。

## parity 对照不能把 R 的 `converged` 当基准（D2.24）

R 在 BCPE 上 deviance 122.825553739168 处报 `converged=TRUE`，且 `n.cyc`
20/30/100/300/1000 **一位不变**；逐迭代最后一步变化 0.0006 < `c_crit=0.001`。
停的原因是**绝对判据**，不是到达驻点 —— 在 R 自己的最优点上 eta 梯度
`dldm=-21.6`、`sigma*dldd=-48.0`、`dldv=-8.2`。我们至少会报 `max_iterations_reached`。
**R 的 `converged` 是被 `c_crit` 定义的事实，不是最优性证明。**

## 响应域校验（D2.4 起）

`omnilss/response_domain.py`。接入点在 `gamlss()` 内 `resolve_family` 之后、所有方法
路由之前 → 五条路径行为一致。**`gamlss_ml` 故意不加**（R 的 `gamlssML` 也不校验 ——
正确的 parity 边界）。R 校验点 `gamlss/R/gamlss-5.R:960`。

## JAX RS 批量路径的结构性风险（已登记未消除）

`jax_rs_batch` 的 vmap kernel 是 `jax_rs_core` IRLS 的**独立重写**（vmap 不能在 traced
值上调 core），已因此出过一次崩溃缺陷。两处裁剪逻辑一处用 Python float、一处用 traced
数组；改为**扩展 parity 断言到全部 7 个字段**锁漂移。

## 族 parity 扫描器（D2.9 起）—— 排查族缺陷的默认入口

`tools/rs_family_parity_scan.py` + `tools/r_parity_batch.R`。三关：`dpqr`（d/p/q 对活 R）、
`deriv`（score 对中心有限差分）、`rsdev`（RS deviance 对活 R）。146 族单进程 10–30 分钟，
**必须后台跑**。`--timeout`（默认 180s）是**必需的**：Windows 没有 `signal.alarm`，
逐族 `try/except` 挡得住抛异常挡不住不返回。

**六条踩过的坑**：

1. **deriv 关对"响应约定错位"是盲的**。FD 与解析导数跑同一个错误变换，自洽但全错（BI 把计数当比例，deriv 全过）。这类缺陷只有 `dpqr` 能抓。
2. **探测网格从族对象推导，不用白名单**：`type` 定支撑、`links` 定参数区间、`fixed_parameters` 定是否补 `bd`。支撑判定用**相对**阈值 `d >= 1e-6 * max(d)`（LOGITNO 在 y>1 处返回极小正数，`d > 0` 会让越界网格满分通过）。
3. **无法探测记 `skip` 不记 `fail`**。"没查到"和"查错了"是两件事；假失败比漏检更危险。
4. **`skip → fail` 不是回退**（D2.19）。判据是"是否测到"不是"是否通过"。
5. **一个坏形状可以伪装成"一族没有支撑网格"**（D2.23）。`calibrate` 传的 `bd` 是标量而参数是向量 → reshape 异常 → **每个**候选都抛 → `deriv: skip`。形状判据要看**全部**参数。同理 `skip` 也可能是"探针元数错了"（DBI 变双参数后报 `missing 1 required positional argument: 'sigma'`，读起来像族坏了）。
6. **`dpqr` 的 `no_r_counterpart`**：R 侧族对象上没有 `d`/`p`/`q` 槽时记 pass + note，不是 fail。

族发现从 `resolve_family` 的报错信息抓（**146 个**），`dir(distributions)` 只有 19 个。
注册名是**大写**（`BCTO` / `BCTUNTR`）。**最新全族基线见 git log 的 commit message。**

**基准比对先排除超时干扰**：`--timeout 60` 会让 6 个慢族（SI/SICHEL/ZABNB/ZASICHEL/
ZIBNB/ZISICHEL）三关一起消失（`dpqr` 键整个缺失），看起来像 12 处回退。用 300 秒预算补跑。

## `build_ad_family` 的导数 override（D2.17）

`custom_jvp` **只补一阶**。RS 的 Hessian 对 JVP 再求导会**重新进入**特殊函数
（`betainc` / `gammainc`）→ 抛异常。所以 AD 派生的族，log-pdf 不能碰无二阶导的特殊函数。

解法：`score_overrides` / `hessian_overrides`（**形参名不能叫 `score_functions`** —— 会与
函数内后续的局部 dict 遮蔽），且带 override 的参数**跳过 warm-up**。

## parity 对照的五条判据纪律（D2.18 确立）

### 1. 复现 gate 比较的数据要对齐**三件事**

- **族列表**：所有族样本出自同一个 `default_rng(20261006)` 共享流，某族的数据取决于它前面排了多少族。跑 `--families A,B,C` 子集会改变 R 侧基准（ST1 的 R deviance 138.379 → 142.148）。
- **网格 `n`**：`main()` 的 `--n` 默认 **8**（不是 `N_OBS`）。`n` 传给 `_family_grids` → 决定**响应网格** → 决定**哪 48 个观测**（数量恒为 `N_OBS=48`）。
- **拟合配置**：gate 用 `sigma_formula="~1"`、`max_iter=N_CYC`(20)，与公开默认不同。

D2.23 实测三个"看起来都像证据"的数（113.69 / 137.57 / 138.44）**全都不是** gate 真正比较的 208.04。
**一个复现脚本错了不会报错，它只会给你另一个问题的答案。** 正确入口是调 `_build_jobs`。

**前缀等价性**（省时间用）：`[f for f in discover_families() if f <= X]` 与全量对 X 之后的族
**完全等价**（rng 状态只取决于 X 之前抽过的量）。只在"你比的是 X 自己"时成立。

### 2. `rel` 偏差的**方向**决定它是缺陷还是发现

- **我们更高** → 明确缺陷。
- **我们更低** → 先怀疑 **R 提前停**（判据是**绝对量** `abs(G.dev.old - G.dev) > c_crit`，默认 0.001，deviance 在百量级时两三步就够小）。

判定必须**两步，缺一不可**：① 打印 R 的**逐迭代轨迹**；② **用 R 自己的 `d*` 评估我们的系数**。
D2.24 补第三步：**在 R 自己的最优点上看 eta 梯度**（BCPE 的 R 三连：轨迹、梯度、`n.cyc` 扫描）。
处置写在 `_RSDEV_R_STOPPED_EARLY`，**只对"我们更低"开口**，数字漂移即退回 fail。

`gamlss.control` **没有 `delta` 参数**（签名止于 `save, ...`）——传了被 `...` 静默吞掉。
**parity 对比必须用同一判据**，否则是自欺。

### 3. 系数是 **link 尺度**，且 R 拟合对象的**结构随参数个数变**

`coefAll()` 返回 link 尺度：ST1 最优 `tau_eta = 129.28` → 自然值 **1.4e56**，当成自然尺度会把
`sigma = -0.0239`（**负数**）送进密度 → 静默 NaN。两参数族 `DBI`：

```r
length(coef(fit))                        # 2  <- 只回 mu
fit$coefficients                         # NULL
fit[["parameters"]][["mu"]]$coefficients # subscript out of bounds
fit$mu.coefficients                      # <- 正解：顶层按参数名平铺
fit$sigma.coefficients                   # 长度 1（~1 截距，link 尺度）
fit$mu.fv / fit$sigma.fv                 # 48 个拟合值，不是系数
o$dDBI                                   # NULL → 用 gamlss.dist::dDBI
```

`fit[["mu.fv"]]` 是 n 元素 fitted values，取 `[1]`/`[2]` 当截距/斜率会让 deviance 抬到 464
—— **用 `y ~ 1` 让 mu 是标量**消除歧义。

### 4. `ifelse(tau < 1e+06, ...)` 之类的 kernel 开关要单独核

不是装饰：`dST1`/`dST2`/`dTF` 的 R 密度是**两套 kernel 加开关**。而 **R 自己的最优解就落在
正态分支里**（ST1: `tau=1.4e56`；TF2: `nu=3.6e20`）。判据：分支两侧必须**可观测**
（实测差 3e-07），并加"两分支可区分"的断言防删分支后仍全绿。

**R 自己不一致时，复现 R 而不是复现自洽**（D2.23）。`dDBI` 在 `|sigma-1| < 0.001` 切到 `dbinom`，
但 `dldd` **不跟随**：窗口内 R 自己密度的中心差分给 `0.000000000000`，`dldd` 给 `-0.480176057746`。
我们逐位抄 R（差 1.24e-08），登记 `_DERIV_TOLERANCE[("DBI","sigma")] = 1e-2` + 活 R 证据表。

### 5. parity 断言要问活 R，不要硬编码

抄错的 16 位常数会变成一个**通过的**测试 —— 那是 parity 测试最不能有的失败模式。

## R 探针的读取要点（每条都踩过）

- `$nu.initial` 是 **expression 不是闭包**，要 `eval()`（上下文需 `y`、`length`）；`numeric.deriv(f, "m")` 要求变量名 `m` 实际存在于 f 的环境。
- `jsonlite::toJSON` **丢命名向量的 names** → 用 `as.list()`。R 的 `NULL` 被渲染成 **`{}`** 或 `null`（取决于是否经 `as.list()`）—— **判空要同时接受 `None` 和空 dict**。`format(NaN/大数, digits=17)` 返回**字符**。
- `o$parameters` 是 list，`cat()` 报错 → `paste(unlist(...), collapse=", ")`。`o$parameters` 打印出来是 4 个 `TRUE`（R 的 list 走 `[[` 的怪异行为），要用 `names(o$parameters)`。
- `gamlss.family` 对象**没有 `d`/`p`/`q` 槽**（止于 `y.valid`）；`fit$family` 是字符串；`fit$parameters` 是字符向量；`fitted(f, type=)` 对 Box-Cox 族失效。
- 测试里硬编码 Rscript 路径的失败模式与通过一模一样（真路径带 `x64`）。**复用 `rs_family_parity_scan._rscript_path()`**。
- **`gamlss()` 的 `start=` 只接受 gamlss 对象**，传 list 报 "The object in start.from is not a gamlss object"。

**同文件同形不等于同族**：`grep '"nu": "identity"'` 在 `distributions_b15.py` 命中 4 处
（`SEP1`/`SEP2`/`ST3C`/`LOGSHASH`），**只有 `ST3C` 应为 `log`**。批量替换会改坏其余三个。
**元数据对了 ≠ 问题解决**：单修 `ST3C.nu.link` 时 deviance 一位未动（168.410 → 168.410）。
**验收标准是 deviance 落在 R 上。**

## Python 的 `max` ≠ R 的 `ifelse` 夹取

R: `ifelse(v < -1e-15, v, -1e-15)` —— 只替换"不比阈值小"的。
Python: `max(v, -1e-15)` —— **把所有负值都往上夹**。正确的 `-0.02435836` 被夹成 `-1e-15`，
一度看成实现 bug。

## Box-Cox 簇：R 实际有**三组**九个族（D2.13–D2.16 已全部修完）

| 组 | R 名 | 我们 | `mu.link` | 截断 |
|---|---|---|---|---|
| 截断 | `BCT`/`BCCG`/`BCPE` | 同名 | `identity` | 有 |
| `o` | `BCTo`/`BCCGo`/`BCPEo` | 同名 | `log` | 有 |
| 无截断 | `BCTuntr`/`BCCGuntr`/`BCPEuntr` | `BCTUNTR`/… | `identity` | **无** |

**`o` 后缀只改 `mu.link`，不改密度**。无截断族**没有 d/p/q**，只有 `G.dev.incr` + warning。
截断只进入三处：密度的 `-log F(...)`；`dldd`/`dldv`/`dldt` 的 hazard（**`dldm` 不含**）；`p`/`q`
的重标。唯一例外 `BCPEuntr` 的 `d2ldm2` 没有 `tau<1.05 -> -dldm^2` 分支。

**修法顺序：先密度、再导数、最后 link**，且**每族漏项不同，逐族查**（BCT 缺截断项 0.2199；
BCCG 缺 `-log(pnorm(1/(|nu|sigma)))` 0.2018；BCPE 缺标准化常数 `c` + 截断项 + kernel 少 `/c`，
偏差**随 y 变** spread 1.539）。**偏差是常数还是随 y 变决定危险程度**：前者对 deviance 基本无害
但污染全部 score；后者直接改变最优解。打分路径**直接 import `dpqr_functions` 的助手**。

**link 与导数是耦合的，中间态可能更差**。BCT 三步实测（R=199.5647）：
`数值导数+旧link 200.7269` → `仅解析导数 209.5343（更差!）` → `解析导数+R 的 link 199.5666`。
**"换 link 只改优化几何"有前提：导数已正确。不要在单步变好时停下。**

抄 R 导数三条：① **连粗糙一起抄**（`d2ldv2 = -7*sigma^2/4` 是 R 的粗近似，但那就是 R 喂给 IRLS 的东西）；② **R 自身的数值近似会污染验收阈值**（BCT 的 `dldt` 距真实导数 1.84e-05，我们抄来的距 **R** 仅 2.8e-14），断言要问"离**谁**近"；③ RS 只用**对角** Hessian。

其他：`density` 里出现 `scipy.stats.*` 就是定时炸弹（`jit` 内求导 → `TracerArrayConversionError`）。
Box-Cox 簇响应必须严格正（R 遇 `y<=0` 直接 `stop`，我们静默 clamp 成 eps）。`BCT()$dldv` /
`BCCG()$dld*` **向量化调用因 `ifelse` 标量坍缩返回不同值**，R 对照一律逐元素求值。
**"我们的族等于 R 的某个族"这种断言会把缺陷锁死**（`BCTUNTR` 曾断言等于 `dBCT`——那正是缺陷
本身，而它一直是绿的）。

## "密度对、导数错"要往密度里看（D2.20）

`dpqr` pass + `deriv` fail + `rsdev` 大偏 → 入口信号是"密度对、导数错"，但**根因常常在密度**。
`TF2` 三处根因里两处是密度问题：`d2ldv2` 里 R 写 `TF()$d2ldd2(sigma1, nu)`（**两参数**闭式），
我们转发自己的**四参数 AD 版**，`nu=3` 时 `-1.0` vs R `-3.0` → **签名不同就足以偷换函数**；
`dTF` 缺 `nu > 1e6` 正态分支，而 R 自己的最优解落在 `nu = 3.6e20`。

**修好一处可能让结果更差**：score 修好后 deviance 3399 → **6835**（撞地板）。
**不要在单步变好时停下**（BCT 第三次）。

### 同一个密度有**两个**实现，必须同步改

`dpqr_functions.dTF`（RS 评估 deviance用）与 `distributions.TF._log_density`（AD 微分导数用）。
**只改一处 = 重新制造密度/导数分裂**，症状与原缺陷一模一样。

### 转发前先问：被转发的是同一个函数吗

重参数化壳（`TF2`→`TF`）转发的 score/hessian 若用**原始 sigma**，链式法则全丢。更隐蔽的是
**hessian 与转发后的 score 自洽到 1e-11** —— RS 在迭代一个内部自洽但不属于所最小化密度的
梯度，`converged=True` + 合理 Hessian 同时成立。

## 弱识别平坦区的判定（D2.20）

**判据：固定其余参数、扫该参数看目标函数**。TF2 实测 `nu_eta` 到 20 后 deviance 平坦到 15 位
数字 → 两个停止点是**同一个目标函数值**，R 提前停是 `c.crit` 绝对判据。**断言用双向带容差，
不用相等。** 注意此判据的前提是基准点已收敛（见"切片"条）。

## rel 大到荒谬时先怀疑约定层（D2.21 + D2.25）

`BI`/`DBI` 的 `dpqr d` 报 **rel 1.246e+90** —— 全registry 最严重。真相是**扫描器在撒谎**。
**1e90、乃至 `inf` 这种量级不是数值分歧，是"参数/约定送错了"的签名。真族缺陷不会有。**

- **Python 侧 dpqr 必须按名传参**：R 的 `dBI(x, bd, mu, log)` —— **bd 在 mu 前**；按位置传 → `mu` 收到 10 → clip 到 `1-1e-10`，二项系数爆炸到 1e90。R 侧本来就按名传，**两边可以一个对一个错**。
- **离散族的网格必须是整数**：`np.arange(n) * (bd/(n-1))` 在 `n=8, bd=10` 下给出 `0, 1.43, 2.86, ...`，R 的 `dbinom` 对非整数 `x` 返回 **`-Inf`** → 8 点里 6 点在 R 侧不可评估。
- **多个缺陷可能必须叠加才产生症状**：非整数网格下错位峰值 1.2e+90，整数网格下 1.4e+02。"任一即可解释症状"在这里**不成立**。
- **D2.25 的元数据版本**：`ZAPIG`/`ZAZIPF` 的 deviance 从第 1 步就 `inf`，
  读起来像密度缺陷，真根因是 `type_="Mixed"` 一行。**探针没撒谎 —— 是元数据先骗了探针**
  （`Mixed` → 连续网格 → 离散 log-pmf 整段 `-inf`）。**异常量级先怀疑约定层。**

## "我们比 R 严"要先问 R 自己是否有定义（D2.25）

`pig.py:40` 有整数守卫 `x == floor(x)`，R 的 `dPIG` 没有。看起来该去掉，**但 R 在非整数处的取值是插值残留**：
函数在整数处跳跃（`x=3.99`→-3.283，`x=4.0`→-2.069），`p(4.397)=0.068` 落在 `p(4)=0.126` 与 `p(5)=0.100` 之间，
**只有整数和才等于 1**。R 自己把 `PIG` 标 `Discrete` + `y.valid=all(y>=0)` —— 非整数是域外，只是没拦。

**`type`/`y.valid` 表达域，函数行为是实现；两者不一致时以域为准。**
对照 D2.23 的 `dDBI`（切 `dbinom` 但 `dldd` 不跟随时**复现 R**）：那次不一致在**域内**，这次在**域外**。

**逐族修的错要改成规则**：6 个 ZA 族同源，逐族手改等于把同一个 bug 抄 6 遍。
`_base_response_type()` 从底座族推导（底座 `Discrete`→`Discrete`、`Continuous`→`Mixed`、
**其他 → raise**），静默退回默认值就是缺陷本身。

## 二项族的约定链（D2.22）

- **响应约定（计数 vs 比例）要在入口统一**：二项族 `y` 是**成功计数**、`mu` 是**比例**。D2.22 里这个约定被三处各自重新解释（初值 / deviance 的 bd / warm start 的 mean(y)）。**"y 是计数还是比例"必须在入口定一次**。
- **R 的 `BI`/`DBI`/`BB` 的 `mu.initial` 统一是 `mu <- (y + 0.5)/(bd + 1)`**。
- **固定参数必须一路传到似然**：`estimable_parameters` 排除 fixed_parameters 是对的，但 `rs_fit` 的 `other_params`（score/Hessian context）与 `dev_kwargs`（`g_dev_inc`）从 `family.parameters` 取，会**连带把 `bd` 排除** → `g_dev_inc(bd=None)` 走 Bernoulli 分支（y 的计数 7/10 → deviance 8566 而非 202.85，mu 跑到 1-3e-7 却报 `converged=True`）。**这个耦合零测试覆盖。**
- **R 的二项拟合只接受两列响应**：`gamlss()` 只认 `cbind(success, failure)`。我们反过来要 `y`（计数）+ 显式 `bd`。**两边形状必然不同但内容必须相同。**
- **`FamilyHeuristicInit` 的 `mean(y)` 对非均值型 mu 族是陷阱**：`BI`/`DBI`/`BB`/`WEI2`/`RGE`/`PE2` 已列入 `warm_start_policy._NON_MEAN_MU_FAMILIES`。

## `DBI` 双参数似然（D2.23）

`DBI` 曾是 `BI` 的转发壳（`parameters == ("mu",)`），现为 Borg 1988 双参数族。

- **`C` 是有限求和，不是数值积分**。`R/gamlss.dist/src/getBI_C2.c`（65 行，已 vendored）是完整定义。三个闭式猜测全错；误判线索是 `mu ∈ {0.2,0.5,0.8}` 上"C 与 mu 无关" —— **那张网格关于 0.5 对称，而 C 也对称。**
- **`GetBI_C` 先做 `sigma <- 1/sigma`** 再调 C，`ss` 写成那个值的函数。搞反会得到"看起来合理但不是 R 的"密度。
- **`dDBI` 在 `|sigma-1| < 0.001` 切到纯 `dbinom`**。R 的 `sigma.initial` 是 `rep(1.1, ...)`，RS 会**穿过**这个分支。
- **Hessian 是 `-score²` 地板 `-1e-15`**（Fisher scoring，不是 `d2logL`）。
- 验收：`C` worst 1.52e-14 / log密度 4.26e-13 / `p` 12 位 / `q` 逐位；score 对自己密度 FD 1e-09..1e-10，对 R worst ABS 2.45e-05。
- **capability 从 `COMPAT_ALIAS` 移到 `EXPERIMENTAL`**；棘轮推进 6 个符号。**族变真后，拿它当"compat_alias 代表样本"的测试要换样本**（已换成 `DBURR12`），不是删断言。
- **"不再是兼容别名"与"可进生产"是两句话**：`require_production_family_route` **刻意不提供**任何迁移开关（docstring 明说），用 `inspect.signature` 断言 opt-in 不存在（用 `except TypeError` 探测的话，将来有人故意加上时测试会以错误的理由继续绿）。

## helper 缓存范围要匹配实际作用域

`_scanner_sample` 只缓存两族时，D2.20 的四个 R-backed 测试首跑全是 skip，**长得和通过一模一样**。
反向教训：缓存全 146 族要 200 秒，缓存**排序前缀到目标族为止**是精确等价且快 4 倍。