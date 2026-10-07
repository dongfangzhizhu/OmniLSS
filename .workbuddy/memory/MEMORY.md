# OmniLSS 项目长期记忆

## 回归测试：必须分块跑

`omnilss/tests/` 共 **425** 个测试文件。`pytest tests/ -m "not slow"` 全量单次
运行实测 **超过 1h38m 未结束**。JAX 编译密集的单文件就很慢
（`test_consistency_families.py` 9 分钟）。

- 一律按目录/主题分块，每块控制在数分钟内。
- 前台 2 分钟超时会把慢批次打成 SIGTERM —— **看起来像失败，其实是超时**。
  慢批次用 `run_in_background`。
- 无 `pytest-timeout` 插件，`--timeout` 参数不可用。
- 沙箱会吞掉 `--collect-only` 与重定向到文件的输出；看到 "0 cases" 或空输出
  别当结论，换直连 stderr 或 `grep` 数 `def test_`。

## 覆盖工件纪律

`test_suite_sha256` 内嵌于 JSON 工件 → **任何 `tests/` 文件变更必须在同一
commit 内再生工件**：

```
tools/generate_coverage_artifacts.py --date YYYY-MM-DD
tools/coverage_diff_report.py        # 确认 result: OK
```

（`generate_capability_matrix.py` 在 **`omnilss/tools/`** 下，不在 `tools/`。）
门禁测试：`tests/test_symbol_coverage_tool.py`（13 项）。

## 判定"既有失败 vs 本次引入"：`git stash`

改动已提交时用 `git checkout -q <commit>~1` 造对照；未提交时 `git stash push -u`。
这是隔离两类失败的最快手段，本轮多次使用。

## 两条踩过的认知坑

1. **既有用例失败，先分清是代码错还是期望错。**
   `test_grpc_batch_fit_service_direct` 用了非生产族 `GA`，被 fail-closed 准入
   正确拒绝 —— 是期望过时。若"修服务让它放行"就拆掉了安全边界。
2. **"R 没说" ≠ "可以靠常识补"。**
   族级响应域表初版给 19 个族凭分布常识加了限制，核对 R 源码后发现这些族
   **根本没有 `y.valid`**。假拒绝比漏检更糟。族规则只能逐条从 R 源码抄。

## 活 R 对照是唯一可信的 parity 证据

`converged=True` 只说明算法跑完，不构成 parity 证据。
`BE`/`BEINF` 的 `g_dev` 为负一度被判为缺陷，活 R 对照给出**完全相同的值**
（-3.836092659932398），证明是 beta 族密度可 >1 的正常结果。

- bridge 用法：`from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge`，
  然后 `RBridge().call_r_gamlss(data=..., formula=..., family="<NAME>", ...)`。
  **`family` 必须传字符串名**，传 Family 对象会 JSON 序列化失败。
- R 的 `stop()` 被 bridge 包成 **`RuntimeError`** 抛出，不是返回
  `{"success": False}` dict —— 断言要落在异常消息上。

## 收敛状态（D2.2 起）

`GAMLSSModel` 有 `converged` / `termination_reason` 一等字段 + 访问器
`convergence_status()`。词汇表：`relative_deviance_converged` /
`max_iterations_reached` / `non_finite_deviance`（RS+JAX-RS）；
`step_rejected` / `no_progress_step_rejected`（CG）；
`cg_stage_not_converged`（MIXED 补位）。

## 响应域校验（D2.4 起）

`omnilss/response_domain.py`。接入点在 `gamlss()` 内 `resolve_family` 之后、
所有方法路由之前 → 五条路径行为一致。
**`gamlss_ml` 故意不加**（R 的 `gamlssML` 也不校验 —— 正确的 parity 边界）。
R 的 `gamlss()` 校验点在 `gamlss/R/gamlss-5.R:960`。

## JAX RS 批量路径的结构性风险（已登记未消除）

`jax_rs_batch` 的 vmap kernel 是 `jax_rs_core` IRLS 的**独立重写**
（vmap 不能在 traced 值上调 core）。已因此出过一次崩溃缺陷。
两处裁剪逻辑一处用 Python float、一处用 traced 数组，强行抽公共函数不划算；
改为**扩展 parity 断言到全部 7 个字段**锁漂移。

## 族 parity 扫描器（D2.9 起）—— 排查族缺陷的默认入口

`tools/rs_family_parity_scan.py` + `tools/r_parity_batch.R`。三关：
`dpqr`（d/p/q 对活 R）、`deriv`（score 对中心有限差分）、`rsdev`
（RS deviance 对活 R）。单进程跑完 146 族，约 10–30 分钟，**必须后台跑**。

```bash
JAX_PLATFORMS=cpu python tools/rs_family_parity_scan.py --json out.json
JAX_PLATFORMS=cpu python tools/rs_family_parity_scan.py --no-r --families NO,GA
```

**三条踩过的坑，改动前务必记住**：

1. **deriv 关对"响应约定错位"是盲的**。FD 与解析导数跑同一个错误变换，
   自洽但全错（BI 把计数当比例，deriv 全过）。这类缺陷只有 `dpqr` 能抓。
2. **探测网格必须从族对象推导，不能用白名单**：`type` 定响应支撑、
   `links` 定参数区间、`fixed_parameters` 定是否补 `bd`。
   支撑判定用**相对**阈值 `d >= 1e-6 * max(d)` —— LOGITNO 在 y>1 处返回
   极小正数，`d > 0` 会让越界网格"满分"通过。
3. **无法探测的族记 `skip` 不记 `fail`**。"没查到"和"查错了"是两件事；
   假失败比漏检更危险，它会训练你忽略整个工具。

族发现要从 `resolve_family` 的报错信息抓（**146 个**，
已与扫描器输出核对零遗漏），`dir(distributions)` 只有 19 个。
注册名是**大写**（`BCTO` / `BCTUNTR`）—— 别拿 R 风格的混合大小写名字去比。

**全族基线（`--timeout 60`，22 分钟）**：`dpqr` 60/95、`deriv` 130/139、
`rsdev` 47/61。`rsdev` 的 14 个真实缺陷里 `ST1` / `ST2` 与 D2.17 同类。

## `build_ad_family` 的导数 override（D2.17）

`custom_jvp` **只补一阶**。RS 的 Hessian 对 JVP 再求导会**重新进入**
特殊函数（`betainc(tau/2,·)` / `gammainc(1/tau,·)`）→ 抛异常。
所以 AD 派生的族，log-pdf 不能碰无二阶导的特殊函数。

解法：`score_overrides` / `hessian_overrides`（**形参名不能叫
`score_functions`** —— 会与函数内后续的局部 dict 遮蔽），
且带 override 的参数**跳过 warm-up**：warm-up 会对每个参数 trace
`_vmap_hess`，即使结果丢弃，那次 trace 就是抛错处。

## parity 对比前必须核对 R 的控制参数

`gamlss.control` **没有 `delta` 参数**（签名止于 `save, ...`）——
传了会被 `...` 静默吞掉，R 照常用默认 `c.crit = 0.001`。
曾据此把"两侧停止规则差 1000 倍"误判成"我们 deviance 一律更低"。

R 的 RS 判据是**绝对**量：`abs(G.dev.old - G.dev) > c.crit`。
我们的是 `abs_change < tol or rel_change < rtol`（绝对**或**相对）。
**parity 对比必须用同一判据**，否则是自欺。

R 提前停的识别方法：R 系数处的 score 和**显著非零** → 不是驻点；
BCT 的 `tau` 方向 score 峰值仅 3.9e-9 → 完全平坦。

## R 拟合对象的三个语义坑

- `fit[["mu.fv"]]` 是 **n 元素 fitted values**，不是系数对。取 `[1]`/`[2]`
  当截距/斜率重建 `mu(x)` 会得到 `mu(40) ≈ 71`（`y` 在 [0.4,4]），
  deviance 抬到 **464**。**用 `y ~ 1` 让 mu 是标量**消除歧义。
- `gamlss.family` 对象**没有 `d`/`p`/`q` 槽**（止于 `y.valid`）。
  `!is.null(o$p)` 为 TRUE 但 `o$p(...)` 报 "attempt to apply non-function"
  （`$` 部分匹配到 `parameters`）。dpqr 是命名空间级 `pBCTo`/`qBCTo`，
  第一参数名是 `q`。
- `fit$family` 是**字符串**、`fit$parameters` 是**字符向量**、
  `fitted(f, type=...)` 对 Box-Cox 族失效（无论传什么都返回 mu）。

## R 的二项响应约定（D2.10 查明）

`gamlss()` 对二项族收**两列矩阵** `cbind(成功, 失败)` → `y` = 成功**计数**、
`bd` = 总数。`dBI` = `dbinom(x, size=bd, prob=mu)`，`dldm = (y-bd*mu)/(mu(1-mu))`。
**读 R 的响应约定要从 `gamlss()` 入口读，不是从 `dXY()` 函数体猜。**

## 扫描器必须有单族超时（D2.11）

`tools/rs_family_parity_scan.py` 的 `--timeout`（默认 180s）是**必需**的，
不是可选项：Windows 没有 `signal.alarm`，线程也打不断循环，
逐族 `try/except` 只能挡"抛异常"挡不住"不返回"。实测有族能让整轮挂 35 分钟。
超时族记 `skip` 不记 `fail`。

## Box-Cox 簇：R 实际有**三组**九个族（D2.13–D2.16 全部修完）

| 组 | R 名 | 我们 | `mu.link` | 截断 |
|---|---|---|---|---|
| 截断 | `BCT`/`BCCG`/`BCPE` | 同名 | `identity` | 有 |
| `o` | `BCTo`/`BCCGo`/`BCPEo` | 同名 | `log` | 有 |
| 无截断 | `BCTuntr`/`BCCGuntr`/`BCPEuntr` | `BCTUNTR`/… | `identity` | **无** |

**`o` 后缀只改 `mu.link`，不改密度** ——
`identical(deparse(body(dBCTo)), deparse(body(dBCT)))` == TRUE。
"o = untruncated" 是错的推断，别再犯。R 的无截断族另有其名，且
**没有 d/p/q**，只有 `G.dev.incr` + 一条 `F(1/(sigma*|nu|))<0.99` 的 warning。

截断只进入三处：密度的 `-log F(1/(sigma*|nu|))`；`dldd`/`dldv`/`dldt` 的
hazard（**`dldm` 不含**）；`p`/`q` 的重标。link 与对角二阶导数两版相同，
唯一例外 `BCPEuntr` 的 `d2ldm2` 没有 `tau<1.05 -> -dldm^2` 分支。

### 必须先修密度、再修导数、最后修 link

**每族的漏项不同，逐族查，别从一族的结论推全簇**：

| 族 | 密度缺什么 | 偏差形状 |
|---|---|---|
| BCT | 截断项 `-log(pt(1/(\|nu\|sigma),tau))` | 常数0.2199 |
| BCCG | 截断项 `-log(pnorm(1/(\|nu\|sigma)))` | 常数 0.2018 |
| BCPE | 标准化常数 `c` + 截断项 + kernel 少 `/c` | **随 y 变**，spread 1.539 |

**偏差是常数还是随 y 变，决定危险程度**：常数对 deviance 基本无害但污染
全部 score；随 y 变直接改变最优解。两种都要查。

修法：打分路径**直接 import `dpqr_functions` 里 d/p/q/r 用的助手**
（`_bcpe_c`/`_bcpe_ftrunc_log`），从机制上保证"一个族只有一个密度"。

### link 与导数是耦合的，中间态可能更差

BCT 三步实测（RS deviance，R = 199.5647）：
`数值导数+旧link 200.7269` → `仅解析导数 209.5343（更差!）` →
`解析导数+R 的 link 199.5666`。

原因：**`logshiftto2` = `log(tau-2)` 隐含 `tau>2` 地板**，RS 把 eta 推到
-35.35 即 `tau=2.0` 正撞地板；扫 tau 可见 deviance 单调下降并在 `tau→∞`
（Box-Cox 正态极限）触底。**旧 link 是够不到最优解，不是收敛慢。**

"换 link 只改优化几何"**有前提**：导数已正确。导数错时 link 决定的是
RS 会撞上哪块隐含地板。

### "我们比 R 好"要查证，不能默认接受

查法：**把双方系数放到 R 自己的密度函数上算**。BCPE 实测我们 161.9236 <
R 162.0165，在 R 的 `dBCPE` 上复核确认我们确实更优 → 弱识别平坦区
（同 PE），**不是缺陷**。这一步能区分"找到更优解"与"优化了错误的目标"。

### 抄 R 导数时的三条纪律

1. **连粗糙一起抄**：`d2ldv2 = -7*sigma^2/4`、`d2ldd2 = -2*tau/(sigma^2*(tau+3))`
   都是 R 的粗近似，但它们就是 R 喂给 IRLS 的东西 —— "改进"会让 parity 变差。
2. **R 自身的数值近似会污染验收阈值**。BCT 的 `dldt` 用 0.01 前向步长
   （距真实导数 1.84e-5，我们抄来的距 **R** 仅 2.8e-14）；BCPE 的 `dldt`
   用 0.001 步长。**断言要问"离谁近"，不能默认"离我们自己的密度近"。**
3. RS 只用**对角** Hessian，交叉项（`d2ldmdd` 等）用不上，不用移植。

### R 各函数签名不统一

同一族里 `d2ldm2` 有的带 `y` 有的不带，且**带 `y` 的值也可能不依赖 y**
（BCPE 的 `z` 是死代码，y=0.5 与 y=99 同值）。用
`names(formals(...))` 查，别猜。

### 特殊函数梯度支持：逐个测，不能类推

- `jax.scipy.stats.t`：只有 `pdf`/`logpdf`，**无** `cdf`/`logcdf`；
  对**第三参数有梯度**。
- `jax.scipy.special.betainc`：对 `x`（第三参数）有梯度，对 `a`/`b` **无**。
  `a=df/2` 里 df 就是 tau → 按 tau 求导直接抛
  `Betainc gradient not supported`。需 `custom_jvp` 补 d/df。
- `polygamma(1, ·)` = trigamma、`digamma`：与 scipy 一致到 1e-16。
- `dnorm(x)` 漏 `1/sqrt(2*pi)` → hazard 放大 **2.5066**。足以伪装成
  "R 的导数错了" —— 排查时先怀疑自己的探针。

**density 里出现 `scipy.stats.*` 就是定时炸弹**：`fitting_cg` 在 `jit` 内对
`family.d(..., log=True)` 求导，scipy 会调 `__array__` →
`TracerArrayConversionError`，**每个 CG 周期都失败**（BCT 踩过）。

### Box-Cox 簇的响应必须严格正

`y > 0`。R 遇 `y<=0` 直接 `stop("response variable out of range")`，
而我们的 `d*` 静默 clamp 成 eps —— **两侧在非法数据上根本没在比同一个东西**。
测试数据必须构造为严格正。

### R 向量化 quirk

`BCT()$dldv(y, ...)` / `BCCG()$dld*` 向量化调用因 `ifelse(nu != 0, ...)`
标量坍缩，返回值与逐元素调用**不同**（甚至只返回一个元素）。
R 对照一律**逐元素**求值。

**"我们的族等于 R 的某个族"这种断言会把缺陷锁死。**
`test_r_consistency_boxcox_untr_more.py` 曾断言"`BCTUNTR` 的 `d` 等于 R 的
`dBCT`"—— 那正是缺陷本身，而它一直是绿的。参照物必须先独立确认是对的。

**量化函数的支撑边界要单独测。** 无截断族支撑 `1 + nu*sigma*z > 0`，而
**截断族的截断点恰好就是这条边界**，所以截断版的 `q` 从不越界、无截断版会。
沿用 `jnp.maximum(q, eps)` 会返回极小正数而其 CDF 是 0.17（回环静默断裂）。
无截断版返回边界 `0`、`p(0)=0`。

## parity 对照的四条判据纪律（D2.18 确立）

### 1. 跨数据比较 deviance 毫无意义

扫描器**所有族的样本出自同一个 `default_rng(20261006)` 共享流**，
某一族的数据取决于它前面排了多少族。跑 `--families A,B,C` 子集会改变
rng 流位置 → **R 侧基准全变**（实测 ST1 的 R deviance 138.379 → 142.148）。

**唯一可比口径 = 全族扫描**。要复现某一族的数据，调扫描器自己的
`_build_jobs` 导出，**不要另起种子"造一份差不多的"**。这条踩了两次。

### 2. `rel` 偏差的**方向**决定它是缺陷还是发现

- **我们更高** → 明确缺陷。
- **我们更低** → 先怀疑 **R 提前停止**。R 的 RS 判据是**绝对量**
  `abs(G.dev.old - G.dev) > c.crit`（默认 0.001），deviance 在百量级时
  两三次迭代就够小 → 某些方向**从未被探索**。

判定要**两步，缺一不可**：
1. 打印 R 的**逐迭代轨迹**，看哪个方向被冻结（ST1 的 `tau_eta` 从第 3 次
   起冻结在 129.2813、`nu` 停在 0.462 从未动）；
2. **用 R 自己的 `d*` 评估我们的系数**（"同一密度互评"）。逐位吻合 →
   同一目标函数，我们确实更优，是弱识别平坦区。

处置写在 `rs_family_parity_scan.py` 的 `_RSDEV_R_STOPPED_EARLY`：
**只对"我们更低"开口**，且数字必须仍与登记一致（漂移即退回 fail），
并按**登记时的预算**重跑。

### 3. R 的 `coefAll()` 返回 **link 尺度**，不是自然尺度

R 的 ST1 最优 `tau_eta = 129.28` → 自然值 **1.4e56**。当成自然尺度会把
`sigma = -0.0239`（**负数**）送进密度 → 静默 NaN。我们的
`model.coefficients` 同样是 link 尺度。

**唯一出路是荒谬值自曝**（`gamma = 1.404e+00`）。两侧一律用
`[[par]].coefficients`（link）与 `[[par]].fv`（自然）**成对打印**，一次定死。

### 4. `ifelse(tau < 1e+06, ...)` 之类的 kernel 开关要单独核

不是装饰。`dST1`/`dST2` 的 R 密度是**两套 kernel 加开关**，
`ST3`/`ST4` 都有，**ST1/ST2 曾是仅剩的两个漏项** ——
而 **R 自己的 ST1 最优解就落在 `tau = 1.4e56` 的正态分支里**，
漏掉它意味着我们的密度**在 R 的解处就是错的**。

判据：分支两侧必须**可观测**（实测差 3e-07）。两侧相同 → 分支是惰性的，
测了等于没测。另加一条"两分支可区分"的断言，防止删掉分支后测试仍全绿。

### 5. parity 断言要问活 R，不要硬编码

抄错的 16 位常数会变成一个**通过的**测试 —— 那是 parity 测试最不能有的
失败模式。查询式 fixture 还能顺带暴露 R 的 quirk（1.3e-07 的差来自
R 的 `ifelse` 向量化坍缩）。

## 参数地板（`jnp.maximum(par, k+eps)`）的正确性判据（D2.19）

**形状相同的两个地板，性质可以完全相反** —— 读代码区分不出来：

| | R 的边界 | 边界另一侧 | 地板 |
|---|---|---|---|
| `SST.tau` `2+eps` | `m2` 带 `(tau-2)` 分母 | R 自己 `sqrt` 得 `NaN` 并**直接 abort** | **正当**，标的是有效域 |
| `ST3C.tau` `1+eps` | `tau.valid = all(tau > 0)` | 有定义且随 tau 变 | **不正当**，删掉整个区域 |

**唯一判据：拿活 R 在边界另一侧求值，看 R 自己是否真有定义。**
边界另一侧 R 有定义 → 地板删掉了真解空间，是缺陷；
R 是 `NaN`/abort → 地板标的是有效域，应保留。

症状很特征：**地板把一段参数压成常函数 → 导数恒 0 → 密度关与导数关同时
报相对偏差**（ST3C 的 `dpqr d` 8.2e-02 + `deriv tau` 9.7e-02，同一根因）。

定位手法：**分轴扫描**（固定一轴扫另一轴），从"整体不符"缩到"单轴不符"
最快。

## 扫描器 `skip → fail` 不是回退（D2.19）

判据是**"是否测到"**，不是"是否通过"。v6 时 `ST3C` 的 `dpqr`/`deriv` 是
`skip`（从未测到），修好拟合后变`fail`（测到并失败）→ **信息量增加**。
覆盖率停滞时更该关心 skip 率。

## 族元数据（link / valid / initial）必须逐族向 R 求证

同文件同形不等于同族：`grep '"nu": "identity"'` 在 `distributions_b15.py`
命中 4 处（`SEP1`/`SEP2`/`ST3C`/`LOGSHASH`），**只有 `ST3C` 应为 `log`**。
**批量替换会改坏 `SEP1`/`SEP2`**。

R 侧读取要点：`$nu.initial` 是 **expression 不是闭包**（`body()` 返回 NULL
并警告），要 `eval()`（上下文需 `y`、`length`）；`jsonlite::toJSON`
**丢命名向量的 names**，要用 `as.list()` 显式保名。

**元数据对了 ≠ 问题解决**：单修 `ST3C.nu.link` 时 deviance 一位未动
（168.410 → 168.410）。**验收标准是 deviance 落在 R 上。**

## Rscript 路径写错 = 静默全绿

测试里硬编码 R 路径的失败模式与通过一模一样（真路径带 `x64`）。
**复用 `rs_family_parity_scan._rscript_path()`**，别维护第二份候选列表。

## "密度对、导数错"要往密度里看（D2.20）

`dpqr` pass + `deriv` fail + `rsdev` 大偏 → 入口信号是"密度对、导数错"，
但**根因常常在密度**。`TF2` 三处根因里两处是密度问题：

- `d2ldv2` 里R 写 `TF()$d2ldd2(sigma1, nu)` —— **两参数闭式**
  `-(2*nu)/((nu+3)*sigma1^2)`；我们转发自己的**四参数 AD 版**，
  `nu=3` 时`-1.0` vs R `-3.0` → `d2ldv2` 差 13 倍。
  **签名不同就足以偷换函数。**
- `dTF` 缺 `nu > 1e6` 正态分支（D2.18 kernel 开关的漏项），
  而 **R 自己的 `TF2` 最优解落在 `nu = 3.6e20`** —— 漏分支意味着
  我们的密度**在 R 的解处就是错的**（单点 -23.68 vs R -1.2956，
  48 点累加 2334.79 vs 144.17）。

**修好一处可能让结果更差**：score 修好后 deviance3399 → **6835**
（撞 `logshiftto2` 地板）。**不要在单步变好时停下**（BCT 第三次）。

### 同一个密度有**两个**实现，必须同步改

`dpqr_functions.dTF`（RS 评估 deviance 用）与
`distributions.TF._log_density`（AD 微分导数用）。
**只改一处 = 重新制造密度/导数分裂**，症状与原缺陷一模一样。

### 转发前先问：被转发的是同一个函数吗

重参数化壳（`TF2`→`TF`）转发的score/hessian若用**原始 sigma**，
链式法则全丢。更隐蔽的是 **hessian 与转发后的 score 自洽到 1e-11** ——
RS 在迭代一个内部自洽但不属于所最小化密度的梯度，
`converged=True` + 合理 Hessian 同时成立。

## 弱识别平坦区的判定（D2.20）

我们停在 `nu_eta=240`、R 停在 47.33，**deviance 逐位相同**。
判据：**固定其余参数、扫该参数看目标函数**。

| `nu_eta` | 4 | 10 | 20 | 47.33 | 240 |
|---|---|---|---|---|---|
| deviance | 144.742391 | 144.173133 | 144.171705 | 144.171705 | 144.171705 |

`>= 20` 后平坦到 15 位数字 → 两个停止点是**同一个目标函数值**，
R 提前停是 `c_crit = 0.001` **绝对**判据（D2.18 同一模式）。
**断言用双向带容差，不用相等。**

## R 探针的两种"不可评估"形状（都要判空）

`jsonlite` 把 R 的 `NULL` 渲染成 **`{}`**（空对象）**或** `null`，
取决于值是否经过 `as.list()`。D2.19 的 SST 探针拿到 `null`，
D2.20 的 TF2 探针拿到 `{}` → `float(dict)` 抛 `TypeError`。
**判空要同时接受 `None` 和空 dict。**

## Python 的 `max` ≠ R 的 `ifelse` 夹取

R: `ifelse(v < -1e-15, v, -1e-15)` —— 只替换"不比阈值小"的。
Python: `max(v, -1e-15)` —— **把所有负值都往上夹**。
正确的 `-0.02435836` 被夹成 `-1e-15`，一度看成实现 bug。

## 可复用 helper 的缓存范围要匹配实际作用域

`_scanner_sample` 缓存了全部族后，D2.20 的四个 R-backed 测试才真正跑起来
（原先只缓存两族 → 首跑全是 skip，**长得和通过一模一样**）。

## rel 大到荒谬时先怀疑探针（D2.21）

`BI`/`DBI` 的 `dpqr d` 报 **rel 1.246e+90** —— 全registry 最严重。
真相是**扫描器在撒谎**，两族的 d/p/q 全部正确。

**1e90 这种量级不是数值分歧，是"参数送错了"的签名。真族缺陷不会有。**

### Python 侧的 dpqr 必须按名传参

R 的 `dBI(x, bd, mu, log)` —— **bd 在 mu 前**；我们的是
`dBI(x, mu, bd, log)`。`_python_dpqr` 按位置传 `[x, bd, *params]`
→ **`mu` 收到 10**、`bd` 收到概率 → `mu` 被 clip 到 `1-1e-10`，
二项系数爆炸到 1e90。

R 侧本来就按名传（`r_parity_batch.R` 的 `names(aa) <- argnames`），
**Python 侧没有** —— 两边可以一个对一个错。
已改为 `fn(**kwargs)`，参数名取自 `fam.parameters`。

### 离散族的网格必须是整数

`calibrate` 的二项候选 `np.arange(n) * (bd/(n-1))`在 `n=8, bd=10` 下
给出 `0, 1.43, 2.86, ... 10`。R 的 `dbinom` 对非整数 `x` 返回 **`-Inf`**
并打印 `non-integer x = 1.428571` → **8 点里 6 点在 R 侧不可评估**；
我们的 `dBI` 用 `jnp.round(x)` 照算。

**两边对"函数是否有定义"都不一致的网格点，不可能产出有意义的 parity 结论。**

### 多个缺陷可能必须**叠加**才产生症状

| | 非整数网格 | 整数网格 |
|---|---|---|
| 错位调用峰值 | **1.2e+90** | 1.4e+02 |

"两个 bug"最自然的读法是"任一即可解释症状"，**这里不是**。
只修一个会留下"错但看着合理"的数字，比两个都错更难查。

## 写断言前先看实际数值（D2.21）

本轮在新测试里翻了 4 次，**每次都是没看实际数值就下断言**：
把 plain density 当成 log 密度（`d <= 0`）、
对已经是 density 的值又套一次 `np.exp`、
把 `|d| < 1`用在 log 密度上、期望值算错。

**D2.19 立的"探针报错先查探针"同样适用于自己刚写的测试** ——
测试失败时先打印实际值，不要先改断言。

## 固定参数必须一路传到似然（D2.22）

`FamilyDefinition.estimable_parameters` 排除 fixed_parameters是对的
（不进设计矩阵）。但 `rs_fit` 的 `other_params`（score/Hessian 的
context）与 `dev_kwargs`（`g_dev_inc`）从 `family.parameters` 取，
会**连带把 `bd` 排除**。`g_dev_inc(bd=None)` 走 Bernoulli 分支，
把二项成功计数当比例 → deviance 虚高且 IRLS 方向错。
**这个耦合零测试覆盖**，改`rs_algorithm` 时务必一起看。

## `converged=True` 不等于到达极小（D2.22）

它报告的是**步长塌缩**。在两个被 clip 的 eta 值之间两周期震荡的失控
过程，从外面看就是完美收敛 —— BI 的 8566.07 全程 `converged=True`。
判据：`converged=True` 但目标值明显不合理 → 先怀疑震荡，别加迭代数。

## 响应约定（计数 vs 比例）要在入口统一

二项族的 `y` 是**成功计数**、`mu` 是**比例**，这个约定在 D2.22 里被
三处各自重新解释了一次（初值 / deviance 的 bd / warm start 的 mean(y)）。
`dBI` 密度本身完全正确。**"y 是计数还是比例"必须在入口定一次**，
下面所有函数只接受已定好的东西。

R 的 `BI()$mu.initial` / `DBI()$mu.initial` / `BB()$mu.initial`统一是
`mu <- (y + 0.5)/(bd + 1)`（半计数修正，防 logit 无穷）。

## R 的二项拟合只接受两列响应

`gamlss()` 对二项族只认 `cbind(success, failure)`，单列 + `bd` 列会被
拒（`y values must be 0 <= y <= 1`），即使 `BI()$y.valid` 是
`all(y >= 0)`。我们自己的 `gamlss()` 反过来：要 `y`（计数）+ 显式 `bd`。
**两边形状必然不同但内容必须相同** —— 扫描器里由 `_fit_data` 发 `fail`、
R 侧拼 cbind、我们这侧从两列重建 `bd`。

## 分不清"族缺口"还是"拟合缺陷"→ 先 dump 双方参数列表

我们的 `DBI` 是单参数壳 `("mu",)`，R 的 `DBI` 是 `(mu, sigma)`。
连参数个数都不同，一查就清楚。我却是在写断言失败之后才去查的。
