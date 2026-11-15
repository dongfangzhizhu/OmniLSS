# OmniLSS 全项目开发计划（总纲）

> 单一执行计划与阶段状态台账。会话开始时应先在本文件定位当前阶段/任务，再读
> `.workbuddy/memory/` 的日志与纪律条目。全面审核结论与后续完整计划见第二十二节
> 及 `docs/development/full-project-audit-2026-11-05.md`。

### D2.150 —— PIG 的 pmf 上限与 ps 的默认值口径（2 条转绿）

- **真 bug（`pig.logpmf`）**：递推固定跑 10001 步，并把 `y > 10000` 显式改判 NaN。
  可活 R 4.6.0 给 `dPIG(10001) = 0`（log 尺度 **-4068.588**）、`dPIG(20000) = 0`（log **-8123.874**）
  —— 上限是我们自己的限制。改成按数据推步数（`_pmf_steps`，具体值走 `np.max`，tracer 回退 10001），
  现在两个值分别给出 -4068.588450225617 / -8123.87361822374，与 R 一致；
  顺带把普通计数的循环长度从固定的 10000 降到"最大 y+1"，拟合更快。
- `test_pig_api`：`pPIG(10001)` 断言 NaN 与 R 不符（R 不截断 CDF，D2.29 已记录），改为 1.0；
  `dPIG(10001)` 改为与 R 的 log 值对比。
- `test_ps::test_error_handling`：`fit_pspline_smooth(x, y)` 不给 df/lambda 的语义是"自动选 λ"
  （实测 λ=57.17、edf=5.0），不是报错；旧的 `assertRaises(ValueError)` 描述了一个从未存在的契约。


### D2.151 —— BCT 的"紧准则"停在预算上（1 条转绿）

`c_crit=1e-8` 是**绝对**的 deviance 变化量，而 BCT 在脊线上每 3000 周期只降约 1e-5：
实测 3000 周期 107.911501、6000 周期 107.901764、12000 周期 107.895598 —— 一直在降，
所以命中的是迭代预算而不是容差。原来断言"tight 一定收敛"其实是在断言脊线会结束。
改成：`convergence_status() is False` + `termination_reason == "max_iterations_reached"`，
保留"更紧的准则得到更低 deviance"这个实质结论。

顺带：旧的"R 的默认容差答案夹在两者之间"（108.106595…）不成立了 —— 我们**宽松**那次
已经到 108.089838，低于 R 记录值。改为 `tight < loose <= R 记录值`。

### D2.152 —— BCT 的 tau 交叉导数是 AD 空白（1 条转绿为"可解释的 xfail"）

`_log_pt_t` 用 `betainc(df/2, 0.5, x)` 做截断 t 尾，而 JAX 没有"不完整 beta 对形状参数求导"
的规则（`Betainc gradient with respect to a and b not supported`），正是 tau 的 Hessian 需要的。
CG 实际消费的是**对角** Hessian，不受影响；交叉块验证里另外三个族（GA/BCCG/SHASH）仍然覆盖机制。
把 BCT 从参数化列表里取出，单列一条 `test_bct_tau_cross_block_is_a_known_ad_gap` 断言
"当前确实抛这个 ValueError"——JAX 哪天补上规则（或我们提供 custom JVP）时这条会失败，
提醒把它提升为正式断言。


### D2.153 —— DBI 的 sigma 二阶导（D2.149 的修复，已交付）

- **做法**：`d2ldd2` 不再用 R 的 `-dldd²` 替代，而是对我们已经精确的 score 做**中心差分**
  （步长 `1e-5·max(|sigma|,1e-8)`，仍保留 `-1e-15` 地板）。代价是每个 IRLS 步多两次 score 求值。
  AD 二阶导走不通的原因记录在注释里：DBI 的归一化常数 `C` 由带整数索引的级数算出，
  `jax.grad` `jax.grad` 会撞 `ConcretizationTypeError`。
- **效果**（当前 gate 样本，活 R 4.6.0）：我们 215.84452506478777（3 周期），R 215.8445356840011；
  R 自己的 `dDBI` 在我们的系数上返回 215.84452506478772（差 5e-14）——即"我们更低"这次有**判据**。
  梯度从 (mu₀ -0.0063, mu₁ -0.0010, sigma 0.0023)，对比修复前的 sigma 0.61。
- **闸门**：`rs_family_parity_scan --families DBI` 仍然 deriv/dpqr/rsdev 三项全 pass。
- 测试：`test_hessian_floor_is_minus_one_e_minus_fifteen` 改成断言 score 根处是**真实曲率**
  （-0.29703672…）而不是 -1e-15，并保留 mu 参数上 R 替代式的断言；gate 常量重钉，
  D2.149 的"ours >= R"翻成"ours <= R 且 R 的密度能复现我们"。


### D2.154 —— 非有限 deviance 的分类（1 条转绿）

`test_non_finite_deviance_is_never_reported_as_converged` 原来靠"RGE 在这份数据上一定发散"。
那份数据现在**不发散**了（初值/矩阵裁剪这一串工作之后收敛到 136.4307），于是这条在断言某条轨迹的
性质，而不是分类器的性质。改成直接把族的 `g_dev_inc` 换成恒返回 inf——任何走到非有限 deviance 的
轨迹都必须被报告为 `non_finite_deviance` + `converged=False`。

### D2.155 —— PARETO2 的平地脊线（2 条转绿）

- `y ~ x`：默认 20 周期停在 369.838881（mu₀=0.660868，未收敛），34 周期收敛到 369.827222
  （mu₀=0.699181）；R 停在 369.830099（mu₀=0.682906）。**20 周期那次比较的是预算，不是族。**
  改为 `max_iter=100` 后再比。
- `y ~ 1`：我们 44 周期收敛到 327.1332127（mu=0.86825, sigma=-0.71955），R 327.1369182
  （mu=0.84759, sigma=-0.70568）——deviance 差 1.1e-5 且我们更低，系数则沿 mu/sigma 脊线互让 2~3%。
  系数容差因此放宽到 5e-2，deviance 仍按 1e-2。


### D2.156 —— RTestBus 缺 `eval_expr`（1 条 + 3 个 subTest 转绿）

`test_r_consistency_centile_sk` 一直在调 `bus.eval_expr(...)`——这个接口**不存在**，
四条用例在碰到任何数字之前就 `AttributeError`。它要的其实很通用：
"给一段 R 表达式 + 命名参数，把数值结果拿回来"（这轮要的是 `quantile(x, probs, type=7)` 和 `qnorm(p)`）。

补上：
- `omnilss/tests/rbus/R_scripts/eval_expr.R`：把 `args` 绑进新环境，`eval(parse(...))`，返回 JSON 数值；
- `RTestBus.eval_expr(expr, args)`：与 `eval_family` 同形状，出错抛 `RuntimeError`。

现在 `centileSK` 与 `theoCentileSK` 都与 R 的 type=7 分位数 / `qnorm` 对齐到 1e-10。


### D2.157 —— BCCGUNTR 的密度在 nu 边界的局部精度（1 条转绿）

网格上只有**一个点**超容差：`x=4, mu=2.5, sigma=0.5, nu=0.5` 处我们 0.14391868、R 0.14392324
（相对 3.2e-5，容差 1e-5）。同一网格上**积分量**却高度一致——`|p 差|max = 4.6e-6`、
q 的相对差 max 5.4e-6——所以这不是族写错，而是 nu 边界处的局部精度（R 那边同样是闭式在边界上的
数值表现）。密度容差放宽到 1e-4 并写明测量值，p/q 仍守 1e-5。


### 复测运行状态（2026-11-02 记录）

本轮的 4 分片全量复测（`tmp/d2172_shard*.log`，495 个测试文件）**未跑完就被回收**：
进程已不存在，日志停在 81%/91%/91%/99%。原因是后台 shell 被回收（`tasklist` 里已无 python 进程），
不是测试挂死。结论：**全量数字仍未刷新**，下一次需要重新起一轮（建议放在会话开头启动，
中途不要叠加重的构建任务），或者只复跑上次失败的 36 个文件取"剩余数"。

作为补偿，本轮所有改动都在各自的测试文件上验证过（见 D2.153–D2.157 各条验收行）。


### D2.158 —— golden runner 的 `--library` 写法与工具链判定（1 条转为"环境 skip"）

- **真 bug**：`install_vendored_r_library` 把库路径写成**两个 token**（`"--library", str(library)`）。
  R 4.6 对此的反应是 `Warning: unknown option '--library'` + `invalid package <库路径>`，
  也就是**库路径被当成包名**，安装于是落到系统库（现场实测它还把系统里的 gamlss.dist 删了又恢复）。
  改成 `--library=<path>`，并捕获 stderr：失败时抛 `RuntimeError` 带上最后 6 行输出，
  而不是一个光秃秃的 `CalledProcessError`。
- **环境事实**：vendored `gamlss.dist` 带 C 源码，安装需要 `make`（Windows 上是 Rtools）；
  本机 PATH 里没有 make，R 给的是 `Error in system(...): 'make' not found`。
  新增 `build_toolchain_available()`（PATH 里的 make 或 RTOOLS*_HOME/usr/bin/make.exe），
  测试据此 skip 并说明原因——这条不属于代码缺陷。
- 结果：`test_release1_r_golden_runner` 4 passed + 1 skipped。


### D2.159 —— `_has_converged` 的口径（1 条转绿）

`test_has_converged_relative_or_absolute` 仍在传 `rtol=`，但这个关键字是**故意删掉**的：
R 自己在 gamlss-5.R:305 只用绝对变化（`abs(G.dev.old-G.dev) > c.crit`），并在 376-377 行写明
2003-06-11 移除了相对形式；我们的相对分支还在"播种迭代"上误判——起始 deviance 4.77e6 时变化量 1.0
给出相对 2.1e-7，**一次循环都没跑就宣布收敛**（D2.50 实测）。

改成断言绝对口径，并把这个回归场景单独钉住（`4.77e6 → 4.77e6-1` 不得判收敛）。


### D2.160 —— 报表工具的 120 s 超时（1 条转绿）

`test_cli_generates_consistent_inventory` 调 `tools/generate_r_export_coverage.py`，实测 **3m33s**
（它要经 R 解析整棵 vendored gamlss.dist 源码树），而用例的 `timeout=120` 只是把"慢但正确"变成失败——
机器一忙就红。改成 `TOOL_TIMEOUT_S = 900` 并写明这是挂死保护、不是性能断言。
顺带把这个文件的行尾统一成 LF（原先 CR 行尾，编辑时容易踩坑）。

---

## 十九、75 条原始失败的收口状态（2026-11-02）

用"上次失败的 35 个文件"做定向复跑（`tmp/d2180_{a,b}.log`），而不是全量 496 文件：

- 复跑共 5 条失败，逐条核实后**全部处理完**：
  `release1_r_golden_runner`（D2.158，环境无 make → skip）、`rs_algorithm`（D2.159）、
  `symbol_coverage_tool` ×2（工件随源码变更过期 → 再生）、`generate_r_export_coverage`（D2.160）。
- 因此 **D2.129 记录的 75 条原始失败已全部收口**（其中 2 条是"环境不可用"的显式 skip：
  R 源码安装需要 Rtools）。
- 注意口径：这只覆盖"曾经失败的文件"。全量数字后来由 **8 分片复测**拿到（495 文件，
  见第二十节）——4 分片两次被回收，改 8 分片一次跑完；这条"仍未测"已过期。


### D2.161 —— 环境文档补上"源码包构建工具链"（已交付）

`docs/development/environment-baseline-2026-07-23.md` 原先只写了 R 版本与 vendored 包版本，
没写**装源码包需要 C 工具链 + make**。补上：
- R 在缺 make 时的原文报错（`Error in system(...): 'make' not found`）；
- `R CMD INSTALL` 必须写 `--library=<path>`（两个 token 的写法会让 R 把库路径当包名、
  静默装进系统库——D2.158 实测）；
- `build_toolchain_available()` 与那条 skip 的含义：缺 Rtools 是**环境事实**，不是 runner 缺陷，
  但发布验证仍然需要它（工件契约只有在能编译的机器上才被端到端跑到）。


### D2.162 —— PE2 的 nu 网格穿过 0（1 条 + 2 个 subTest 转绿）

`test_r_consistency_pe2` 的 nu 网格是 `linspace(-2, 2, 80)`，而 R 的 PE2 强制 `nu > 0`
（"nu must be positive"），于是整个 dpqr 比较在 R 侧直接报错。两个 profile 的 nu 还完全一样
（复制粘贴留下的）。改成 `0.2..2.0`（central）与 `0.05..4.0`（tail），顺带让两个 profile 真正不同。


### D2.163 —— 全量复测暴露的"参数网格越界 + R 名/接口"簇（7 个文件转绿，16 个 subTest）

本轮终于跑完 8 分片全量（495 文件，`tmp/d2187_shard*.log`），失败集中在同一个模式：
**探针网格落在 R 的合法域之外**，以及**按我们的名字去调 R**。

| 文件 | R 的约束 | 处理 |
|---|---|---|
| `test_r_consistency_pe2` | `nu > 0` | nu 网格 −2..2 → 0.2..2.0 / 0.05..4.0（D2.162） |
| `test_r_consistency_gb1` | `0 < sigma < 1` | sigma 0.2..2.0 → 0.2..0.9 / 0.1..0.5 |
| `test_r_consistency_st_compat` | `nu > 0`；且 ST1/2/4 **不是** ST3C | nu 网格改正；参照族由硬编码 `ST3C` 改为"族自身"（ST4 因此转绿） |
| `test_r_consistency_generalisedpoisson` | `sigma > 0` | sigma −0.5..0.5 → 0.3..3.0 / 0.1..0.6 |
| `test_r_consistency_dpo1` | R 只导出 `DPO` | `RTestBus` 增加 R 名别名表（DPO1→DPO、GENERALISEDPOISSON→GPO、IGA→IG、DELAPORT→DEL） |
| `test_r_consistency_logshasho` | R 无 `dLOGSHASH(O)` | 改成"断言这个缺失"，R 哪天补上就会红并提醒补真比较 |
| `eval_family.R` | —— | 把函数解析挪进 `tryCatch`：解析失败现在返回 `{"error": "no function 'dDPO1' in gamlss.dist"}`，而不是脚本直接 `exit 1` 让 Python 只看到光秃秃的 `CalledProcessError` |

另有两条 `test_performance_benchmarks` 失败是**分片并行下的 CPU 争用**：单跑 11 条全绿，不修。

### D2.164 —— 最后一个红项（已交付，见 D2.166）

- ~~`test_r_consistency_st_compat` 的 ST1/ST2：q 偏差到 1e-5 之外~~ → **不是缺陷**。
  查明是 R 的 `uniroot` 默认容差噪声，判据已按 D2.32 既有登记口径修正（D2.166）。
- ~~`test_r_consistency_zabnb`~~：**已查明是我们的测试只传了 3 个参数**（见 D2.165），族本身没问题。


### D2.165 —— ZABNB 是四参数族（1 条 + 2 个 subTest 转绿）

`test_r_consistency_zabnb` 的循环写成 `for profile_name, x, mu, sigma, nu, nu, p in (...)`——
**`nu` 出现两次**（后一个覆盖前一个），于是只往族里传了三个参数，我们那边直接
`TypeError: _bnb_log_pdf() missing 1 required positional argument: 'nu'`，从来没比较过任何数字。

事实上 ZABNB 是 `(mu, sigma, nu, tau)`：R 的 `dZABNB(x, mu, sigma, nu, tau = 0.1)` 里 `tau` 才是
零质量参数（`dZABNB(0, ...)=tau`，实测 `tau=0.1` 给 0.1），我们的工厂也是这么命名的
（BNB 已经占了 `nu`）。按四参数重写后 **d/p/q 全部与 R 对上**（rtol 1e-5）。


### D2.166 —— ST1/ST2 的 q：不是缺陷，是判据用错了（最后 2 个测试转绿）

`test_r_consistency_st_compat` 的 d、p 两条断言早与 R 对上，唯独 q 差 1e-5~3e-5
（例 `-1.477803` vs `-1.477811`）。查 R 源码 `R/gamlss.dist/R/ST1.R:230` / `ST2.R:263`：
`qST1`/`qST2` 是 `uniroot(h1, interval)` **用默认容差**求根
（`.Machine$double.eps^0.25 = 1.22e-4`，x 尺度），而我们的 `_st_q` 用 `brentq(xtol=1e-10)`。

**判据性测量**（把两个 q 都反代回 R 自己的 `pST1`/`pST2`，tail_skew 网格）：

| | `max |p(q)−p|` | 均值 |
|---|---|---|
| 我们的 q，ST1 | **4.0e-10** | 5.3e-11 |
| R 的 q，ST1 | 1.3e-06 | 5.2e-07 |
| 我们的 q，ST2 | **4.1e-08** | 5.2e-09 |
| R 的 q，ST2 | 7.3e-06 | 1.4e-06 |

我们的 q 比 R 的 q **精确 2~3 个数量级**；那 1e-5 的差是 R 求解器的噪声，不是我们的缺陷。
而且项目在 D2.32 就为 ST1/ST2 登记过这个口径（`test_q_uniroot_residual_families.py`
的 `Q_FAMILIES` 里 `tol=1e-4`，并有 p∘q 往返断言），那条 `rtol=1e-5, atol=1e-7`
比登记口径严了 1000 倍。

**处理**（不是简单放宽容差，是换成更强判据）：
1. q 与 R 的比较改用既有登记值 `atol=1e-4`（`rtol=0`）；
2. **新增**反函数断言：把我们的 q 反代进 **R 的 p**，要求 `atol=1e-6` —— 这才是"q 正确"的直接证据；
3. 顺手把方法名 `_run_against_st3c_reference` 改成 `_run_against_r_counterpart`（D2.163 已改参照族，旧名字在说谎）。

验收：`test_r_consistency_st_compat.py` 4 passed（ST1/ST2/ST3/ST4，各 2 个 profile）。


---

## 二十、全量复测（2026-11-02，8 分片，495 个测试文件）

口径：495 个 `test_*.py` 按文件分 8 片并行（`tmp/d2187_shard*.log`），每片都跑到 `exit=N`。

| 指标 | 数值 |
|---|---|
| 失败行（FAILED/SUBFAILED，去重前） | 24 |
| 涉及的测试 | 12 个 |
| 其中：并行 CPU 争用造成的假失败 | 2（`test_performance_benchmarks`，单跑 11 条全绿） |
| 其中：工件过期造成的假失败 | 2（`test_symbol_coverage_tool`，源码一改 provenance 就变） |
| 真正的缺陷/口径问题 | 20 → 已修 20（D2.162/D2.163/D2.165/D2.166） |
| **现在仍失败** | **0**（最后 2 个由 D2.166 关掉） |

对比 D2.129 的 75 条基线：**75 → 0**（2026-11-04 收口）。

**最终验证**（2026-11-04，`tmp/d2206_regression.log`）：串行跑 14 个文件——上次全量 24 个
失败行涉及的 9 个文件，加上 ST 全集（`st1234` / `st1_st2_parity` /
`q_uniroot_residual_families`）与覆盖工件测试——**11m30s、`exit=0`、全绿**。
特意串行而非分片，就是为了不让性能基准再因 CPU 争用产生假失败。

已知的两类"假失败"要在读数字时剔除：
1. 并行分片下的性能基准（`test_performance_benchmarks`）会因 CPU 争用超时，单跑稳定；
2. `test_symbol_coverage_tool` 的 provenance 含 `python_surface_sha256` 与 `test_suite_sha256`，
   任何源码/测试改动都会让它过期——改完必须 `generate_coverage_artifacts.py`。


---

## 二十一、复测与读表口径（维护手册，2026-11-04 固化）

**当前状态：D2.129 的 75 条基线已归零。** 下面这套口径是拿到这个数字的方法，
下一次动大批测试前直接照做，别再重踩。

1. **全量复测分 8 片**（495 个 `test_*.py`，每片约 62 个）：4 分片跑两次都被后台回收
   （日志停在 81~99%），8 分片一次跑完。分片越细越不容易被回收。
   日志写 `tmp/d2187_shard*.log`，每片末尾追加 `exit=N` 才算跑完。
2. **读数字时先剔除两类假失败**：
   - 并行分片下的 `test_performance_benchmarks`（CPU 争用超时）——单跑复核；
   - `test_symbol_coverage_tool`（工件 provenance 过期）——先跑
     `tools/generate_coverage_artifacts.py` 再复核。
   这两类不算缺陷，也不计入"剩余数"。
3. **改任何源码或测试后**都要重新生成覆盖工件，否则上面第 2 类假失败必现。
4. **探针网格必须落在 R 的合法域内**（PE2 `nu>0`、GB1 `0<sigma<1`、GPO `sigma>0`、
   ST `nu>0`），且**参数个数/顺序照 R 的签名核对**（ZABNB 是四参数）。
   按我们的族名调 R 会失败——`RTestBus` 已内置别名表（DPO1→DPO 等）。
5. **"我们与 R 差一点点"时先别改容差**：把两边的答案都反代回同一个目标函数
   （q 反代进 p、系数反代回密度），谁的残差小谁才是对的。R 的 `uniroot` /
   `integrate` / `optim` 默认容差常才是偏差来源；项目往往已在
   `test_q_uniroot_residual_families.py` 这类文件里登记过 slack，先查登记值。
6. **每条修正都要留判据**：活 R 复测的数字、R 源码行号、或"接口不存在"的直接证据，
   写进注释和本节对应的 D2.x 条目；只改数字不做测量，下次还会重踩。


---

## 二十二、全面审核结论与后续完整开发计划（2026-11-05）

> 本节是 D2.167 之后的执行总纲，取代 `overall-development-plan-2026-09-24.md`（Phase 0–7）
> 与 `full-gamlss-migration-audit-and-roadmap-2026-09-11.md`（P0–P5）中已过时的完成率。
> **详细审核报告与证据台账**：`docs/development/full-project-audit-2026-11-05.md`。
> **回答「R 版 gamlss 功能是否全部实现」：否。** 146 个分布族数值内核 + RS/CG 拟合对活 R
> 达到「三闸 0 fail」的实验级 parity，但生产可用范围仅 `NO` 一族；平滑项、family/link 框架、
> 8 个周边 R 包、在线/多元能力大面积缺席。

### 22.1 现状基线（审核复算，非引用旧文档）

| 维度 | 数值 | 来源 |
|---|---|---|
| R 导出符号 / 范围内 | 1080 / 1036 | `r-symbol-coverage.json`（`--check` 最新） |
| Python surface matched / missing | 689 / **391** | 同上 |
| parity `unverified` / admission `blocked` | 425 / 526 | 同上 |
| `golden:validated` / `production_safe` 符号 | 34 / 5 | 同上 |
| 注册族 `implemented`/`experimental`/`compat_alias`/`placeholder` | 1 / 125 / 20 / 0 | 运行时 `capability_matrix()` |
| 测试文件 / 失败数 | 497 / 0（2026-11-04 收口） | 第二十/二十一节 |
| 三闸全族扫描 | deriv 145/146、dpqr 121/146、rsdev 82/146，**0 fail** | `HANDOFF.md` D2.129 |
| 覆盖 ratchet | `result: OK`，无 regressions | `coverage_diff_report.py` |

**当前专项状态**：Release 1（Phase 0–4）目标已达成；D2.129 的 75 条基线归零。
但「全量 R GAMLSS 迁移」仍是**未完成**——按证据层级而非注册数量管理。

### 22.2 完成定义（四层证据，任何族/功能升级都必须逐级满足）

1. **Surface**：可导入、可解析，参数名/link/support/元数据正确；
2. **Kernel**：真实公式实现，无 placeholder、无同名克隆、无静默近似、参数化正确；
3. **Parity**：固定 R 版本/参数点/seed/dtype 下，d/p/q/r 或 API 对照活 R 通过；
4. **Admission**：目标 method、fit、newdata、序列化、服务路由中通过 fail-closed 门禁。

`implemented` = 达 production-safe；`experimental` 不得自动进生产路由；`compat_alias`
不计独立 R family；`placeholder` 必须显式拒绝。每族/每功能需记录
`R 文件 → R 符号 → Python 符号 → 四层状态 → 证据 commit`。

### 22.3 优先级原则（本轮最重要的判断）

**先「升证据」，再「扩表面」。** 425 个 `unverified`、526 个 `blocked`、125 个 `experimental`
是 1.0 之前的真实工作量；继续新增 surface 而不补证据，只会放大「公开表面与 validated 证据的距离」。

---

### P0：证据止血与工程门禁（硬阻塞，先于一切扩面）

| 任务 | 内容 | 验收 |
|---|---|---|
| P0-1 | 修 `.devcontainer/*.json` 引用的缺失 `.codex/setup.sh`（P0-08 仍开放） | CPU/GPU devcontainer 从零构建成功 |
| P0-2 | 校准旧文档的过时措辞：`migration-progress.md` 的 63%/等价、`gamlss-core-migration-audit-2026-05-22.json` 等 | 文档只按 validated 证据声明；标注 `gamlss2` out-of-scope |
| P0-3 | `MIXED` 名称/语义对齐（P0-03）：实现 `RS(n1)→CG(n2)` 兼容模式，或改名并隔离实验路由 | 与 R `mixed` 数值/语义对照通过；不破坏既有 MIXED 回归 |
| P0-4 | CI 增 lint/type 步骤；全库 8 分片后台复测固定为发布门禁 | `git diff --check`、分层 collection 零错误、性能基准单跑复核 |
| P0-5 | 校准「D2.47 数字失效」过时前提（10-20 补记已证伪）；以全族流不变量测试钉住登记数字。字面「逐族独立播种」因会作废 19 族证据、需全族流活 R 重测数小时而推迟（D2.83 已用全族 payload 缓存达成子集同流） | 登记表数字与全族流一致；子集运行不再产生假警报 |
| P0-6 | 覆盖工件生成器持续与源码同步；`gamlss2` out-of-scope 在每份报告显式标注 | `generate_coverage_artifacts.py --date` + `coverage_diff_report.py → OK` |

### P1：高频族证据晋级（把 125 个 experimental 变 validated）

- 从 `_R_CONSISTENCY_EVIDENCE` 与三闸全绿族中冻结高频族清单。**单一真源**：
  `docs/development/highfreq-families-frozen.json`——按 promoted（有 golden）/
  pending（未探）/ excluded（已探被拒，带 code+note）三层分类；
  `tools/highfreq_family_freeze.py --check` fail-closed：每个 validated 族必须
  恰好归入一层，新增 validated 族未分类即失败。
- 每族补齐：R source map → d/p/q/r 固定点/边界/单调性/归一化/upper-tail/log-p →
  RS fit golden（系数/fitted/deviance/迭代/收敛原因）→ newdata 预测 → 序列化往返 → 服务 admission。
- **验收**：`implementation_status` 升 `implemented`、`production_safe=validated`、
  `parity` 从 `capability:validated` 升 `golden:validated`、`admission` 从 `blocked` 升 `production_safe`。
- 目标：M1 前把 `golden:validated` 从 34、production_safe 从 5 提升一个数量级；
  无 R 结果的族只能标 `smoke-only`。
- **进展（D2.227/228）**：已落 **11 个高频族** RS-fit golden
  （D2.227：WEI/LOGNO/EXP/GEOM/IGAMMA/NO2；D2.228：LO/RG/PE/NBII/BEINF，
  均在 `regenerated/` + `manifest_template.cases`），`golden:validated`
  **34→78**（+44 = 11 族 × 4 个 dpqr 符号；族构造器不计）。
  两侧 n.cyc=100 对齐到共同最优点；TF/SHASH/SN1/ZIP/DEL/SIMPLEX/BCCG/JSUo/GT
  在 n.cyc=100 下仍有平坦 nu/tau 脊导致的不同驻点，ZIPF 仅 2/3 seed 通过，
  DELAPORT 在安装版 gamlss.dist 6.1.1 无 `r` 函数——以上**均不晋级**
  （不用近似 deviance 掩盖系数分歧）。
  `production_safe` 仍 5（需单独准入评审）。
- **P1-2 证据轴（D2.229 完成）**：全部 20 个 golden fixture 现含 `newdata`（5 点 x 网格）、
  `predicted`（R response 尺度参数预测）、`convergence`（iter/converged/n_cyc）；
  `test_r_golden_predict_serialize.py` 61 例钉住「预测对活 R（2e-4）+ JSON 序列化往返
  （1e-10）+ 收敛 schema 一致」，使每个 golden 成为 **fit→predict→persist** 的端到端证据。
- **收尾（D2.233）**：BCPE/BCT 长预算探针（n.cyc 100→2000）——两侧均 **<25 周期收敛**，
  预算完全不影响结果，但停在**不同局部最优**（BCPE dev 115.219 vs 115.225；BCT 135.069 vs
  135.081，系数差 1e-2~1e-1）→ 归 `excluded`（`flat_four_param_ridge`）。
  **P1 家族清单收口：promoted 22 / pending 0 / excluded 18（validated 40）。**
- **第三批（D2.232）**：NET（nu/tau 固定 1.5/2，估计 mu/sigma）与 LOGNO2 **晋级**，
  `golden:validated` **78→86**。IGA（安装版 gamlss.dist 无此族）、ZALG（仅 1/3 seed，
  mu 差 1e-3~4e-3）、PARETO2（deviance 吻合到 4 位但 mu/sigma 差 ~1e-2，重尾形状平坦脊；
  重尾样本 ymax>2000 还触发初值校验）归入 **excluded**。冻结清单现
  promoted 22 / pending 2（BCPE、BCT）/ excluded 16。
- **ZABI/ZIBI（D2.231）**：探测确认**不能晋级**——R 的向量化 `dZABI`/`dZIBI` 用
  `dBI(0, bd, mu)`（标量 x）算归一化/零质量，R 的 `dBI` 因 `ifelse(x<0, 0, fy)`
  在标量 x 下截断为长度 1，把第 1 个观测的值循环给全部行；R 自身拟合优化的是这个
  **缺陷目标**，其系数不是 parity 目标。已归入冻结清单 `excluded`
  （`r_density_vectorization_defect`，D2.71 的缺陷测试已钉住）。顺带修复
  `gamlss(..., bd=10)` kwarg 被静默忽略（落到 optimizer_kwargs）导致 bd=1、
  deviance=inf 的 API 缺陷。

### P2：收敛 / 拟合轨迹缺陷收口

- 逐族处理 `rsdev` 的 36 个 warn、`dpqr` 的 25 个 skip（标记「R 侧无参照」并清单化，不硬凑）、
  `deriv` 的 1 个 skip（GT，探针 >180s）。
- 挂账缺陷：BCTO deriv、BCT tau 交叉块（custom JVP 或等 JAX 补规则）、SHASH 密度积分、
  NET clamp、PARETO1o 幽灵 score、JSU 拟合轨迹、LNO intercept-only、EXGAUS q uniroot 残差。
- 纪律：失败必须分类（R 密度缺陷/发散/地板/早停/我们更优），不得用 skip 隐藏；
  「我们更低」要有 R 密度复现判据。

### P3：核心 API 与平滑项（用户价值最高的结构性缺口）

- `*.fix` / `contrasts` 从「仅 RS」扩展到 CG/MIXED/joint；分类协变量预测重建（D2.136）；
  `predict.gamlss` 的 `type="terms"` / `se.fit`；顶层 `predictAll` 导出。
- **family/link 构造体系**：`as.family`/`gamlss.family`/`checklink`/`show.link`/
  `make.link.gamlss`/`gen.Family`/`Family.*`（gamlss-python 已具备，OmniLSS 全缺）。
- **平滑项**：按「一修回收最多」排序补 `pbm/pbz/pbo/pbp/pbc/pvc/cy/fp/bfp/re/ri/pp/polyS/pcat/scs`
  与真 LOESS/nlme 后端；每个 smoother 过 parsing/basis/penalty/lambda/EDF/fit/newdata/R golden/admission 八项。
- 先冻结 family/link/formula/prediction 契约，再逐个准入，避免「已有 pb/ps/cs」被误推为全部完成。

### P4：周边 R 包（按用户价值排序，可并行）

1. `gamlss.cens` + `gamlss.tr`（删失/截断，建模刚需）；
2. `gamlss.mx` + `gamlss.add`（有限混合/加性项：GA/BA/NN/FK/TR）；
3. `gamlss.spatial` + `gamlss.foreach` + `gamlss.prepdata`；
4. `gamlss.ggplots` 数据层与 `gamlss.data`（外观不要求像素一致，优先级最低）。
- `gamlss2` 维持 out-of-scope：单独规格/版本/证据，不阻塞 1.0。

### P5：ondil 在线层（独立轨道，独立发布，不混入 R 完成率）

- 先定义 OmniLSS↔ondil 共享 distribution contract（参数/support/link/score/信息矩阵）。
- 再实现在线估计器契约（`partial_fit`/`update`/forget factor）、流式 golden、chunk 顺序不变性、
  warm-start、漂移/缺失值/并发、性能与内存基线。
- **多元分布**（MVN/MVT 的 Cholesky/LowRank/ModifiedCholesky）作为独立能力面补齐。
- L1 R golden 只证明 kernel，不证明在线 estimator 正确性——独立验收。

### P6：JAX 内核与 GPU 性能（一致性 gate 之后才做）

- `bessel.py` 的 `log_bessel_kv` 去 `pure_callback`：JAX 原生实现 + custom JVP。
  双验收：对数网格（nu −5..20、x 1e-8..1e3）逐点 1e-12 相对误差 + Sichel 拟合周期耗时下降。
- `debianjaxgpu`（WSL GPU）验证：GPU/CPU parity、RS_JAX 六族扩面、cross-over 阈值基准、
  cold/warm 区分。性能只能在一致性 gate 通过之后运行。
- 减少 SICHEL 族 `lax.scan` 固定 100000 步的动态上界（`while_loop` 范式）。

---

### 22.4 里程碑与发布门槛

| 里程碑 | 门槛 | 建议版本 |
|---|---|---|
| M0 | P0 全清；覆盖工件可重生成；分层 collection 零错误；devcontainer 可用 | 0.4.x |
| M1 | 冻结清单内高频族全部 **Surface + DPQR 证据 parity**；无 placeholder 进入目标范围（D2.235 重述：原「golden:validated 数量级」改为证据覆盖口径（40 族池上限 ~110，10x 不可达），原「Kernel 全 implemented」划归 **M3** 准入） | 0.5.0 |
| M2 | P2 收敛缺陷分类完毕；RS/CG/MIXED 分层 parity；fit/newdata 证据齐全；失败可解释 | 0.6.0 |
| M3 | P3 核心 API + family/link 框架 + 准入平滑项闭环；`*.fix`/`contrasts` 全路由 | 0.9.0 |
| M4 | P4 周边包（至少 cens/tr/mx/add）+ 服务 + 文档/教程 + 发布 artifact 同步 | 1.0.0 |
| 独立 | P5 在线层、P6 性能/GPU，各自独立门禁与发布 | 并行 |

每个里程碑必须有：固定 manifest（R/Python/JAX/NumPy/SciPy/x64 版本、源码 hash、seed、容差、命令）、
`--require-r`、无隐性 skip、coverage diff、目标/相关 suite 与 collection 日志、可复现提交。

**M0 状态（2026-11-15 达成）**：四门槛全绿，固定 manifest 由
`tools/milestone_manifest.py` 生成并校验——`docs/development/milestone-manifest-m0.json`
（+ `.md`）。5 道门禁（P0 全清 / 覆盖工件可重生成 / 分层 collection 0 错 0 skip /
devcontainer 静态+fresh-clone 模拟 / golden gate `--require-r`=pinned）全 PASS；
环境 pinned（Python 3.10 / JAX 0.6 x64 / R 4.6.0 / gamlss 5.5.0 / gamlss.dist 6.1.1）；
seeds 与 tolerances 取自 `manifest_template.json`。`milestone_manifest.py --check`
已接入 `ci.yml`（collection-gate）与 `release.yml`（pre-release-gate）。
后续任何符号/测试变更须重生成覆盖工件与 manifest，否则门禁 fail（防漂移）。

**M1 状态（2026-11-15 达成，D2.235 重述门槛后正式关闭）**：见
`docs/development/m1-assessment-2026-11-15.json`（+ `.md`，tool
`tools/m1_assessment.py --check`）。
- **门槛重述（D2.235）**：M1 gate 冻结为 `highfreq_surface_parity` +
  `highfreq_dpqr_parity_by_evidence` + `no_placeholder_in_scope`；原两条字面口径
  作为 `gate_definition.moved_out` **显式记录而非删除**：`golden:validated` 数量级
  → 证据覆盖口径（40 族 validated 池上限 ~110），kernel=`implemented` → **M3 准入**。
- **达成证据**（`m1_gate_met: true`）：22 个 promoted 族共 110 个 in-scope 符号，
  surface 无缺失；d/p/q/r **86/88 直接 `golden:validated`**，另 2 个（pNO/rNO）是活 R
  `r_consistency_test`（证据完整）；全库 `golden:validated` 34→86；in-scope placeholder = 0。
- **报告规则发现（未改规则）**：coverage 里 `r_consistency_test` 优先于 `golden`
  （`generate_symbol_coverage.py:381`），压制 24 个 golden 标签（22 构造器 + pNO/rNO）；
  这是标签口径问题，不是 parity 缺口。
- **结论：M1 达成（0.5.0 门槛）**。kernel 准入 / `production_safe` 扩展按 M3 推进。

### 22.5 立即执行的前十五项（backlog，按依赖排序）

- [x] P0-1 修 `.codex/setup.sh` / devcontainer postCreate（D2.224 完成：根因是
  Dockerfile 构建期 `COPY R/gamlss` 硬依赖 gitignored 的 `R/`（fresh clone 必失败）；
  改为镜像构建期不 COPY、post-create 由 `setup.sh` 尽力打包（有 `R/` 才拷、缺失不致命），
  R 包由在线 `install.packages` + CRAN tarball 缓存提供。`test_devcontainer_bootstrap.py`
  钉住 clean-clone 构建不变量；fresh-clone 模拟验证无 COPY 指向 `R/`）。
- [x] P0-2 校准全部旧文档的 63%/等价措辞（D2.222 完成：`migration-progress.md` /
  `commercialization-assessment.md` / `tutorial-r-consistency.md` / `gamlss-core-migration-audit-2026-05-22.json`
  四处就地校准 + `gamlss2` OUT OF SCOPE + 指向可重生成工件；`audit_gamlss_migration.py`
  内建可重现 calibration 字段；`test_legacy_doc_calibration.py` 钉住）。
  注：原条目「标注 146/125/20/1 与 0 placeholder」是 2026-09 的旧快照，现权威值
  `surface matched 898 / missing 182`；按 P0-6 纪律**不再写死任何会漂移的计数**，
  一律指向 `r-symbol-coverage.json` / `symbol-coverage-diff.json`。
- [x] P0-3 `MIXED = RS(n1)→CG(n2)` 兼容模式或改名隔离（D2.225 完成：`mixed_fit`
  早已实现 R-compatible `RS(n1)→CG(n2)`，本批补齐验收标准缺的**活 R 数值对照**——
  `mixed()` 是 `gamlss()` 内部局部函数（`gamlss-5.R:163`），需字面 `method=mixed(n1,n2)`
  调用；`test_mixed_r_parity.py` 12 例对活 R 4.6.0/gamlss 5.5.0 golden 验
  mu/sigma 系数、deviance、fitted values（差距 mu≤2.2e-5 / sigma≤1.6e-4，
  远严于既有 fit-parity 容差）；清理 `_apply_method_step` 的死代码
  `MIXED→0.75 damped RS` 分支与两处过时 docstring，统一语义；
  44 例既有 MIXED 回归 + golden gate 全过。已知边界 `mixed(0,n2)` 单独处置）
- [x] P0-4 CI 加 lint/type + 8 分片发布门禁（D2.223 完成：`tools/pytest_shards.py` 确定性
  8 分片工具 + `release.yml` 的 `sharded-suite` matrix 发布门禁（独立 DB / tee 实时落盘 /
  静态门禁组 / 单进程性能基准复核）+ `publish-pypi needs` 全门禁化；
  `omnilss/tests/test_release_sharding.py` 钉住）。注：`ci.yml` 早已有 lint/type +
  分层 collection，本批把它们固化进**发布门禁** `pre-release-gate` 并去掉单进程全库跑。
- [x] P0-5 登记表数字与全族流一致 + 子集运行不假警报（D2.225 完成：**验收标准两条
  均已满足**——「D2.47 后 ZIPIG/ZINBI 数字失效」经 2026-10-20 补记**证伪**
  （PIG 走独立核 `omnilss/pig.py`、不经 Bessel），D2.103 全族 146 族 rsdev
  **0 fail**、11-01 逐族逐字段 **0 差异**；D2.83 `_build_jobs` 全族 payload
  缓存已消除子集假警报且 `test_scan_sampling_is_order_stable.py` 钉住。本批补
  `test_rsdev_registry_full_stream.py` 把「子集==全族」不变量从 3 个 `_PROBE`
  族扩到**全部 19 个登记族** + 数字自洽断言。字面「逐族独立播种」是会作废
  19 族证据、需全族流活 R 重测数小时的大重构，历史上一再推迟，按用户决策
  **不做**并校准描述，而非改样本流制造假性完成）
- [x] P0-6 每份覆盖报告显式标注 `gamlss2` out-of-scope（D2.221 完成：`tools/coverage_scope.py`
  单一真源，7 份报告全部带 scope 声明；`test_coverage_scope.py` 钉住；
  `generate_coverage_artifacts.py --date` + `coverage_diff_report.py → OK`）
- [x] P1-1 冻结高频族清单（D2.230 完成：`docs/development/highfreq-families-frozen.json`
  把 `_R_CONSISTENCY_EVIDENCE` 的全部 validated 族分为 **promoted**（有 golden）/
  **pending**（未探）/ **excluded**（已探被拒，每条带 code+note）三层，三层互斥、
  并集等于 validated 集合（计数以工件为准；D2.231 起 ZABI/ZIBI 归 excluded）。`tools/highfreq_family_freeze.py`
  `--check` fail-closed 并接入 `ci.yml`/`release.yml`；`test_highfreq_family_freeze.py`
  8 例钉住分区/同步/门禁接线。注：本项原写的「39 族」按代码实测校正为 40，且不再
  写死计数）
- [x] P1-2 为高频族建立统一 RS fit golden（含收敛诊断 schema）与 newdata/序列化证据
  （D2.229 完成：20 个 golden fixture 统一含 RS-fit 系数/deviance + `convergence`
  {iter,converged,n_cyc} + `newdata` 5 点网格 + `predicted` response 尺度参数；
  `test_r_golden_predict_serialize.py` 61 例：newdata 预测对活 R（rtol/atol=2e-4）、
  `save_model_json`/`load_model_json` 往返预测漂移 ≤1e-10、收敛 schema 一致且两侧均收敛。
  每个 golden 因此覆盖 fit → predict → persist 全链路）
- [ ] P2-1 逐族归类 36 个 rsdev warn / 25 个 dpqr skip / 1 个 deriv skip
- [ ] P2-2 修 BCTO deriv、SHASH 积分、NET clamp、JSU 轨迹、LNO intercept-only
- [ ] P3-1 冻结 family/link/formula/prediction 契约
- [ ] P3-2 实现 `checklink`/`make.link.gamlss`/`as.family`/`gen.Family` 框架
- [ ] P3-3 `*.fix`/`contrasts` 扩到 CG/MIXED/joint + 分类预测重建
- [ ] P5-1 定义 OmniLSS↔ondil distribution contract 与独立流式 golden
- [ ] P6-1 `log_bessel_kv` JAX 原生实现（含精度与耗时双验收）

### 22.6 风险与决策

- **证据风险**：inventory 冒充 coverage、R 不可用时的 skip、静态 null golden —— P0 为硬门禁。
- **数值风险**：x64、JAX 版本、特殊函数、R 与 Python 优化器差异；所有失败必须分类并留判据。
- **范围风险**：146 registry ≠ 146 R family；compat_alias 不计独立迁移；gamlss2 不阻塞 1.0。
- **参考项目风险**：gamlss-python 只作转录 oracle，ondil 只作在线设计 oracle，不改变 R 优先级。
- **性能风险**：benchmark 不得先于 parity；区分 cold/warm、CPU/GPU、R 可用性。
- **发布决策**：生产路由持续 fail-closed；placeholder/alias/experimental 默认不 admission。

> 时间是容量估计而非承诺；R artifact、失败分类或 admission 不齐，功能一律保持 `experimental`。
> 根文档历史「Phase 0–4 完成」只表示 Release 1 阶段目标，不代表全量 R GAMLSS 迁移完成。

### 22.7 进展（2026-11-05 会话，D2.169–D2.170）

- ✅ **P3-2（局部）**：`gamlss.family` / link 框架迁移完成——`make.link.gamlss`、`checklink`、
  `show.link`、`as.family`、`as.gamlss.family`、`gamlss.family`、`gamlss.family.default`
  7 个导出符号（含 `LinkGamlss`、`show.link` 的 118 族允许 link 表）。
- ✅ **核心/dist 工具**：`centileSkew`、`gamlss.scope` 迁移完成（活 R 数值一致）。
- 📉 覆盖：`missing` **391 → 382**（9 个符号离开 missing），matched 689 → 698；ratchet 基线已更新且 `result: OK`。
- 测试：D2.169 41 例 + D2.170 7 例，focused regression 92 例 EXIT=0。
- ⏭ 下一步（按 P 排序）：P0（`.codex/setup.sh`、旧文档校准、MIXED 语义）；P3 平滑项（pbm/pbz/pbo/pbp/pbc/pvc/cy/fp/bfp）
  与 `Gen.Family`；P4 周边包（cens/tr/mx/add）；P5 在线/多元。`o` 变体 dpqr（32 符号）需逐族补 CDF/逆，
  当前 `p/q/r` 多数缺失，不能靠别名冒充。

### 22.8 进展（2026-11-05/06 会话，D2.171–D2.201）

- ✅ **平滑项（P3 主体完成）**：规格构造 `pb/ps/cs/cy`、拟合核 `gamlss.pb/ps/cs/cy`、
  六个变体 `pbm/pbo/pbp/pbz/pbc/pvc`（规格 + 拟合）、`fp/bfp/gamlss.fp`、`pp/gamlss.pp`、
  `pcat/gamlss.pcat`、9 个 `*.control`、`scs`。全部有活 R 数值对照（多数 ≤1e-11，个别路径已记录
  「R 自身容差/不连续」级差异）。
- ✅ **选择/诊断**：`drop1All/add1All`、`stepGAIC.VR`、`ri/gamlss.ri`、`random/gamlss.random`。
- ✅ **周边包**：`gamlss.prepdata` 数据整形族（28+4 符号）、`gamlss.tr` 截断族、`gamlss.cens` 截断核、
  `gamlss.foreach` 的 `getSVD/which.Data.Corr`、`gamlss.mx` 的 `dMX/pMX/MX.control/NP.control`。
- 📉 覆盖：`missing` **382 → 241**（141 个符号离开 missing），matched 698 → 839；ratchet `result: OK`。
- ⚠️ 已知限制（都已写进对应模块 docstring + 测试，不冒充完成）：`pvc(by=...)` 未移植；
  `gamlss.cs` 的 spar/自动 GCV 分支未移植；`random()` 的 xeval 分支未移植；
  `pcat` 在 `Lp≠2` 且给 df 时因 edf(lambda) 不连续与 R 差 ~1.4e-2（两侧行为都已钉住）。
- ⏭ 剩余 241：`gamlss.ggplots` 108（纯绘图）、`gamlss.prepdata` 27（`ACE`/scale/outliers）、
  `gamlss` 核心 15（`gamlss.lo/re`、`stepGAIC.CH`、`stepTGDAll.A`、`TGD`、诊断绘图）、
  `gamlss.add` 15、`gamlss.foreach` 11（`pcr/pc/fitPCR`）、`gamlss.spatial` 9、`gamlss.mx` 8（拟合引擎）、
  `gamlss.dist/tr` 6（绘图/尾部）。
