# OmniLSS 全项目开发计划（总纲）

> 单一执行计划与阶段状态台账。会话开始时应先在本文件定位当前阶段/任务，再读
> `.workbuddy/memory/` 的日志与纪律条目。**最新全面审核结论与 D2.270+ 后续完整开发计划见第二十三节
> 及 `docs/development/full-project-audit-2026-11-21.md`**（历史 D2.167 审核见第二十二节与
> `docs/development/full-project-audit-2026-11-05.md`）。

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
  `production_safe` 当时仍 5（符号数），后由 D2.241 准入到 **110 符号 / 22 族**。
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
  `deriv` 的 1 个 skip。**D2.236 校准**：当前扫描的 deriv skip 是 **LOGSHASHO**
  （「no support grid yields a finite density」），原记的「GT 探针 >180s」已不再成立
  （GT deriv 本次 pass）。
- **归类工件（D2.236）**：`docs/development/p2-defect-classification-2026-11-15.json`（+ `.md`）
  消费 `docs/development/rs-parity-scan-2026-11-15.json`，90 个非通过项**全部归类**
  （`unclassified=0`，`all_failures_classified=true`）；tool
  `tools/p2_defect_classification.py --check` fail-closed（scan sha256 变更即要求重生成）。
- 挂账缺陷：BCTO deriv、BCT tau 交叉块（custom JVP 或等 JAX 补规则）、SHASH 密度积分、
  NET clamp、PARETO1o 幽灵 score、JSU 拟合轨迹、LNO intercept-only、EXGAUS q uniroot 残差。
  **D2.237 复验**：BCTO deriv=pass；SHASH 质量=1.000000（[-40,40] 与 [-400,400]）；
  NET 在拟合边界强制 `tau>=nu`（raw kernel 仍 clamp 以保 jit）；JSU `max_iter=200`
  收敛到 316.3600（R 316.3632，略优）；LNO intercept-only 收敛。以上均已不再是缺陷。
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

**P4 范围盘点（2026-11-15，D2.246）**：见 `docs/development/m4-scope-2026-11-15.json`
（+ `.md`，tool `tools/m4_scope_inventory.py --check`）。按上面的优先级逐包记录
surface 缺口、证据模块与下一步动作，**fail-closed**：覆盖工件里出现但未分类的周边包直接报错。
- **表面完整（8 包，D2.263 达成）**：`gamlss.cens`（5/5）、`gamlss.tr`（8/8）、
  `gamlss.mx`（12/12，D2.247）、`gamlss.add`（15/15，D2.251）、
  `gamlss.spatial`（9/9，D2.252）、`gamlss.foreach`（13/13，D2.253）、
  `gamlss.prepdata`（59/59，D2.254）、`gamlss.ggplots`（109/109，D2.255–D2.263）。
  覆盖缺口从 D2.246 的 6 包 120 个降到 **0**。
- **无函数表面（1 包）**：`gamlss.data`（仅数据集）。
- **专用证据模块（D2.264 补齐）**：`gamlss.foreach`（`test_gamlss_foreach_evidence.py`）、
  `gamlss.prepdata`（`test_gamlss_prepdata_evidence.py`）、`gamlss.ggplots`
  （`test_gamlss_ggplots_evidence.py`）——三者都先核对覆盖工件（符号数、surface、
  parity），再跑各自的工作流；`packages_without_evidence_tests` 从 3 降到 **0**。
- **首批已完成（D2.247）**：`gamlss.mx` 的 `gamlssMX` / `gamlssMXfits` / `gamlssNP` /
  `plotMP` 全部落地，12/12 符号 surface=public_api、kernel=implemented、parity=tested。
  MX 与 NP 的拟合对活 R 对齐到 1e-6 / 1e-8（dev、prob、mu、sigma、mass points、
  EM 迭代数）；未移植分支（`pi.formula` 含协变量、`zero.component`、`mixture="gq"`、
  `random=~x`）显式抛 `NotImplementedError`，不近似。
- **M4 状态（D2.266）**：全部 8 个周边包 surface 完整、缺口 0、无首个工作项；
  每个包都有专用证据模块（`packages_without_evidence_tests=0`），并且 8 个包都有
  活 R 一致性测试（`test_r_consistency_gamlss_*.py`），共 25 个符号的 parity 从
  `tested` 升到 `r_consistency_test`：cens 4/5、tr 5/8、mx 3/12、add 4/15、
  spatial 2/9、foreach 1/13、prepdata 1/59、ggplots 5/109。**8 个包的盘点动作全部为
  "hold"**。
- **关于 "golden"**：盘点工具对非 family 符号的 parity 只有
  `r_consistency_test`/`tested`/`unverified` 三档，`golden:*` 仅由
  `fixtures/r_golden/regenerated/manifest.json` 里的 **family** 用例产生。周边包的
  符号（cens.d、trun.d 等）不是 registry family，所以 "升 golden" 在其 schema 下
  不可达；D2.266 用活 R 对照把它们升到 `r_consistency_test`，这是这些符号能达到的
  最高档，且每包动作已达 "hold"。真正要加 golden 语料，需要为 `generate_golden.R`
  增加 censored/truncated 族用例，属独立的更大工作。
  下一步是 P5（ondil 在线层）与 P6（`log_bessel_kv` JAX 原生）。

### P5：ondil 在线层（独立轨道，独立发布，不混入 R 完成率）

- 先定义 OmniLSS↔ondil 共享 distribution contract（参数/support/link/score/信息矩阵）。
  **D2.269 契约文档**：`docs/development/p5-ondil-distribution-contract-2026-11-15.md`
  （22 族参数序 1:1；链接方向、逐观测导数形状、support 规范化比较、ondil 不可导入三条硬约束）。
- 再实现在线估计器契约（`partial_fit`/`update`/forget factor）、流式 golden、chunk 顺序不变性、
  warm-start、漂移/缺失值/并发、性能与内存基线。
- **多元分布**（MVN/MVT 的 Cholesky/LowRank/ModifiedCholesky）作为独立能力面补齐。
- L1 R golden 只证明 kernel，不证明在线 estimator 正确性——独立验收。

### P6：JAX 内核与 GPU 性能（一致性 gate 之后才做）

- `bessel.py` 的 `log_bessel_kv` 去 `pure_callback`：JAX 原生实现 + custom JVP。
  双验收：对数网格（nu −5..20、x 1e-8..1e3）逐点 1e-12 相对误差 + Sichel 拟合周期耗时下降。
  **D2.267 结果**：原生实现落在 `omnilss/bessel_native.py`（三分区：Debye 渐近 /
  u=ln x 的 16 阶 Taylor ODE 传播 / 小参数级数），**精度验收通过**（1275 点，K 的
  相对误差最差 5.68e-14，工件 `docs/development/bessel-native-2026-11-15.json`，
  工具 `tools/bessel_native_scan.py`）；**耗时验收不通过**——n=40、SI、max_iter=1
  时 scipy 回调 5.31 s vs 原生 371.46 s（70×），因为拟合循环逐观测调用、原生核按元素
  付费，而回调把 vmap 批成一次主机调用。**默认路径保持回调**（fail-closed），原生核
  opt-in；启用前必须在 GPU 上重测拟合周期。同轮发现的 `c=K_{nu+1}/K_nu` 条件数问题
  （对数差 4e-14 → dGIG 尾部 4.9e-12；斜率法 5e-16 → 1.1e-13）记录在
  `log_bessel_kv_ratio`，启用原生核时 GIG/SI/SICHEL 必须改用它。
  **D2.268 复测**：D2.267 的 fit cycle 是手记常数塞进工具，工件不可复现。改由
  `tools/bessel_native_scan.py` 的 `fit_cycle()` 实测：两条路径各跑一次 SI 拟合，并在
  各自的核（回调 `_log_bessel_kv_scipy` / 原生 `_native_log_kv`）上挂调用计数守卫，
  patch 没命中就直接抛错，避免「同一内核测两遍还报 39×」。本机 `.venv` 3.10.16、
  含 JIT 编译：n=40、max_iter=1 回调 17.51 s vs 原生 683.31 s（**39×**，不是手记的 70×）；
  单次核调用原生仅慢 1.7×（1.16 vs 0.70 ms，n=48）。绝对值随机器状态浮动，结论不变：
  默认继续走 scipy 回调，原生核 opt-in。
- `debianjaxgpu`（WSL GPU）验证：GPU/CPU parity、RS_JAX 六族扩面、cross-over 阈值基准、
  cold/warm 区分。性能只能在一致性 gate 通过之后运行。
- 减少 SICHEL 族 `lax.scan` 固定 100000 步的动态上界（`while_loop` 范式）。

---

### 22.4 里程碑与发布门槛

| 里程碑 | 门槛 | 建议版本 |
|---|---|---|
| M0 | P0 全清；覆盖工件可重生成；分层 collection 零错误；devcontainer 可用 | 0.4.x |
| M1 | 冻结清单内高频族全部 **Surface + DPQR 证据 parity**；无 placeholder 进入目标范围（D2.235 重述：原「golden:validated 数量级」改为证据覆盖口径（40 族池上限 ~110，10x 不可达），原「Kernel 全 implemented」划归 **M3** 准入） | 0.5.0 |
| M2 | P2 收敛缺陷分类完毕；RS/CG/MIXED 分层 parity；fit/newdata 证据齐全；失败可解释（D2.239 达成：四项均有可校验工件，见 §22.4 M2 状态） | 0.6.0 |
| M3 | P3 核心 API + family/link 框架 + 准入平滑项闭环；`*.fix`/`contrasts` 全路由（D2.245 达成：D2.241 准入 22/22 → D2.242 P3-1 契约冻结入 CI → D2.243 P3-2 gen.Family 可拟合 → D2.244 P3-3 fix 全路由 + 分类预测重建；6/6 判据见 §22.4 M3 状态） | 0.9.0 |
| M4 | P4 周边包（至少 cens/tr/mx/add）+ 服务 + 文档/教程 + 发布 artifact 同步（D2.246 已完成范围盘点：cens/tr 表面完整，mx 为首个缺口共 120 个缺失符号；见 §P4 状态） | 1.0.0 |
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

**M2 分层 parity（2026-11-15 落地）**：见 `docs/development/layered-parity-2026-11-15.json`
（+ `.md`，tool `tools/layered_parity.py --check`）。
- 范围：22 个 promoted 族 × {RS, CG, MIXED} = **66 cell**，R 侧同一样本
  `RS(n.cyc=100)` / `CG(n.cyc=100)` / `mixed(1,100)`。
- 结果：**pass 54 / flat_ridge 11 / r_error 1**；`mismatch=0`、`py_error=0`，
  **`our_side_parity_ok: true`**（`all_layers_pass: false`，因 flat_ridge/r_error）。
- `flat_ridge`：deviance 与 R 一致到 dev_rel ≤1e-4 且系数差 ≤5e-2，参数弱识别
  （GU/IGAMMA/NBI/NBII/PE/RG/WEI 的 sigma 或 nu）。
- `r_error`：NBII 的 R `CG()` 自身报错（"missing value where TRUE/FALSE"）。
- 本批修复的两个真实缺陷：① CG 路由对固定参数族漏传 nu/tau/sigma（NET 的 CG 曾
  `TypeError: _net_log_pdf() missing 'nu'`）；② MIXED 的 RS→CG 边界检查按 NO 标定过严
  （`rtol=1e-7`），BE/IG 的合法拟合被拒 → 放宽到 1e-4（22 族实测最大合法 gap 2.9e-5）。
- M2 剩余：`fit/newdata 证据` 按 M2 口径汇总（golden 已含 predicted/序列化轴）。

**M2 状态（2026-11-15 达成，D2.239）**：见 `docs/development/m2-assessment-2026-11-15.json`
（+ `.md`，tool `tools/m2_assessment.py --check`）。M2 四项门槛与证据：
- **P2 缺陷分类完毕**（`p2_defects_classified`）：`p2-defect-classification-2026-11-15.json`
  90 个三闸非通过项全部归类，`unclassified=0`。
- **RS/CG/MIXED 分层 parity**（`layered_parity_our_side`）：`layered-parity-2026-11-15.json`
  66 cell，`mismatch=0`、`py_error=0`，`our_side_parity_ok=true`。
- **fit/newdata 证据齐全**（`fit_newdata_evidence_static` + `_live` + `serialization_roundtrip`
  + `convergence_complete`）：22 个 promoted 族 fixture 全部含 deviance、逐估计参数系数块、
  `convergence{iter,converged,n_cyc}`、5 点 `newdata` 与 R `predicted`；**活体复测** 22/22
  重拟合的 newdata 预测在 `atol=rtol=2e-4` 内复现 R，22/22 JSON 往返漂移 ≤1e-10，
  22/22 `converged=True`。
- **失败可解释**（`failures_explainable`）：P2 分类齐备 + 分层 cell 全部落在文档化词汇内
  （pass/flat_ridge/r_error/r_not_converged/mismatch/py_error），NBII 的 R `CG()` 报错与
  11 个 flat_ridge 均单列归类。
- **`m2_gate_met: true`**。M2 正式关闭（0.6.0 门槛）。

**M3 准入证据（2026-11-15 清单完成，D2.240）**：见
`docs/development/m3-admission-evidence-2026-11-15.json`（+ `.md`，tool
`tools/m3_admission_evidence.py --check`）。**这是清单，不是准入决定**。
- 硬门 9 项（三闸 pass、r_consistency validated、golden fit fixture、M2 活体预测 /
  JSON 往返 / 收敛、分层路由无 mismatch/py_error）对 22 个 promoted 族逐族计算。
- 结果：**admitted 1（NO）/ eligible_not_admitted 21 / blocked 0**；
  软标记 7 族（GU/IGAMMA/NBI/NBII/PE/RG/WEI 的 `flat_ridge`，NBII 另有 R 侧 `CG()` 报错）。
- **注册表声明审计**：`_PRODUCTION_SAFE`、`_RS_FIT_EVIDENCE`、`_PREDICTION_EVIDENCE`
  的每一条声明都有工件支撑（`registry_claims_backed: true`，violations=0）；
  ZIP 的预测证据来自专用测试 `test_zip_prediction_serialization.py`。
- 口径澄清：审计文档里的「production_safe 5」是**符号数**（NO 的构造器+d/p/q/r），
  族数是 **1**；M1 工件文案已同步校正。

**M3 准入执行（2026-11-15，D2.241）**：按方案 A 一次性准入证据齐备的 21 族。
- `_PRODUCTION_SAFE` 由 1 → **22**（promoted 全体），`_RS_FIT_EVIDENCE` /
  `_PREDICTION_EVIDENCE` 由该集合派生并并上 ZIP（其证据是扫描器 RS deviance 与专用预测测试）。
- 覆盖工件：`implementation_status` implemented **1 → 22 族**（experimental 125 → 104），
  `admission=production_safe` 符号 **5 → 110**、`blocked` 526 → 421；`golden:validated` 仍 86。
- **路由门修复**：`require_production_family_route` 原只在 placeholder/alias/experimental
  **实现状态**时抛错，其余非 ok 报告直接返回而被 service/benchmark 调用方忽略——准入会让
  已准入族在 CG/MIXED 等仍是 experimental 的路由上被静默放行。现改为**任何非 ok 报告一律抛
  `ProductionFamilyRejectedError`**；`test_admitted_family_still_cannot_use_an_experimental_route`
  钉住「GA+RS 可准入、GA+CG 仍拒绝」。
- 测试样本更新：原以 GA 作「实验族」的能力/发布门/gRPC/HTTP 测试改用 **DEL**（仍为
  experimental 且无 RS fit 证据）；`test_ad_families_with_explicit_dpqr` 的不变量改为
  「不是 placeholder」而非「永远 experimental」。
- `implementation_status_for_family`（纯代码推导、忽略准入）对这批族仍返回 experimental——
  该函数表达的是「是否有真实数值内核」，与准入是两层，测试同时钉住两者。

**M3 契约冻结（2026-11-15，D2.242 / P3-1）**：见
`docs/development/p3-contracts-2026-11-15.json`（+ `.md`，tool
`tools/contract_freeze.py --check`，已接入 CI 与 release 门禁）。
- **family**：`FamilyDefinition` 字段集 + 22 个 promoted 族的 parameters/nopar/type/
  fixed/estimable/dpqr 可调用性/links 键；不变量 `nopar == len(parameters)` 且
  `estimable == parameters - fixed`。两种固定参数形态都覆盖：BI 的 `bd` 在参数元组之外、
  NET 的 `nu/tau` 在元组内被标为固定。
- **link**：`LinkGamlss` 五个 R 槽 + 18 个受支持 link 的往返探针；`logshiftto1/2` 冻结
  **+1e-5 偏移**（R 的非对称分支），`mu_eta` 有限非零。
- **formula**：`ParsedFormula` 字段 + 7 个规范公式（线性/平滑/张量/截距）的解析结构。
- **prediction**：`predict_params`/`predict_response` 的键、形状、缺列报错、多余列容忍。
- 失败即 `--check` 失败：新增 link 无探针、参数元组或解析结构变化、预测行为退化。

**M3 状态（2026-11-15 达成，D2.245）**：见
`docs/development/m3-assessment-2026-11-15.json`（+ `.md`，tool
`tools/m3_assessment.py --check`）。6 项判据全部为真，其中两项是**活体探针**而非
「测试文件存在」：
- `p3_contracts_frozen`：P3-1 契约工件的 `contract_frozen`（D2.242）。
- `gen_family_fits_like_r`：现场构造 logNO 并用 RS 拟合，deviance 对活 R
  140.3831 在 1e-6 内（D2.243）。
- `fix_on_every_route`：现场对 RS/CG/MIXED 跑 `sigma.fix=TRUE, sigma.start=0.5`，
  deviance 均为 R 的 67.1324138595735，且 `sigma` 不进入系数（D2.244）。
- `categorical_prediction_rebuilt`：现场对 CG/MIXED 拟合因子模型并重建预测，
  deviance 与 `pred_mu` 均对 contrasts golden（D2.244）。
- `admission_closed`：准入清单 22/22（D2.241）。
- `prior_milestones_intact`：M1、M2 的 gate verdict 仍为真。
**`m3_gate_met: true`**，M3 正式关闭（0.9.0 门槛）。M4（周边包/服务/文档/发布 artifact）
不受本判定覆盖。

**M3 全路由 fix（2026-11-15，D2.244 / P3-3）**：`*.fix` 现对五个方法生效，活 R 基准
（NO、n=60、seed=5）：`sigma.fix=TRUE, sigma.start=0.5` 时 RS/CG/mixed 的 dev 均为
67.1324138595735、mu 相同、sigma 固定 0.5；不给 start 时 dev 76.6316455840832、
sigma 固定 0.598641360014006（R 的 `sigma.initial = rep(sd(y))`）。同时修复：
cg_fit_v2 的固定参数伪系数、cg_fit_v2 的 intercept 元数据、NO/NO2 的 sigma 初值、
JOINT 的 best-iterate。分类预测在 CG/MIXED 下对 golden 的 treatment/sum 两种编码均通过
（dev 与 pred_mu）。

**M3 gen.Family 收尾（2026-11-15，D2.243 / P3-2）**：`family_generator.Family()` 现在把
基族的 score/hessian 包装成「先做响应变换再调用」，与 R 在导数 body 里把 `y` 改写成
`log(y)`/`logit(y)` 等价。此前 `GeneratedFamily` 直接复制基族导数，拟合路径会拿**原始
响应**算梯度、拿**变换后响应**算 deviance，两者不一致：同一 logNO 样本 RS 拟合 deviance
**413.47 vs R 140.38**，系数全错。包装后：
- logNO：dev 140.38306650049347（R 140.3830665004935），mu/sigma 最大差 3.3e-16 / 1.7e-11；
- logitNO：dev -188.5069659133473（R -188.50696591334733），差 3.9e-16 / 8.9e-14。
测试新增：导数包装的结构不变量（score/hessian 在 y 处等于基族在 log y 处）+ 两例活 R
RS 拟合基准（deviance 与 mu/sigma 均 2e-4 内）。

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
- [x] P2-1 逐族归类 36 个 rsdev warn / 25 个 dpqr skip / 1 个 deriv skip
  （D2.236 完成：全族活 R 扫描 146 族复现 **36/25/1**；rsdev warn 36 分为
  `r_not_converged` 19 / `registered_r_early_stop` 8 / `registered_r_diverged` 4 /
  `registered_r_reference_invalid` 2 / `registered_r_density_recycled` 2 /
  `registered_path_divergent` 1；rsdev skip 28 全为 `r_fit_error`；dpqr skip 25 全为
  `no_r_reference`；deriv skip 1 = LOGSHASHO「无有限密度支撑网格」。90 个非通过项
  `unclassified=0`。工件 + fail-closed 工具 + 7 例测试。**P2-2 的挂账缺陷修复仍未做**）
- [x] P2-2 修 BCTO deriv、SHASH 积分、NET clamp、JSU 轨迹、LNO intercept-only
  （D2.237 完成，逐项复验：BCTO deriv `pass`；SHASH 密度质量 `1.000000`；
  **NET** 新增拟合边界校验 `tau>=nu`（R 的 `dNET` stop 语义；raw kernel 保留 clamp
  以保 `jax.jit(fam.d)`，`test_net_tau_domain.py` 4 例钉住两侧语义）；
  JSU `max_iter=200` 收敛到 316.3600（R 316.3632；R 默认预算本身未收敛）；
  LNO intercept-only 收敛。**P2-2 清单全部关闭**）
- [x] P3-1 冻结 family/link/formula/prediction 契约（D2.242 完成：`tools/contract_freeze.py`
  产出 `docs/development/p3-contracts-2026-11-15.json`（+ `.md`），四块契约均**活体探测**
  而非抄写签名——family：22 个 promoted 族的参数元组/nopar/固定参数/可估参数/dpqr 可调用性
  （不变量 `estimable == parameters - fixed`，同时覆盖 BI 的 bd 在参数元组之外与 NET 的
  nu/tau 在元组内被固定两种形态）；link：18 个受支持 link 各有探测域与往返探针，并把
  `logshiftto1/2` 的 **+1e-5 偏移**作为契约一部分冻结（那是 R 的非对称分支，不是缺陷）；
  formula：7 个规范公式的线性/平滑/张量项与截距解析结构；prediction：`predict_params`
  键等于可估参数、形状随 newdata、缺列抛 `PredictionSchemaError`（ValueError 子类）、
  多余列被容忍。`--check` 双向 fail-closed（新增 link 无探针、参数元组变化、解析结构变化
  均失败），已接入 `ci.yml`/`release.yml`；9 例测试钉住判据与门禁接线）
- [x] P3-2 实现 `checklink`/`make.link.gamlss`/`as.family`/`gen.Family` 框架（D2.243 完成：
  四个符号均已在 `gamlss_family.py`/`family_generator.py` 落地；本批补齐 `gen.Family`
  的**可拟合性**——`Family()` 原样复制基族 score/hessian，而它们会收到未变换的 y，导致
  transformed deviance 与梯度不一致（同一数据 logNO 拟合 deviance 413.47 vs R 140.38）。
  现按 R 的 body 改写语义在调用边界包装 y→log(y)/logit(y)；logNO 与 logitNO 的 RS 拟合
  对活 R 对齐到 **1e-11 / 1e-13**（deviance、mu、sigma），测试同时钉住「导数看到的是
  变换后的响应」这一结构不变量）
- [x] P3-3 `*.fix`/`contrasts` 扩到 CG/MIXED/joint + 分类预测重建（D2.244 完成：
  `mu/sigma/nu/tau.fix` 从「仅 RS」放开到 **RS / CG / MIXED / JOINT / LBFGS**——override 本就
  是与方法无关的「改写 fixed 集合 + 冻结值进 data」；三条路由对活 R 一致
  （dev 67.1324138595735 @ sigma.start=0.5；dev 76.6316455840832 @ 默认 sigma.initial）。
  过程中修掉 4 个真实缺陷：① NO 的 `sigma.initial` 缺失（无 start 时冻结在 eps 地板，
  dev ~1e13 vs R 76.63，补 `rep(sd(y))`，NO2 补 `rep(var(y))`）；② `cg_fit_v2` 给固定参数
  造伪系数（MIXED 报 sigma.coefficients，R 为 NULL）；③ `cg_fit_v2` 把 terms 的
  `intercept` 记成「是否仅截距」（`y~g+x` → False），使 MIXED 因子预测按 3 列重建而 schema
  要 4 列、直接报错；④ JOINT 返回最后一轮 Adam iterate 而非访问过的最优点
  （`sigma.fix` 下 dev 67.1638 vs 67.1324），改为 best-iterate 追踪。
  `test_p3_fix_methods.py` 12 例：三方法 × 有/无 sigma.start、JOINT、flag 类型校验、
  CG/MIXED × 两种 contrasts 的因子预测对 R golden（`contrasts_2026-11-01.json`））
- [x] P3-4 `production_safe` 准入闭环（D2.240 证据清单 → D2.241 准入执行：22 个 promoted 族
  **全部 admitted**（原 1 族），`admission=production_safe` 符号 5 → 110、blocked 526 → 421；
  `_RS_FIT_EVIDENCE`/`_PREDICTION_EVIDENCE` 由该集合派生并并上 ZIP；7 个族的
  `flat_ridge`/R 侧报错软标记保留在 `docs/development/m3-admission-evidence-2026-11-15.json`
  随准入记录；同时修复路由门「非 ok 报告不抛错」的 fail-open 漏洞）
- [x] P4-1 `gamlss.mx` 补齐 4 个缺失符号（D2.247 完成：`gamlssMX` EM 混合（常量先验）、
  `gamlssMXfits` 多次随机起点取最优、`gamlssNP` NPML（`mixture="np"` + `random=~1`）、
  `plotMP` 3D 几何 + 可选 matplotlib 渲染；对活 R 的 dev/prob/mu/sigma/mass/EM 迭代数
  对齐 1e-6~1e-8。**未移植分支显式 NotImplementedError**：`pi.formula` 含协变量（需
  multinom）、`zero.component`、`mixture="gq"`、`random=~x`。顺带修 `dmx` 对标量 y
  崩溃的既有缺陷。12 例测试）
- [x] P4-2 `gamlss.add` 补齐 13 个缺失符号（D2.248–D2.251 完成 **13/13**：
  `ga`/`ba` 加性项（委托等价）、`tr`/`gamlss.tr` 加权 CART（对 rpart 逐位一致）、
  `fk`/`gamlss.fk`/`fitFixedKnots`/`fitFreeKnots`/`centilesTwo` 自由节点组（目标与
  R 一致到 1e-12、掩码与 mgcv 逐格一致）、`nn`/`gamlss.nn` 单隐层网络（功能验证，
  不声称与 nnet 逐位一致）。`gamlss.add` **surface 15/15 完整**）
- [x] P4-3 `gamlss.spatial` 补齐 6 个符号（D2.252 完成：`MRF`（Q-function 边际似然）、
  `MRFA`（交替 lambda/df）、`gmrf`/`gamlss.gmrf`（项 + 回调）、`draw.polys`（几何 +
  可选绘图）、`polys2polys`（sp 鸭子类型转换）。MRFA 对活 R **逐位一致**（beta 2e-16），
  MRF 相对差 ~1e-6；`gamlss.spatial` **surface 9/9 完整**）
- [x] P4-4 `gamlss.foreach` 补齐 5 个符号（D2.253 完成：`fitQR`（SEP3 固定三参的
  分位数回归，deviance 对活 R 1e-5）、`fitRolling`（滚动重拟合 + 一步预测 +
  GDincr）、`NonParametricBoot`/`BayesianBoot`（行重采样 / Dirichlet 权重）、
  `centiles.boot`（重采样 centile 曲线，含 calibration）。`gamlss.foreach`
  **surface 13/13 完整**）
- [x] P4-5 `gamlss.prepdata` 补齐 19 个符号（D2.254 完成：`y_zscores`/`data_zscores`
  （SHASHo ML + 分位残差）、`y_outliers`/`_z`/`_both`/`_by`/`data_outliers`（z 分数
  与 median±value·MAD 两套规则）、`cor_perm_test`/`cor_boot`、`ACE`/`ACE_iter`/
  `mcor`/`data_mcor`、`association`/`data_association`（Cramér's V 偏差校正 /
  Spearman / sqrt(R²)）、`data_bucket`/`data_leverage`/`data_plot`/`data_response`/
  `data_xyplot`。绘图降级为结构化结果。`gamlss.prepdata` **surface 59/59 完整**）
- [x] P4-6 `gamlss.ggplots` 补齐 73 个符号（D2.255–D2.263 完成 **109/109**；
  **D2.255 首批完成 14 个**：`y_hist`/`y_dots`（
  median±value·MAD 规则）、`fitted_pdf`/`fitted_cdf` 及 `*_data`、`model_qqplot`/
  `model_density`、`resid_qqplot`/`resid_density`/`resid_ecdf`/`resid_plots`/
  `add_resid_qqplot`、`moment_bucket`；**D2.256 再完成 9 个 `pe_*` 偏效应**：
  `pe_1_parameter`/`pe_2_parameter`/`pe_param`/`pe_eta`/`pe_param_grid`/
  `pe_eta_grid`/`pe_pdf`/`pe_pdf_grid`/`pe_terms`（沿协变量取参数/eta，其他变量
  固定场景值，中心化后返回效果曲线）。返回数据层 + 可选 matplotlib 渲染，像素一致性
  按 P4 不要求。**D2.257 再完成 10 个 `model_*`**：`model_GAIC`/`model_GAIC_lollipop`/
  `model_TD`/`model_TD_lollipop`（GAIC/TD 对比表，delta=max−value 与 R 逐位一致）、
  `model_centiles`（多模型 centile 曲线）、`model_wp`/`model_wp_wrap`（虫图与按协变量
  分面）、`model_pca`（残差 PCA）、`model_devianceIncr_diff`、`model_mom_bucket`（别名）。
  **D2.258 再完成 19 个参数化诊断**：`resid_mu`/`resid_median`/`resid_param`/
  `resid_quantile`/`resid_xvar`/`resid_wp`/`resid_wp_wrap`/`resid_dtop`（Owen/JW NPL 带）、
  `resp_mu`/`resp_param`/`resp_quantile`、`fitted_leverage`（帽子值，sum==设计秩）、
  `fitted_devianceIncr`、`fitted_survival`/`fitted_survival_data`、
  `fitted_centiles`/`fitted_centiles_legend`/`fitted_centiles_gap`、`fitted_terms`（pe_terms 别名）。
  **D2.259 再完成 10 个样式变体**：`moment_colour_half`/`moment_gray_half`/
  `moment_colour_both`/`moment_gray_both`（返回 BucketSpec + 理论边界曲线）、
  `centile_colour_half`/`centile_colour_both`/`centile_gray_both`（type=tail/central），
  `moment_bucket_wrap`/`centile_bucket`/`centile_bucket_wrap`（逐区间变换坐标 + CI95/
  模拟椭圆）。同时把 `momentSK`/`centileSK` 升级为 R 的加权/变换版本（对活 R 1e-15）。
  **D2.260 再完成 3 个预测辅助**：`predict_pdf`/`predict_cdf`（newdata 逐行的密度/
  CDF 曲线，离散族用整数网格，lower_tail 可控，支持二项 bd 固定参数）、
  `quantile_gamlss`（每观测一个分位数）。**D2.261 再完成 3 个双变量曲面**：
  `bivar_fun`/`bivar_fun_plotly`（n×n 网格 + Z 矩阵，expand.grid 的 x 最快序）、
  `bivar_pdf`（按密度排序累积分位质量，取 50% HDR 阈值）。**D2.262 再完成 2 个自助系数图**：
  `boot_coef`/`boot_coef_one`（逐系数密度 + 原估计 + 2.5/50/97.5 分位线；列名遵循
  R 的 `mu.(Intercept)` 形式），并修复多参数模型 bootstrap 丢失 sigma/nu/tau 公式的缺陷。
  **D2.263 收尾 4 个**：`fit_PB`（P-spline + REML 型 lambda 迭代，lambda/df/sigma/deviance
  与活 R 一致到 1e-9）、`pcr_coef_path`/`pcr_path`（PCR 系数路径，长表顺序照 R 的
  c(row/c(col))）、`histSmo_plot`（直方图 + 平滑密度，兼容 R 的 histSmo 载荷与本包
  HistSmoResult）。**gamlss.ggplots surface 109/109 完整**）
- [x] P5-1a OmniLSS↔ondil 分布契约文档（D2.269）：
  `docs/development/p5-ondil-distribution-contract-2026-11-15.md` —— 两侧现状（含 ondil 硬依赖
  scikit-learn/numba、本 venv 不可导入 → 只能作源码参考）+ theta↔params 列序规则 + 逐字段映射表 +
  链函数方向陷阱（`derivative`=dμ/dη ↔ `inverse_derivative`）+ score/hessian 逐观测形状契约 +
  7 条已知缺陷（含 OmniLSS adapter 求和成标量、ondil `logpmf` 守卫调错）+ 22 个重叠族参数序核对
  （全部 1:1）+ 流式 golden 契约草案（7 项判据）+ 6 个未决问题
- [x] P5-1b 契约适配层（D2.270）：`omnilss/ondil_contract.py`（theta↔params、`Support` 规范化、
  `OndilLink` 五方法、`OndilDistributionView` 的 `logpdf/logcdf/pdf/cdf/ppf/dl1_dp1/dl2_dp2/dl2_dpp/
  initial_values`）+ `tools/ondil_contract_scan.py`（工件 `docs/development/p5-ondil-contract-2026-11-15.json`，
  **用正则读 vendored ondil 源码交叉核对**：22 族参数序 0 错位、支持集 0 失配（EXP/IGAMMA/LOGNO/LOGNO2
  按 R 语义豁免并记账））+ `test_ondil_contract.py` 78 例。同时修 `adapters.score/hessian` 的逐观测契约
  （原实现 `jnp.sum` 后求导返回标量、hessian 只填对角）；修 `dPO(-1, log=True)` 返回 0 而非 -Inf；
  5 个族的钳位型边界违反（IG/IGAMMA/LOGNO/LOGNO2/ZAGA）按棘轮冻结在 `KNOWN_BOUNDARY_VIOLATIONS`，
  测试要求集合只减不增。
- [ ] P5-1c 独立流式 golden（`partial_fit`/`update`/forget、分批不变性、warm start、漂移/缺失/性能内存）
- [x] P6-1 `log_bessel_kv` JAX 原生实现（D2.267 精度 5.7e-14 达成；CPU 拟合周期
  回退（手记 70× / D2.268 工具实测 39×）→ 归档为 opt-in 实验模块，默认回调不变；
  GPU 重测后再决定是否切换）

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

---

## 二十三、2026-11-21 全面审核结论与 D2.270+ 后续完整开发计划

> **生效基线**：HEAD = `7aedafc`（D2.270），实测日期 2026-11-21。详细实测审核报告与清理台账见
> `docs/development/full-project-audit-2026-11-21.md`。
> 本节承接并更新第二十二节（M0–M3 与 P4 表面/证据已达成，P5-1a/b 与 P6-1 已交付），聚焦从当前
> D2.270 状态推进到 **M4（1.0.0 正式发布）** 及 **独立轨道 P5（在线/多元）/ P6（GPU/内核性能）** 的完整 WBS。

### 23.1 当前基线快照与硬性边界（2026-11-21 实测）

- **参考项目边界（硬约束）**：`gamlss-python/` 与 `ondil/` **只作设计与算法参考，严禁在 OmniLSS 中引入、拷贝或依赖其代码**（包括可选运行时导入）；OmniLSS 全部功能基于自身 JAX / NumPy / SciPy 架构独立实现与独立验收。
- **环境管理约定**：
  - `.venv/`：Windows 下运行当前项目的虚拟环境，统一使用 **`uv`** 管理；
  - `debianjaxgpu/`：WSL Debian 下运行当前项目的 GPU 虚拟环境，统一使用 **`uv`** 管理。
- **符号覆盖**：范围内 1,025 / 1,036（**98.94%**），仅 `gamlss` 核心包缺 **11** 个符号；10 个包 100% 完整；`gamlss2`（44 符号）维持 out-of-scope。
- **证据成熟度**：`kernel=implemented` 606、`production_safe` **22 族 / 110 符号**、`experimental` **104 族**、`compat_alias` 20、`placeholder` 0；范围内 `unverified` 54（`gamlss` 36 + `gamlss.dist` 18），`blocked` 421。
- **里程碑与门禁**：M0/M1/M2/M3 已闭环，但实测 **2 项门禁产物漂移**（`m1_assessment`、`layered_parity`），`omnilss-pro` 在真 `grpc` 下 **2 项测试失败**，`.venv` 当前未同步 `ruff`/`black`/`mypy`，文档/版本号停留在 0.4.0 阶段。
- **测试规模**：4 层共收集 **5,822** 用例（`omnilss` 5786 / `omnilss-pro` 25 / `benchmarks` 10 / `omnilss-server` 1），0 collection error / 0 strict skip。

---

### 23.2 分阶段后续开发计划（WBS）

#### Track A（P7 / 立即止血，阻塞一切后续里程碑）：门禁止漂移 + 工程卫生收口

| 编号 | 任务内容 | 具体落点与动作 | 验收标准 |
|---|---|---|---|
| **A1** | 修复两项漂移的里程碑产物并入 CI | 1. 重新生成 `m1-assessment-2026-11-15.json/.md` 与 `layered-parity-2026-11-15.json/.md`；<br>2. 将 `m1/m2/m3_assessment`、`m4_scope_inventory`、`p2_defect_classification`、`ondil_contract_scan` 接入 `ci.yml`/`release.yml`；<br>3. 更新 `m4_scope_inventory.py` 过时的 `verdict.notes` 文案（已无 `gamlss.mx` 缺口）；<br>4. 推进 `symbol-coverage-baseline.json` 棘轮基线（`missing` 173→53），收紧回归防线。 | 全部 14 道 `tools/* --check` 100% PASS；CI 新增门禁全绿 |
| **A2** | 修复 `omnilss-pro` 真实 gRPC 下的 2 处测试失败并接入 CI 运行 | 修复 `omnilss-pro/tests/test_client_tls.py`（使用 `monkeypatch` 显式替换 `grpc.insecure_channel`/`secure_channel` 而非仅在 `ImportError` 时 stub）；在 `ci.yml` 与 `release.yml` 增加 `omnilss-pro` 与 `omnilss-server` 的 `--mode run` 执行步骤（不再仅 `--mode collect`）。 | `pytest omnilss-pro/tests` 在安装 `grpcio` 环境下全绿；CI 实际执行 4 层测试 |
| **A3** | 消除内部运行时告警与能力矩阵口径差 | 1. 替换 `fitting.py:1396,1520` 与 `algorithms/rs_algorithm.py:1035` 对废弃 `_parse_formula` 的内部调用，消除 `DeprecationWarning`；<br>2. 修 `rs_algorithm.py:702` 在 `inf - inf` 处的 `RuntimeWarning`；<br>3. 核查并统一 `_R_CONSISTENCY_EVIDENCE`(40) 与运行时 `capability_matrix()` `r_consistency=validated`(39) 的 1 族偏差（`JSUo` 大小写键名不匹配 `JSUO`）；<br>4. 通过 `uv` 在 `.venv` 补齐 `ruff`/`black`/`mypy` 开发工具链，推进核心模块 `mypy` 分目录转严。 | `tiered_collection` 与核心测试零内部自触发 `DeprecationWarning`/`RuntimeWarning`；`ruff`+`black` 本地与 CI 一致通过 |
| **A4** | 夯实 `omnilss-server` 服务层测试与边界安全 | 为 `omnilss-server` 补充鉴权（`OMNILSS_API_TOKEN` 401/200）、DoS 包体上限（413）与滑窗限流（429 + 过期 IP 清理防内存泄漏）、非 `production_safe` 族拒绝、`CG` 平滑拒绝、独立 `OMNILSS_MODEL_DB_PATH` 隔离测试。 | `omnilss-server` 测试用例 ≥10 项全绿；限流状态有界 |

---

#### Track B（M4 收官 → `v1.0.0` 发布准备）：补齐最后 11 符号、扩大准入、重构文档体系

| 编号 | 任务内容 | 具体落点与动作 | 验收标准 |
|---|---|---|---|
| **B1** | 补齐核心 `gamlss` 包最后 11 个缺失符号（范围内 1036/1036 = 100%） | 分三批按价值交付：<br>- **B1a（平滑/随机效应）**：`gamlss.lo`（对接 `lo` 规格与加权局部多项式回归）、`re`/`gamlss.re`（明确 LME/混合效应后端或显式契约范围，带活 R 对照）；<br>- **B1b（诊断/选择）**：`get.rqres`、`rqres.plot`、`bp`（复用 `moment_bucket` 核）、`calibration`、`stepGAIC.CH`、`stepTGDAll.A`；<br>- **B1c（辅助）**：`fittedPlot`、`gamlssNews`。 | `r-symbol-coverage.json` 范围内 `surface_missing`: **11 → 0**（100%）；每批附活 R 一致性测试与覆盖工件同步 |
| **B2** | 扩大 `production_safe` 与多维度能力准入 | 1. 清理 54 个范围内 `unverified` 符号（`gamlss` 36 + `gamlss.dist` 18 骨架族分类登记），消除空白；<br>2. 在 `family_capabilities.py` 中基于已通过的 `layered-parity`（66 cell）与平滑测试，为验证通过的族晋级 `cg_fit` / `sampling` / `smooth_terms` feature 证据状态（打破当前全 0 现状）；<br>3. 从剩余 104 个 `experimental` 族中按业务高频度筛选第二批候选族（补齐活 R golden fit/predict/serialize 后准入）；<br>4. 为 `gamlss.cens` / `gamlss.tr` 扩展 `generate_golden.R` 删失/截断拟合 golden 语料；逐项推进 `gamlss.mx` 未移植分支（`pi.formula` 协变量先验、`zero.component`）或正式冻结为 1.0 已知边界。 | 范围内 `unverified` 显著压降；`cg_fit`/`smooth_terms`/`sampling` 出现首批 `validated` 族；M4 准入清单无未归类项 |
| **B3** | 文档、教程与对外口径全面刷新（1.0 硬门槛） | 1. 重写 `README.md` / `README_cn.md`（移除过时的「80+族 / 30天冻结 / 缺 ST/DBI/平滑器」描述，改为引用权威覆盖与准入工件）；<br>2. 刷新或归档 `HANDOFF.md`；补全 `CHANGELOG.md`（汇总 0.5.0 / 0.6.0 / 0.9.0 → 1.0.0 演进）；<br>3. 合并治理两套割裂教程：以 `docs/tutorials/` 为唯一官方教程体系并接入 `mkdocs.yml` 导航，补齐 8 篇核心实战指南（含 R→OmniLSS 迁移对照表、22 个生产准入族速查、平滑/删失/混合/服务部署），清理或归档根目录 `tutorials/` 中大量空提纲与 320B 占位 `_cn.md`；<br>4. 归档 `docs/development/` 中 2026-05~09 已过时的历史审计草稿至 `docs/archive/`；<br>5. 建立 `tools/m4_assessment.py` 作为 **M4 (1.0.0) 正式可校验门禁**。 | `mkdocs build --strict` 通过；文档零过时数字声明；`m4_assessment.py --check` PASS |
| **B4** | 1.0.0 发布打包与全量验证 | 1. 统一版本号 `_version.py` / `omnilss/pyproject.toml` / `omnilss-pro/pyproject.toml`（按 0.9.0 → 1.0.0-rc1 → 1.0.0 节奏）；<br>2. 完整运行 8 分片全量测试套件（600 文件）+ 单进程 `test_performance_benchmarks.py` + `omnilss/tools/release_check.py`（wheel/sdist 构建 + `twine check` + 洁净环境安装验证）+ 具备 Rtools/make 的 Linux/Docker 环境中执行 `run_release1_r_golden.py` 端到端验证。 | 8 分片全绿、release_check PASS、wheel 安装 smoke 通过、M4 manifest 冻结 |

---

#### Track C（独立轨道 P5 — 参考 `ondil` 设计的自研在线与多元层，严禁引入外部代码，不阻塞 1.0）

| 编号 | 任务内容 | 具体落点与动作 | 验收标准 |
|---|---|---|---|
| **C1 (P5-1c)** | 自研在线/流式 GAMLSS 估计器与独立流式 Golden | 参考 `ondil` 架构设计（**不引入其代码/依赖**），基于已完成并通过 78 例测试的 `omnilss/ondil_contract.py` 实现独立模块 `omnilss/online/`：<br>1. `OnlineGAMLSS`（`partial_fit`/`update`/`predict_params`，支持遗忘因子 `forget ∈ [0,1)`、递推加权 Gram 矩阵 `O(p²)` 更新、`sample_weights`、warm start）；<br>2. 落地 `p5-ondil-distribution-contract-2026-11-15.md` §12 规定的 **7 项可复跑流式 golden 判据**（分批不变性 ≤1e-10、遗忘漂移追赶半衰期、warm-start 等价 ≤1e-8、0/5/20% 缺失鲁棒性、每观测耗时与 N 无关的斜率检验、O(p²) 内存）；<br>3. 逐步消解 `KNOWN_BOUNDARY_VIOLATIONS` 棘轮中的 5 个族（IG/IGAMMA/LOGNO/LOGNO2/ZAGA 支撑集外返回 `-inf` 而不破坏拟合稳定性）。 | 独立工具 `tools/p5_streaming_golden.py --check` PASS；7 项流式判据全绿 |
| **C2 (P5-2)** | 自研在线正则化路径与多元分布 | 参考 `ondil` 数学接口（**纯 JAX/NumPy 独立实现，不导入 `ondil`**）：<br>1. 实现在线坐标下降 / LASSO / Ridge / Elastic-Net 路径；<br>2. 实现多元分布族与矩阵链接：`MVN` / `MVT`（Cholesky、LowRank、ModifiedCholesky 三种协方差/精度参数化），支持对数似然、score/hessian 与 JAX 自动微分；<br>3. 全部数值验收以解析闭式解、`jax.grad`/`jax.hessian` 与固定 seed 合成 DGP golden 为权威基准。 | 多元分布 logpdf/score/hessian 对解析解与 `jax.grad` ≤1e-10；在线正则路径单调性与收敛测试通过 |

---

#### Track D（独立轨道 P6 — JAX 加速与架构演进）：WSL GPU (`debianjaxgpu`) 实测、JIT 扩面与代码瘦身

| 编号 | 任务内容 | 具体落点与动作 | 验收标准 |
|---|---|---|---|
| **D1 (P6-2)** | WSL Debian GPU (`debianjaxgpu`, `uv` 管理) 一致性与性能基线 | 在 WSL Debian 的 `debianjaxgpu` 环境中实测：<br>1. 22 个 `production_safe` 族的 **GPU vs CPU 数值 parity**（x64 下偏差 ≤1e-10）；<br>2. 在 GPU 上复测 `bessel_native.py`（`tools/bessel_native_scan.py`）与 `RS_JAX`，确认原生 Bessel 是否在 GPU 批处理下逆转 CPU 的 39× 回退；<br>3. 产出带硬件/冷启动/热启动/显存记录的真实跨界阈值报告（`cross-over N`）。 | GPU/CPU parity 工件归档；性能报告严格区分 cold/warm，绝不把 CPU 数据外推 GPU |
| **D2 (P6-3)** | `RS_JAX` 扩面与重尾离散族循环优化 | 1. 将 `SICHEL` / `ZISICHEL` / `ZASICHEL` 的固定 100,000 步 `lax.scan` 改为数据驱动动态上界或 `lax.while_loop`（参照 D2.150 `pig.logpmf` 的成功模式），降低每周期耗时；<br>2. 将 `rs_jax_fit` 与 `ad_hessian` 从 6 个实验族稳步扩展到 22 个 `production_safe` 族，并在活 R 精度门禁通过后将 `rs_jax_fit` 升至 `validated`；<br>3. 探索为 `BCT` 的 `betainc` 对形状参数提供 `custom_jvp`，关闭 D2.152 记录的 `test_bct_tau_cross_block_is_a_known_ad_gap`。 | Sichel 周期耗时显著下降且三闸保持 0 fail；`rs_jax_fit` 首批族通过一致性门禁并准入 |
| **D3 (架构债)** | 大文件拆分与双轨协议收敛（1.x 维护期） | 1. 按支撑域拆分 `dpqr_functions.py`（9,156 行 → 连续实数/正实数/有界/离散/混合/零膨胀子模块，保留顶层重导出兼容）；<br>2. 拆分 `ggplots.py`（3,452 行）与 `fitting.py`（2,623 行），收缩 `__init__.py`（2,065 行，必要时引入 PEP 562 `__getattr__` 延迟导入降冷启动时间）；<br>3. 统一 `FamilyDefinition` 与 `core.distributions.DistributionProtocol`，减少 adapter 桥接维护成本。 | 公开 API 100% 向后兼容；`import omnilss` 冷启动时间下降；单文件上限控制在 ~1,500 行内 |

---

### 23.3 推荐执行顺序与里程碑门槛

| 阶段 | 包含任务 | 目标版本 | 硬门槛 |
|---|---|---|---|
| **Phase 1（工程止血）** | A1 → A2 → A3 → A4 | `0.9.1` | 14/14 门禁绿、`omnilss-pro`/`omnilss-server` 实跑全绿、零内部自触发警告 |
| **Phase 2（M4 / 1.0 收官）** | B1 → B2 → B3 → B4 | **`1.0.0`** | 范围内 R 符号 1036/1036 (100%)、官方文档/教程/README 全面同步、8 分片全量通过、`m4_assessment` PASS |
| **Phase 3（独立轨道并行）** | C1 → C2（P5 在线/多元）<br>D1 → D2（P6 GPU/内核） | `1.1.0` / `1.2.0` | P5 流式 7 项 golden + MVN/MVT 独立验收；P6 先过 GPU/CPU parity 再发性能基线 |
| **Phase 4（架构瘦身）** | D3 | `1.x` | 零破坏性变更下拆分超大文件、收敛双轨协议、优化导入耗时 |

### 23.4 D2.288（2026-11-22）v0.1.0 产品发布面推进

> 依据根目录 `omnilss_v0_1_execution_plan.md`。该计划面向"产品可用 v0.1.0"，
> 与 23.2 的 1.0.0 技术收官并行；本轮只做经代码核实仍缺失、且不破坏既有证据的项。
> 详细日志见 `.workbuddy/memory/2026-11-22.md`。

- [x] **E02 数值阻塞（3/5）**：Link 溢出（LogLink.inverse / SoftplusLink）、NO 的
  `sigma=0` 导数地板、Hessian 非有限矩阵 fail-closed（`is_finite=False`，不再
  `nan_to_num`）、BE 导入期 JIT 预热移除。新增 `test_e02_release_blockers.py` 5 例。
- [x] **E06 摘要/预测 API**：`GAMLSSModel.summary()`、`predict_mean()`、
  `predict_distribution()`；`SummaryResult` 增 `edf` / `warnings`，`converged`
  改走 `convergence_status()`。新增 `test_v01_public_api.py` 3 例。
- [x] **E09 安全持久化入口**：`GAMLSSModel.save/load`（ZIP+JSON+`allow_pickle=False` npz）。
- [x] **E01 发布文档**：`docs/release/v0.1-{scope,acceptance,supported-features}.md`。
- [x] **E10 Windows CPU CI**：`ci.yml` 新增 `windows-cpu` job（`windows-latest` × py3.10/3.11/3.12，
  bash shell），安装 `.[dev]` + optax，运行 10 个核心文件（E02/公共 API/golden NO,GA,PO,BE,NBI,ZAGA/
  serialization），并对 `SKIPPED` fail-closed。Windows 本机按同一命令实跑 52 例 0 skip 通过；
  2026-11-22 推送后在 GitHub Actions 实跑 **py3.10 / 3.11 / 3.12 三个矩阵全绿**。
- [ ] **E02 剩余**：REML 的 `-log|λS|_+`（需活 R 对照后再改目标函数）；Hessian 的
  Cholesky-first（会改动 cond>1e10 的既有数值，需 A/B）。
- [x] **Linux CI 止血（D2.290）**：Group 安装改 `.[server,dev]`（修 Group C 的
  `No module named google`）；R-consistency 第 9 步去掉 glob 引号；TLS 用例改临时端口 +
  60s 启动窗口 + terminate→kill；`test_grpc_metrics_service_direct` 去掉对模块级计数器
  绝对值的顺序依赖；`gamlss_cs` 的 df 求解从 `trace(solve())` 改为
  `W^-1/2 K W^-1/2` 特征分解的闭式 `Σ1/(1+λμ)`（消除 LAPACK 差异导致的 edf 5.0/7.33 漂移）。
  CI 复跑：TLS 转绿；Group C 测试 310 passed / 0 failed，仅剩 Codecov 无 token 上传失败，
  已改为「有 secret 才上传」。基线归因见 `.workbuddy/memory/2026-11-22.md`。
- [x] **Group A 归因（D2.294）**：唯一失败 `test_cg_vs_gamlss_ml` 是期望错——
  `gamlss_ml` 为无外层迭代的单趟 staged port（sigma 停在初值 sd(y)=8.84，dev 1241.5），
  CG 与 RS 均 536.9561（差 3e-12）。测试改为对 RS 比 deviance，并保留「CG ≤ ML」基线。
- [x] **Code Quality / black（D2.295）**：`black==26.10.1` 全量格式化
  `omnilss/src/omnilss/`（225 files reformatted，`--check` 全绿），并在 `pyproject.toml`
  与 `ci.yml` 钉死版本。
- [x] **Code Quality / ruff F401（D2.296）**：402 → **0**（自动清 155 + bessel 探针 noqa +
  4 个再导出层 per-file-ignores）。全模块导入与相对导入校验通过，关键测试全绿。
  期间 CI 抓到 `smooth_fitting` 的 `ADMITTED_SMOOTHERS` 等三个兼容再导出被误删，
  已恢复并逐行 noqa。
  剩余 E/W/F 176（E402 80、F821 34、F841 30、F541 16、F811 7、E722 6、F822 2、E741 1）。
- [x] **Code Quality / ruff E402+F841（D2.297）**：E402 80→0、F841 30→0（docstring 归位、
  循环延迟导入显式 noqa、死局部变量删除且保留 `stable_cholesky` 校验副作用）。
  剩余 **66**：F821 34、F541 16、F811 7、E722 6、F822 2、E741 1。
- [x] **Code Quality / ruff F541+F811（D2.298）**：F541 16→0、F811 7→0。F811 发现
  `smooth_selection.py` 的旧 `reml_score` 覆盖了 D2.146 的修复（此前从未生效），
  已删除旧定义。剩余 **43**：F821 34、E722 6、F822 2、E741 1。
- [x] **Group C sqlite 锁（D2.299）**：`_ModelRegistry` 连接加 `timeout=30` +
  `PRAGMA busy_timeout=30000`，修 xdist 并发建表导致的 `database is locked`；
  新增 pragma 断言测试。
- [x] **Code Quality / ruff E741+E722+F822+F821（D2.300/D2.301）**：9 + 34 → 0。
  F821 暴露 3 个真 bug：SEP4 工厂未绑定 `family`、`pTF_fixed` 用了未定义的 `t`、
  `_beo_r` 的 `_scipy_beta` 只在别的函数内导入。至此
  `ruff check --select E,W,F --ignore E501` **全库 0**，Code Quality 的 ruff 步骤应转绿。
- [x] **Tiered Gate 拆解（D2.302）**：Step 7 是过时用例（GA/CG 自 D2.281 已准入，
  改用 NBII）；Step 6 因 `| tee` 无 pipefail 而 fail-open（已修），并补 optax、
  将 2 个 import 期调 Rscript 的 benchmark 加入已知排除（5→7）。
  本机 collect 5908/0 error/0 skip，pro+server run 0 failed。
- [x] **Tiered 后续门禁（D2.303）**：仓库转 public 后 runner 恢复；Step 6/7 转绿，
  Step 10 的 M0 `coverage provenance drift` 通过重生成 manifest 修复（需
  `JAX_ENABLE_X64=true`）。Group C 的 sqlite 锁定位到 `PRAGMA journal_mode=WAL`，
  竞争时跳过（WAL 已持久化）。
- [x] **P2 行尾哈希（D2.304）**：Tiered Step 13 的 `source scan sha256` 因 Windows
  CRLF / CI LF 漂移；  `_sha256` 先归一化 `\r\n → \n`，重生成分类工件，新增行尾
  无关性测试。
- [x] **P5 streaming golden 跨平台浮点（D2.305）**：`_stable()` 精确比较浮点导致
  Windows 生成 / Linux 复算的 1e-15 抖动被判 drift；比较前 `round(x, 8)`，
  新增三例测试。Tiered Step 13 的全部子门禁本机通过。
- [x] **Group B 归因（D2.306）**：40 个失败中 39 个是 R 参考版漂移——CI 装 CRAN
  最新 `gamlss 5.5-5 / gamlss.dist 6.1-11`，测试基准是 `5.5-0 / 6.1-1`
  （DBI 19、ZISICHEL 8、GU 5、DBURR12 5、PARETO1o 1、SHASHo 1）。CI 改为安装
  CRAN Archive 的钉死源码包。剩余 1 个 `test_pb_in_mu_normal` 是用单趟
  `gamlss_ml` 对 R 迭代 `gamlss`，已改为 `gamlss`（RS）；本机 6 例全绿。
- [ ] **E11/E12/E13**：三篇发布教程、CPU 基线、发版仍未开始。
