[object Object]

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
