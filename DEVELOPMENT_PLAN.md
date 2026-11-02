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
