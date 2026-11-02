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
