# 浮点断言规则（D2.308/R5）

适用：所有对比 R 数值的测试，尤其是 DPQR、score/hessian、边界返回值。

## 规则

1. **禁止对 R 浮点做精确字符串相等**
   不要写：

       assert out[0] == "0.99999999999999967"

   R 的同一表达式在不同构建/平台可能差 1 ULP，测试会无故变红。

2. **位级断言只用于稳定值**
   可以精确比较：

   - 整数、计数、离散分位（q 返回 0,1,2...）
   - 明确不会变的字符串输出（错误消息、标签）
   - 已经验证在 Linux/Windows R 4.5.0 上完全一致的格式化值

   新增位级断言前，至少在 CI 与本地各跑一次；不一致就改用容差。

3. **连续值使用容差**
   默认：

       pytest.approx(expected, rel=1e-12, abs=1e-300)

   绝对容差只在有 ULP 依据时放宽，并在注释里写明量级：

       # R builds differ by ~1 ULP here; 1e-14 accepts both.
       assert raw == pytest.approx(1.0, abs=1e-14)
       assert raw < 1.0

4. **保留语义断言**
   容差不能替代语义。例如 DBI raw-sum 用例同时断言：

   - 数值接近 1（abs=1e-14）
   - 仍严格小于 1（未被 cdf[q > bd] <- 1 截断）
   - 下一档触发 STOP

5. **不得用容差掩盖实现错误**
   如果差异是 1e-3 级别，先判定是实现问题还是 R 版本变化；
   只有确认是参考版差异后才调整期望，并在 PR 说明。

## 反例（已修复）

test_r_consistency_dbi.py::test_live_r_returns_the_raw_sum_just_above_bd

- 旧写法：精确比较 R 字符串，CI 返回 ...944，本地返回 ...967，1 ULP 导致失败。
- 新写法：数值容差 + 语义断言，既抓真实回归，也不依赖 R 构建末位。

## 评审清单

- diff 中出现 float 转字符串的 == 比较？要求改容差或给出双平台证据。
- 新增 approx 的 abs 值是否远大于差异量级？要求收回到 1e-12 或注明 ULP。
- 是否删除了语义断言只留数值容差？要求补回。
- 是否因为升级 R 而放宽阈值？引用 r-reference-upgrade-checklist.md 的逐族结论。
