# OmniLSS 开发计划 - 阶段1总结报告 (2026-07-26)

## 📋 任务执行完成度统计

### 按用户指定顺序执行 ✓

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 1 | BI族修复（参数顺序） | ✅ Applied | benchmark params from (mu, bd) to (bd, mu) |
| 2 | NBI族修复（p/q） | ✅ Applied | normal approx → scipy.nbinom exact CDF/PPF |
| 3 | GU族修复（添加zeta参数） | ✅ Applied | pGU/qGU now accept zeta=0.0 default |
| 4 | PO_q修复 | ✅ Applied | normal approx → scipy.poisson.ppf exact |
| 5 | PO RS拟合 | ⏳ Pending | deviance formula needs R comparison |
| 6 | 平滑器pb/ps/cs | ⏳ Pending | full algorithm rewrite required |

---

## 🔧 已完成的具体代码修改

### 1. benchmarks/comprehensive_r_consistency_test.py
- `params = (0.5, 1)` → `params = (1, 0.5)` for BI distribution (matching R's dBI(x, bd=1, mu=0.5))

### 2. omnilss/src/omnilss/dpqr_functions.py
**新增导入:**
- `from scipy.stats import nbinom as _nbinom_dist` (line 21)
- `from scipy.stats import poisson as _pois_dist` (added during earlier fixes)

**pNBI函数修复 (lines ~1728-1780):**
- 移除正态近似 `norm.cdf(z)`
- 改用 `_nbinom.cdf(q_int.astype(float), n=size, loc=0, scale=p_param)`
- 正确参数化: `size = 1/sigma`, `p_param = size/(size+mu)`

**qNBI函数修复 (lines ~1784-1831):**
- 移除 `ndtri(p_val)` 正态分位数近似
- 改用 `_nbinom.ppf(p_val, n=size, loc=0, scale=p_param)`
- 确保非负整数输出: `np.maximum(np.floor(q), 0.0)`

**qPO函数修复 (lines ~223-260):**
- 移除 `jnp.floor(mu + jnp.sqrt(mu) * ndtri(p) + 0.5)` 正态近似
- 改用 `_pois_dist.ppf(p_val, mu=mu)`
- 概率裁剪到 `[1e-12, 1-1e-12]` 避免边界问题

**pGU/qGU签名更新 (lines ~2539, 2580):**
- 增加可选参数 `zeta=0.0` 以兼容R的潜在GEV参数化
- 保持现有Gumbel公式不变（因为zeta=0时等价）

---

## ⚠️ 待解决的关键问题

### 5. PO RS拟合 (deviance差异69.21)
**分析:** PO的g_dev_inc函数在distributions.py中实现正确，形式为:
```python
log_density = y * log(max(mu, eps)) - mu - gammaln(y+1)
deviance = -2 * log_density
```
这与标准Poisson deviance一致。失败可能源于：
- **基准测试中的R脚本参数传递问题**（mu估计值不同导致deviance计算差异）
- **Python与R在数值优化路径上的微小差异**
- **收敛容差设置不同**

**建议下一步:** 运行单独的比较测试，输出Python和R的相同输入下的deviance值，定位具体差异来源。

### 6. 平滑器 pb / ps / cs (全部失败)
**当前问题:**
- `pb_NO_smoothing`: deviance diff = 5.9586, rel_err = 3.34×10⁸
- `ps_NO_smoothing`: deviance diff = 152.22, rel_err = 34.62  
- `cs_NO_smoothing`: deviance diff = 3237.55, rel_err = 180.03

**根本原因（来自审计文件）:**
1. **双列表示bug**: smoothers使用两个不同的矩阵表示训练和预测，导致一致性破坏
2. **lambda生命周期错误**: smoothing parameter lambda在训练和预测阶段没有正确共享
3. **预测回归失败**: smoothes对预测数据的变换与训练不一致

**修复策略:**
```
Phase A (short-term): Implement single-train-representation invariant
  - For each smooth term S(X), maintain ONE explicit representation matrix B
  - Training: estimate coefficients β using GLM with B(Xβ)
  - Prediction: use SAME B(X_pred) without re-computing B
  
Phase B (medium-term): Rewrite pb using mgcv-style thin plate splines
  - Replicate gamlss中的pb smoother行为: penalized spline with degrees of freedom control
  - Reference: Wood, S.N. (2006) Generalized Additive Models: An Introduction with R
  
Phase C (long-term): Complete cs cubic spline重写
  - Implement natural cubic splines with correct boundary conditions
  - Match R mgcv::s(..., bs="cr") behavior
```

---

## 📊 预期基准测试结果（修复后）

应用上述修复后，预计通过率从 **30/42 (71.4%)** 提升至:

| 分布 | d | p | q | 说明 |
|-----|---|---|---|------|
| NO | ✅ | ✅ | ✅ | 原始正确 |
| GA | ✅ | ✅ | ✅ | QA已修复 |
| PO | ✅ | ✅ | ✅ | qPO已修复 |
| BI | ? | ? | ? | 需重测验证参数修复 |
| NBI | ✅ | ✅ | ⏳ | p/q修正后待验证 |
| BE | ✅ | ✅ | ✅ | 原始正确 |
| ZAGA | ✅ | ✅ | ✅ | 原始正确 |
| GU | ✅ | ⏳ | ⏳ | pGU/qGU添加了zeta但未验证R匹配 |
| SHASH | ✅ | ✅ | ✅ | 原始正确 |

**预期修复通过率：** 33-35/42 (79%-83%)，取决于BI和GU的R兼容性。

---

## 🚀 立即行动清单

1. **重新运行基准测试**
   ```bash
   set JAX_ENABLE_X64=True
   set PYTHONPATH=omnilss/src
   python benchmarks/comprehensive_r_consistency_test.py --quick
   ```

2. **分析剩余失败**，重点关注BI、NBI、GU、PO_RS

3. **PO RS拟合调试** — 比较Python/R在相同数据下的deviance值

4. **平滑器路线图确定** — 决定是直接复现R算法还是重新设计API

5. **阶段2规划** — 如果达到E3级验证，开始扩展实验性分布（experimental类别）

---

*报告生成时间: 2026-07-26*  
*作者: OmniLSS Development Team*