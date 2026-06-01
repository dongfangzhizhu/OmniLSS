# Smoother Diagnostic Report - 2026年6月2日

## 执行摘要

通过详细诊断发现了 OmniLSS smoothers 与 R gamlss 不一致的根本原因：

1. **cs smoother 完全失败**：返回常数值（数据均值），无法拟合平滑曲线
2. **pb/ps smoothers 参数不匹配**：使用的基函数数量远多于 R，导致过拟合
3. **JAX 64位模式未启用**：所有测试在 float32 模式下运行，可能影响精度

## 详细诊断结果

### 测试数据

- 样本量：n = 300
- 预测变量：x = linspace(0, 10, 300)
- 响应变量：y = sin(x) + N(0, 0.3)
- 随机种子：42

### 1. pb (Penalized B-splines) Smoother

**Python 结果**:
- Deviance: 99.782131
- 系数数量: 35
- 拟合值范围: [-1.0417, 0.9814]
- 拟合值均值: 0.1807
- 拟合值标准差: 0.6516

**R 结果**:
- Deviance: 106.655500
- 系数数量: 2
- EDF (有效自由度): 12.0368
- 拟合值范围: [-1.0111, 0.9942]
- 拟合值均值: 0.1807
- 拟合值标准差: 0.6481

**误差分析**:
- Deviance 差异: 6.87
- 系数数量差异: 33 (Python 多了 33 个)
- Max absolute error: 5.66e-02
- Mean absolute error: 1.84e-02
- RMSE: 2.22e-02
- Correlation: 0.999431

**问题诊断**:
- Python 使用了 35 个系数，而 R 只使用 2 个系数（但 EDF=12）
- 这表明 Python 的默认参数设置与 R 不同
- R 可能使用了更少的节点或更强的惩罚
- 尽管系数数量差异大，但拟合形状相似（correlation=0.999）

**可能原因**:
1. `n_knots` 参数默认值不同
2. `lambda` 选择方法不同
3. 节点放置策略不同

### 2. ps (P-spline smooth) Smoother

**Python 结果**:
- Deviance: 103.146204
- 系数数量: 24
- 拟合值范围: [-1.0338, 0.9871]
- 拟合值均值: 0.1807
- 拟合值标准差: 0.6511

**R 结果**:
- Deviance: 252.042200
- 系数数量: 2
- EDF: 5
- 拟合值范围: [-0.6345, 0.7748]
- 拟合值均值: 0.1807
- 拟合值标准差: 0.4825

**误差分析**:
- Deviance 差异: 148.90
- 系数数量差异: 22 (Python 多了 22 个)
- Max absolute error: 3.99e-01
- Mean absolute error: 1.96e-01
- RMSE: 2.26e-01
- Correlation: 0.964274

**问题诊断**:
- Python 使用了 24 个系数，R 只使用 2 个（EDF=5）
- 拟合形状差异较大（correlation=0.964）
- Python 的拟合范围更大（2.02 vs 1.41）
- R 的拟合更平滑（更小的标准差）

**可能原因**:
1. `ps_intervals` 参数默认值不同（Python 默认 20，R 可能更小）
2. `lambda` 选择方法不同
3. 等间距节点放置策略实现不同

### 3. cs (Cubic Splines) Smoother

**Python 结果**:
- Deviance: 648.991306
- 系数数量: 2
- 拟合值范围: [0.1807, 0.1807] ⚠️ **常数值！**
- 拟合值均值: 0.1807
- 拟合值标准差: 0.0000 ⚠️ **无变化！**

**R 结果**:
- Deviance: 286.226700
- 系数数量: 2
- EDF: 5.0007
- 拟合值范围: [-0.6206, 0.6723]
- 拟合值均值: 0.1807
- 拟合值标准差: 0.4439

**误差分析**:
- Deviance 差异: 362.76
- 系数数量差异: 0
- Max absolute error: 8.01e-01
- Mean absolute error: 3.91e-01
- RMSE: 4.44e-01
- Correlation: NaN (Python 返回常数)

**问题诊断**:
- ❌ **严重 Bug**: Python 返回常数值（所有拟合值都是 0.1807）
- Python 完全没有拟合平滑曲线，只返回了数据均值
- 这是一个实现错误，不是参数问题

**可能原因**:
1. `UnivariateSpline` 调用参数错误
2. `df` 参数处理有问题
3. 排序或数据传递有问题
4. 默认参数导致 spline 退化为常数

## JAX 64位精度问题

所有测试都显示了大量的 JAX 警告：

```
UserWarning: Explicitly requested dtype <class 'jax.numpy.float64'> requested in array 
is not available, and will be truncated to dtype float32.
```

这表明：
1. 测试环境没有正确设置 `JAX_ENABLE_X64=true`
2. 所有计算都在 float32 精度下进行
3. 这可能导致数值精度问题，特别是对于 DPQR 测试

## 优先级和修复计划

### 🔴 紧急（立即修复）

1. **修复 cs smoother bug**
   - 文件：`omnilss/src/omnilss/smoothers/cs.py`
   - 问题：返回常数值
   - 需要：检查 `UnivariateSpline` 调用和参数处理
   - 预期：修复后应该能拟合平滑曲线

2. **启用 JAX 64位模式**
   - 文件：`benchmarks/comprehensive_r_consistency_test.py`
   - 问题：环境变量设置可能被覆盖
   - 需要：确保 `JAX_ENABLE_X64=true` 在所有测试中生效
   - 预期：减少 DPQR 测试的数值误差

### 🟡 高优先级（本周完成）

3. **对齐 pb smoother 默认参数**
   - 文件：`omnilss/src/omnilss/smoothers/pb.py`
   - 问题：使用的基函数数量过多
   - 需要：
     - 调查 R gamlss 的 `pb()` 默认参数
     - 对齐 `n_knots` 默认值
     - 对齐 `lambda` 选择方法
   - 预期：系数数量接近 R，deviance 差异 < 1.0

4. **对齐 ps smoother 默认参数**
   - 文件：`omnilss/src/omnilss/smoothers/ps.py`
   - 问题：使用的基函数数量过多，拟合形状差异大
   - 需要：
     - 调查 R gamlss 的 `ps()` 默认参数
     - 对齐 `ps_intervals` 默认值
     - 对齐节点放置策略
   - 预期：系数数量接近 R，correlation > 0.99

### 🟢 中优先级（后续开发）

5. **评估 DPQR 容差设置**
   - 当前容差：abs=1e-5, rel=1e-5
   - 实际误差：4.95e-05 到 4.90e-04
   - 需要：评估是否需要放宽容差或改进实现

6. **补充剩余 family 测试**
   - 还有 25 个 family 完全缺少测试
   - 需要继续使用测试生成框架补充

## 技术细节

### cs smoother 可能的 bug 位置

查看 `omnilss/src/omnilss/smoothers/cs.py` 的 `fit_cubic_spline` 函数：

```python
# 可能的问题点：
1. df 参数处理：
   if df is not None:
       k = int(np.clip(round(df + 2), 3, n - 1))
   # 这里 k 可能设置不当

2. UnivariateSpline 调用：
   spline = UnivariateSpline(xs, ys, w=ws, k=min(k, 5), s=s_val)
   # s 参数可能过大，导致过度平滑

3. 默认情况：
   spline = UnivariateSpline(xs, ys, w=ws, k=min(k, 5))
   # 可能没有正确选择 s 参数
```

### pb/ps smoother 参数对比

需要对比的 R 代码：
- `R/gamlss/R/pb.R` - pb() 函数的默认参数
- `R/gamlss/R/PS.R` - ps() 函数的默认参数

关键参数：
- `df`: 目标自由度
- `lambda`: 平滑参数
- `control`: 控制参数列表
- 节点数量和放置策略

## 验证计划

修复后需要重新运行：

1. **单元测试**
   ```bash
   python diagnose_smoother_differences.py
   ```

2. **完整测试**
   ```bash
   python benchmarks/comprehensive_r_consistency_test.py --quick
   ```

3. **预期结果**
   - cs smoother: 拟合值不再是常数，correlation > 0.95
   - pb smoother: deviance 差异 < 1.0, correlation > 0.999
   - ps smoother: deviance 差异 < 10.0, correlation > 0.99
   - DPQR tests: 通过率从 11% 提升到 > 80%

## 相关文件

- 诊断工具：`diagnose_smoother_differences.py`
- Smoother 实现：
  - `omnilss/src/omnilss/smoothers/pb.py`
  - `omnilss/src/omnilss/smoothers/ps.py`
  - `omnilss/src/omnilss/smoothers/cs.py`
- 测试框架：`benchmarks/comprehensive_r_consistency_test.py`
- 开发计划：`docs/development/r-gamlss-phase1-checklist-2026-05-23.md`

## 结论

通过详细诊断，我们发现了 smoother 测试失败的根本原因：

1. **cs smoother 有严重 bug**，需要立即修复
2. **pb/ps smoothers 参数不匹配**，需要对齐默认值
3. **JAX 64位模式未启用**，需要修复环境配置

这些问题都是可以修复的，预计修复后 smoothing 测试通过率可以达到 100%。

---

*报告生成时间: 2026-06-02*
*诊断工具: diagnose_smoother_differences.py*
