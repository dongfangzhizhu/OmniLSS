# 下次会话快速启动指南

## 当前状态（2026-06-02）

### 测试通过率
- DPQR: 11% (3/27) - 容差问题
- Fitting: 92% (11/12) - 良好
- Smoothing: 0% (0/3) - **需要修复**
- **总计: 33% (14/42)**

### 已发现的问题

#### 🔴 紧急问题

1. **cs smoother 完全失败**
   - 文件: `omnilss/src/omnilss/smoothers/cs.py`
   - 现象: 返回常数值（数据均值）
   - 影响: deviance 差异 362.76
   - 状态: 未修复

2. **JAX 64位模式未启用**
   - 文件: `benchmarks/comprehensive_r_consistency_test.py`
   - 现象: 所有 float64 被截断为 float32
   - 影响: DPQR 测试精度问题
   - 状态: 未修复

#### 🟡 高优先级问题

3. **pb smoother 参数不匹配**
   - 文件: `omnilss/src/omnilss/smoothers/pb.py`
   - 现象: Python 35 系数 vs R 2 系数（EDF=12）
   - 影响: deviance 差异 6.87
   - 状态: 未修复

4. **ps smoother 参数不匹配**
   - 文件: `omnilss/src/omnilss/smoothers/ps.py`
   - 现象: Python 24 系数 vs R 2 系数（EDF=5）
   - 影响: deviance 差异 148.90
   - 状态: 未修复

## 立即行动清单

### 第一步：修复 cs smoother（最高优先级）

```bash
# 1. 打开文件
code omnilss/src/omnilss/smoothers/cs.py

# 2. 检查问题点
# - fit_cubic_spline() 函数
# - UnivariateSpline 调用
# - df 参数处理
# - s 参数设置

# 3. 运行诊断验证
python diagnose_smoother_differences.py

# 4. 预期结果
# - 拟合值不再是常数
# - correlation > 0.95
# - deviance 差异 < 50
```

### 第二步：启用 JAX 64位模式

```bash
# 1. 检查环境变量设置
# 文件: benchmarks/comprehensive_r_consistency_test.py
# 行: 23-24

# 2. 确保设置生效
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# 3. 验证
python -c "import jax; print(jax.config.jax_enable_x64)"
# 应该输出: True

# 4. 重新运行测试
python benchmarks/comprehensive_r_consistency_test.py --quick
```

### 第三步：对齐 pb/ps 参数

```bash
# 1. 查看 R 源代码
# R/gamlss/R/pb.R - pb() 默认参数
# R/gamlss/R/PS.R - ps() 默认参数

# 2. 对比 Python 实现
# omnilss/src/omnilss/smoothers/pb.py
# omnilss/src/omnilss/smoothers/ps.py

# 3. 更新默认值
# - pb: n_knots 参数
# - ps: ps_intervals 参数

# 4. 运行诊断验证
python diagnose_smoother_differences.py
```

## 关键文件位置

### 需要修改的文件
- `omnilss/src/omnilss/smoothers/cs.py` - cs smoother 实现
- `omnilss/src/omnilss/smoothers/pb.py` - pb smoother 实现
- `omnilss/src/omnilss/smoothers/ps.py` - ps smoother 实现
- `benchmarks/comprehensive_r_consistency_test.py` - 测试框架

### 参考文件
- `R/gamlss/R/pb.R` - R pb() 实现
- `R/gamlss/R/PS.R` - R ps() 实现
- `R/gamlss/R/cubicSplines-10-08-12.R` - R cs() 实现

### 诊断和测试
- `diagnose_smoother_differences.py` - 诊断工具
- `benchmarks/comprehensive_r_consistency_test.py` - 完整测试

### 文档
- `SMOOTHER_DIAGNOSTIC_REPORT_2026-06-02.md` - 详细诊断报告
- `SESSION_SUMMARY_2026-06-02_AFTERNOON.md` - 会话总结
- `docs/development/r-gamlss-phase1-checklist-2026-05-23.md` - 开发计划

## 验证步骤

### 1. 快速验证（每次修改后）
```bash
python diagnose_smoother_differences.py
```

### 2. 完整测试（所有修复完成后）
```bash
# 设置环境变量
set JAX_ENABLE_X64=true
set JAX_PLATFORMS=cpu

# 运行测试
python benchmarks/comprehensive_r_consistency_test.py --quick

# 检查结果
# - Smoothing: 应该 3/3 通过
# - DPQR: 应该 >22/27 通过
# - Fitting: 应该 12/12 通过
```

### 3. 生成报告
```bash
# 测试结果会自动保存到
# benchmarks/results/reports/consistency_report_*.md
```

## 预期修复时间

- cs smoother 修复: 30-60 分钟
- JAX 64位启用: 10-15 分钟
- pb/ps 参数对齐: 30-45 分钟
- 测试验证: 15-20 分钟
- **总计: 1.5-2.5 小时**

## 成功标准

修复完成后应该达到：

| 测试类别 | 目标通过率 | 当前通过率 |
|---------|-----------|-----------|
| DPQR | >80% | 11% |
| Fitting | 100% | 92% |
| Smoothing | 100% | 0% |
| **总计** | **>90%** | **33%** |

## 常见问题

### Q1: cs smoother 为什么返回常数？
A: 可能是 `UnivariateSpline` 的 `s` 参数设置过大，导致过度平滑。检查 `fit_cubic_spline()` 中的参数设置。

### Q2: JAX 64位模式如何验证？
A: 运行 `python -c "import jax; print(jax.config.jax_enable_x64)"`，应该输出 `True`。

### Q3: pb/ps 参数应该设置为多少？
A: 需要查看 R 源代码确定。可能需要根据样本量动态调整。

### Q4: 修复后如何确认？
A: 运行 `diagnose_smoother_differences.py`，检查：
- cs: correlation > 0.95
- pb: deviance 差异 < 1.0
- ps: deviance 差异 < 10.0

## 调试技巧

### 1. cs smoother 调试
```python
# 在 fit_cubic_spline() 中添加调试输出
print(f"df={df}, k={k}, s={s_val if 's_val' in locals() else 'auto'}")
print(f"spline knots: {len(spline.get_knots())}")
print(f"fitted values range: [{fv.min():.4f}, {fv.max():.4f}]")
```

### 2. 参数对比
```python
# 对比 Python 和 R 的参数
print(f"Python n_knots: {n_knots}")
print(f"Python ps_intervals: {ps_intervals}")
# 然后查看 R 输出的 EDF
```

### 3. JAX 精度检查
```python
import jax
print(f"JAX enable x64: {jax.config.jax_enable_x64}")
print(f"JAX default dtype: {jax.numpy.zeros(1).dtype}")
```

## 联系信息

如果遇到问题，参考：
- 诊断报告: `SMOOTHER_DIAGNOSTIC_REPORT_2026-06-02.md`
- 会话总结: `SESSION_SUMMARY_2026-06-02_AFTERNOON.md`
- 开发计划: `docs/development/r-gamlss-phase1-checklist-2026-05-23.md`

---

*更新时间: 2026-06-02*
*状态: 等待修复*
*预计完成: 2026-06-03*
