# OmniLSS 全面测试指南

**版本**: 2.0  
**创建日期**: 2026-05-04  
**项目**: OmniLSS (原GAMLSS)

---

## 📋 目录

1. [概述](#概述)
2. [新增功能](#新增功能)
3. [测试框架架构](#测试框架架构)
4. [快速开始](#快速开始)
5. [R一致性测试](#r一致性测试)
6. [性能测试](#性能测试)
7. [测试报告](#测试报告)
8. [使用示例](#使用示例)
9. [配置选项](#配置选项)
10. [故障排除](#故障排除)

---

## 概述

OmniLSS全面测试框架提供了完整的测试解决方案，确保：

1. **功能一致性**: 与R GAMLSS包的数值一致性，提供详细的误差范围
2. **性能评估**: 全面的性能数据，包括R版本有的和没有的功能
3. **自动化报告**: 生成详细的Markdown格式测试报告

### 核心特性

✅ **全面覆盖**: 测试所有分布、算法、平滑器和诊断工具  
✅ **详细误差**: 提供绝对误差、相对误差、RMSE、相关系数等多个指标  
✅ **性能对比**: 对比R版本的性能变化，提供加速比数据  
✅ **独有功能**: 测试OmniLSS独有的功能（Adam、L-BFGS、JIT、GPU等）  
✅ **自动报告**: 自动生成详细的测试报告

---

## 新增功能

### 2.0 版本新增 (2026-05-04)

#### 1. 全面R一致性测试 (`comprehensive_r_consistency_test.py`)

**功能**:
- 测试所有分布的d/p/q/r函数
- 测试所有拟合算法（RS, CG, Mixed）
- 测试所有平滑器（pb, ps, cs, tensor）
- 测试模型选择和诊断工具

**误差指标**:
- 最大绝对误差 (Max Absolute Error)
- 最大相对误差 (Max Relative Error)
- 平均绝对误差 (Mean Absolute Error)
- 平均相对误差 (Mean Relative Error)
- 均方根误差 (RMSE)
- 相关系数 (Correlation)

**性能指标**:
- Python执行时间
- R执行时间
- 加速比 (Speedup = R时间 / Python时间)

#### 2. 全面性能测试 (`comprehensive_performance_test.py`)

**R兼容功能测试**:
- 所有分布的d/p/q/r函数性能
- RS、CG、Mixed算法性能
- pb、ps、cs、tensor平滑器性能
- 与R版本的性能对比

**OmniLSS独有功能测试**:
- Adam和L-BFGS优化器性能
- JIT编译加速效果
- 向量化（vmap）性能
- GPU加速性能

**性能指标**:
- 平均时间、标准差、最小/最大时间
- 吞吐量 (samples/second)
- 内存使用
- 收敛率和迭代次数

#### 3. 自动报告生成 (`generate_comprehensive_report.py`)

**报告内容**:
- 总体摘要（成功率、测试数量）
- 按类别统计（dpqr、fitting、smoothing等）
- 详细误差统计（所有误差指标的分布）
- 性能对比（加速比统计）
- 详细测试结果表格
- 结论和建议

**报告格式**:
- Markdown格式，易于阅读和分享
- 包含表格、统计数据和可视化建议
- 自动分类成功和失败的测试

#### 4. 集成运行脚本 (`run_comprehensive_tests.py`)

**功能**:
- 一键运行所有测试
- 自动生成报告
- 支持快速测试模式
- 支持选择性运行（只运行一致性或性能测试）

---

## 测试框架架构

```
performance/
├── comprehensive_r_consistency_test.py    # R一致性测试
├── comprehensive_performance_test.py      # 性能测试
├── generate_comprehensive_report.py       # 报告生成器
├── run_comprehensive_tests.py             # 集成运行脚本
│
├── config.py                              # 配置文件
├── benchmarks/                            # 基准测试模块
│   ├── base.py                           # 基础类
│   ├── distributions.py                  # 分布测试
│   └── data_generators.py                # 数据生成器
│
├── reporters/                             # 报告生成器
│   └── markdown_reporter.py              # Markdown报告
│
└── results/                               # 测试结果
    ├── raw/                              # 原始JSON结果
    │   ├── r_consistency_*.json          # 一致性测试结果
    │   └── performance_*.json            # 性能测试结果
    └── reports/                          # 生成的报告
        └── comprehensive_report_*.md     # 综合报告
```

---

## 快速开始

### 1. 运行所有测试

```bash
# 运行完整测试套件（一致性 + 性能 + 报告）
python performance/run_comprehensive_tests.py
```

### 2. 快速测试

```bash
# 快速测试模式（减少测试数量，约5-10分钟）
python performance/run_comprehensive_tests.py --quick
```

### 3. 选择性测试

```bash
# 只运行一致性测试
python performance/run_comprehensive_tests.py --consistency-only

# 只运行性能测试
python performance/run_comprehensive_tests.py --performance-only
```

### 4. 查看报告

```bash
# 报告保存在 performance/results/reports/
cat performance/results/reports/comprehensive_report_*.md
```

---

## R一致性测试

### 测试内容

#### 1. d/p/q/r函数测试

测试所有分布的概率密度/质量函数(d)、累积分布函数(p)、分位数函数(q)和随机数生成(r)。

**测试分布** (45+):
- 基础分布: NO, GA, LOGNO, WEI, EXP, IG, LO, TF
- 离散分布: PO, BI, NBI, NBII, ZIP, ZIP2, ZINBI
- Beta族: BE, BEINF, ZAGA, ZAIG
- 高级分布: SHASH, SN1, SN2, GT, GG, GB2, NET
- 其他: GU, RG, IGAMMA, PARETO2, PIG, SICHEL, DPO, DEL

**误差指标**:
```python
{
    "max_absolute_error": 最大绝对误差,
    "max_relative_error": 最大相对误差,
    "mean_absolute_error": 平均绝对误差,
    "mean_relative_error": 平均相对误差,
    "rmse": 均方根误差,
    "correlation": 相关系数
}
```

#### 2. 模型拟合测试

测试不同算法的拟合结果一致性。

**测试算法**:
- RS (Rigby-Stasinopoulos)
- CG (Cole-Green)
- Mixed (混合算法)

**对比内容**:
- 系数估计
- 拟合值
- 偏差 (Deviance)
- AIC/BIC

#### 3. 平滑器测试

测试不同平滑器的拟合结果一致性。

**测试平滑器**:
- pb (P-splines)
- ps (P-splines smooth)
- cs (Cubic splines)
- te (Tensor product)

### 运行一致性测试

```bash
# 直接运行
python performance/comprehensive_r_consistency_test.py

# 查看结果
cat performance/results/raw/r_consistency_*.json
```

### 结果解读

**成功标准**:
- 最大相对误差 < 1e-4 (0.01%)
- 相关系数 > 0.9999
- RMSE < 1e-6

**示例输出**:
```
测试 NO_d: 成功
  最大绝对误差: 1.23e-08
  最大相对误差: 2.45e-07
  平均绝对误差: 3.21e-09
  RMSE: 4.56e-09
  相关系数: 0.999999
  加速比: 2.34x
```

---

## 性能测试

### 测试内容

#### 1. R兼容功能性能

测试与R GAMLSS相同的功能，提供性能对比。

**测试项目**:
- d/p/q/r函数 (45+ 分布)
- RS/CG/Mixed算法
- pb/ps/cs/tensor平滑器
- GCV/REML/ML/UBRE参数选择
- 模型选择 (AIC/BIC/GAIC)
- 诊断工具 (残差、QQ图、Worm图)
- 预测功能

**性能指标**:
- 执行时间 (平均、标准差、最小、最大)
- R执行时间
- 加速比 (R时间 / OmniLSS时间)
- 吞吐量 (samples/second)

#### 2. OmniLSS独有功能性能

测试R GAMLSS没有的功能，提供性能数据。

**测试项目**:
- Adam优化器
- L-BFGS优化器
- JIT编译加速
- 向量化 (vmap)
- GPU加速
- 自动微分

**性能指标**:
- 执行时间
- 吞吐量
- 加速效果 (vs 非优化版本)
- 内存使用

### 运行性能测试

```bash
# 直接运行
python performance/comprehensive_performance_test.py

# 查看结果
cat performance/results/raw/performance_*.json
```

### 结果解读

**性能等级**:
- 🚀 优秀: 加速比 > 2.0x
- ✅ 良好: 加速比 1.0x - 2.0x
- ⚠️ 可接受: 加速比 0.8x - 1.0x
- ❌ 需优化: 加速比 < 0.8x

**示例输出**:
```
测试 NO_RS_fitting: 成功
  平均时间: 0.234s
  R时间: 0.678s
  加速比: 2.90x (🚀 优秀)
  吞吐量: 4273 samples/s
```

---

## 测试报告

### 报告内容

生成的报告包含以下部分：

#### 1. 一致性测试报告

- **总体摘要**: 测试数量、成功率
- **按类别统计**: dpqr、fitting、smoothing等
- **误差统计**: 所有误差指标的分布（最小、最大、平均、中位数、标准差）
- **性能对比**: 加速比统计
- **详细结果**: 每个测试的详细数据表格
- **结论**: 自动评估一致性和性能

#### 2. 性能测试报告

- **总体摘要**: 测试数量、成功率
- **按类别统计**: dpqr、fitting、smoothing、jax_features等
- **按功能类型统计**: R兼容功能 vs OmniLSS独有功能
- **性能对比**: R兼容功能的加速比统计
- **时间统计**: 所有测试的时间分布
- **详细结果**: 分类展示R兼容和独有功能
- **结论**: 自动评估性能和功能覆盖

### 生成报告

```bash
# 自动生成（在运行测试时）
python performance/run_comprehensive_tests.py

# 手动生成
python performance/generate_comprehensive_report.py \
    --consistency performance/results/raw/r_consistency_20260504_120000.json \
    --performance performance/results/raw/performance_20260504_120000.json \
    --output performance/results/reports/my_report.md
```

### 报告示例

```markdown
# OmniLSS vs R GAMLSS 功能一致性测试报告

## 总体摘要

- **总测试数**: 150
- **成功测试数**: 147
- **成功率**: 98.0%

### 按类别统计

| 类别 | 总数 | 成功 | 成功率 |
|------|------|------|--------|
| dpqr | 60 | 59 | 98.3% |
| fitting | 36 | 36 | 100.0% |
| smoothing | 24 | 23 | 95.8% |

## 误差统计

### Max Absolute Error

| 统计量 | 值 |
|--------|-----|
| 最小值 | 1.23e-10 |
| 最大值 | 4.56e-06 |
| 平均值 | 2.34e-08 |
| 中位数 | 1.23e-08 |
| 标准差 | 3.45e-08 |

## 性能对比

| 统计量 | 值 |
|--------|-----|
| 最小加速比 | 0.95x |
| 最大加速比 | 15.23x |
| 平均加速比 | 3.45x |
| 中位数加速比 | 2.78x |

## 结论

✅ **优秀**: OmniLSS与R GAMLSS的一致性达到 98.0%，功能实现高度一致。

✅ **数值精度**: 平均绝对误差为 2.34e-08，数值精度极高。

🚀 **性能**: OmniLSS平均比R GAMLSS快 3.45倍，性能优势显著。
```

---

## 使用示例

### 示例1: 完整测试流程

```bash
# 1. 运行所有测试
python performance/run_comprehensive_tests.py

# 2. 查看报告
cat performance/results/reports/comprehensive_report_*.md

# 3. 分析结果
# - 查看成功率
# - 查看误差范围
# - 查看性能对比
# - 识别需要改进的功能
```

### 示例2: 快速验证

```bash
# 快速测试（5-10分钟）
python performance/run_comprehensive_tests.py --quick

# 查看关键指标
grep "成功率" performance/results/reports/comprehensive_report_*.md
grep "加速比" performance/results/reports/comprehensive_report_*.md
```

### 示例3: 只测试特定功能

```bash
# 只测试一致性
python performance/run_comprehensive_tests.py --consistency-only

# 只测试性能
python performance/run_comprehensive_tests.py --performance-only
```

### 示例4: 对比两次测试

```bash
# 运行基准测试
python performance/run_comprehensive_tests.py
mv performance/results/raw/performance_*.json baseline.json

# 进行优化...

# 运行优化后测试
python performance/run_comprehensive_tests.py
mv performance/results/raw/performance_*.json optimized.json

# 对比结果
python performance/compare_results.py baseline.json optimized.json
```

---

## 配置选项

### 测试配置

在 `performance/config.py` 中配置：

```python
# 数据规模
DATA_SIZES = [
    DataSize("tiny", 100, "Tiny dataset"),
    DataSize("small", 500, "Small dataset"),
    DataSize("medium", 5_000, "Medium dataset"),
    DataSize("large", 50_000, "Large dataset"),
]

# 测试分布
DISTRIBUTIONS = [
    "NO", "GA", "LOGNO", "WEI", "EXP", "IG", "LO", "TF",
    "PO", "BI", "NBI", "NBII", "ZIP",
    "BE", "ZAGA",
    # ... 更多分布
]

# 基准测试配置
BenchmarkConfig(
    n_repeats=5,        # 重复次数
    warmup_runs=2,      # 预热次数
    timeout=300.0,      # 超时时间（秒）
    memory_profiling=True,
    save_raw_results=True,
    random_seed=42,
)
```

### 性能阈值

```python
PERFORMANCE_THRESHOLDS = {
    "speedup_good": 1.5,           # 良好加速比
    "speedup_acceptable": 0.8,     # 可接受加速比
    "speedup_poor": 0.5,           # 较差加速比
    "memory_ratio_good": 1.2,      # 良好内存比
    "memory_ratio_acceptable": 2.0, # 可接受内存比
    "convergence_rate_good": 0.95,  # 良好收敛率
    "convergence_rate_acceptable": 0.85, # 可接受收敛率
}
```

---

## 故障排除

### 常见问题

#### 1. R bridge连接失败

**问题**: 无法调用R函数进行对比

**解决方案**:
```bash
# 检查R是否安装
R --version

# 检查gamlss包是否安装
R -e "library(gamlss)"

# 安装gamlss包
R -e "install.packages('gamlss')"
```

#### 2. 测试超时

**问题**: 某些测试运行时间过长

**解决方案**:
```python
# 在config.py中增加超时时间
BenchmarkConfig(
    timeout=600.0,  # 增加到10分钟
)

# 或使用快速测试模式
python performance/run_comprehensive_tests.py --quick
```

#### 3. 内存不足

**问题**: 大数据集测试导致内存不足

**解决方案**:
```python
# 减少数据规模
DATA_SIZES = [
    DataSize("tiny", 100, "Tiny dataset"),
    DataSize("small", 500, "Small dataset"),
    # 注释掉大数据集
    # DataSize("large", 50_000, "Large dataset"),
]
```

#### 4. 某些分布测试失败

**问题**: 特定分布的测试失败

**解决方案**:
```bash
# 单独测试该分布
python -c "
from performance.comprehensive_r_consistency_test import test_dpqr_consistency
results = test_dpqr_consistency('NO', n_samples=1000)
for r in results:
    print(r.to_dict())
"

# 查看详细错误信息
```

---

## 贡献指南

### 添加新的测试

#### 1. 添加新的分布测试

在 `config.py` 中添加：

```python
DISTRIBUTIONS.append(
    DistributionConfig(
        name="NEW_DIST",
        r_name="NEW_DIST",
        python_class="NEW_DIST",
        type="continuous",
        parameters=["mu", "sigma"],
    )
)
```

#### 2. 添加新的性能测试

在 `comprehensive_performance_test.py` 中添加：

```python
def test_new_feature() -> PerformanceResult:
    """测试新功能的性能"""
    try:
        # 实现测试逻辑
        bench_result = benchmark_function(test_func, n_repeats=5)
        
        return PerformanceResult(
            test_name="new_feature",
            category="new_category",
            feature_type="omnilss_only",
            success=True,
            **bench_result,
        )
    except Exception as e:
        return PerformanceResult(
            test_name="new_feature",
            category="new_category",
            feature_type="omnilss_only",
            success=False,
            error_message=str(e),
        )
```

#### 3. 添加新的误差指标

在 `calculate_errors` 函数中添加：

```python
def calculate_errors(python_vals, r_vals):
    # ... 现有代码 ...
    
    # 添加新指标
    new_metric = calculate_new_metric(python_vals, r_vals)
    
    return {
        # ... 现有指标 ...
        "new_metric": new_metric,
    }
```

### 提交测试结果

1. 运行完整测试套件
2. 确保成功率 > 95%
3. 提交测试报告和原始数据
4. 在PR中说明测试环境和配置

---

## 总结

OmniLSS全面测试框架提供了：

✅ **完整的功能一致性测试** - 确保与R GAMLSS的数值一致性  
✅ **详细的误差分析** - 提供多个误差指标和统计数据  
✅ **全面的性能评估** - 覆盖所有功能点，包括独有功能  
✅ **自动化报告生成** - 生成详细的Markdown格式报告  
✅ **易于使用** - 一键运行所有测试  
✅ **可扩展** - 易于添加新的测试和指标

### 下一步

1. 运行测试: `python performance/run_comprehensive_tests.py`
2. 查看报告: `cat performance/results/reports/comprehensive_report_*.md`
3. 分析结果并改进
4. 定期运行测试确保质量

---

**维护者**: OmniLSS 团队  
**最后更新**: 2026-05-04  
**版本**: 2.0
