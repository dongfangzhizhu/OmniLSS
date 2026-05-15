# OmniLSS 测试套件

## 概述

本目录包含OmniLSS的完整测试套件，包括功能测试和R一致性测试。

## 测试组织

### 按类型分类

```
tests/
├── test_basic_*.py          # 基础功能测试
├── test_r_consistency_*.py  # R一致性测试
├── test_operations.py       # 操作和方法测试
├── test_fixed_parameters.py # 固定参数测试
└── rbus/                    # R桥接工具
```

### 按分布批次分类

- **Batch 1**: 基础分布 (GU, RG, NBII, PARETO2, IGAMMA)
- **Batch 2**: 扩展分布 (NO2, LOGNO2, PE, SIMPLEX, exGAUS)
- **Batch 3**: 偏斜分布 (SHASH, SN1, SN2, GT, ST1-4)
- **Batch 4**: Beta变体 (BEINF, BEZI, BEOI)
- **Batch 5**: Zero-inflated (ZAGA, ZAIG, ZIP2, ZAP)
- **Batch 6**: 特殊离散 (PIG, SICHEL, SI, DPO, DEL, YULE, WARING)
- **Batch 7**: 复合离散 (BB, BNB, MN3/4/5)
- **Batch 8**: 连续特殊 (GG, GB2, PARETO, NET, LNO)
- **Batch 9**: 对数分布 (LG, ZIPF, ZALG)
- **Batch 10**: Zero变体 (ZABB, ZIBB, ZAPIG等)
- **Batch 11-16**: 其他分布

## 运行测试

### 运行所有测试

```bash
# 使用pytest
pytest tests/

# 使用项目脚本
./run_tests.ps1
```

### 运行特定批次

```bash
# Batch 1测试
pytest tests/test_basic_batch1_remaining.py

# R一致性测试
pytest tests/test_r_consistency_*.py
```

### 运行特定分布

```bash
# 测试NO分布
pytest tests/test_r_consistency_no.py -v

# 测试所有Poisson相关
pytest tests/ -k "poisson or PO" -v
```

### 跳过慢速测试

```bash
pytest tests/ -m "not slow"
```

## 测试类型

### 1. 基础功能测试 (`test_basic_*.py`)

测试分布的基本功能：

- **DPQ函数**: 密度、概率、分位数函数
- **参数验证**: 参数范围检查
- **数值稳定性**: 极端值处理
- **梯度计算**: 自动微分正确性

**示例**:
```python
def test_no_basic():
    """Test NO distribution basic functions."""
    family = NO()
    y = np.array([0.0, 1.0, 2.0])
    mu = 0.0
    sigma = 1.0
    
    # Test PDF
    pdf = family.d(y, mu=mu, sigma=sigma)
    assert np.all(np.isfinite(pdf))
    
    # Test CDF
    cdf = family.p(y, mu=mu, sigma=sigma)
    assert np.all((cdf >= 0) & (cdf <= 1))
```

### 2. R一致性测试 (`test_r_consistency_*.py`)

对比Python和R实现的结果：

- **拟合一致性**: 相同数据得到相似结果
- **偏差值对比**: 模型拟合质量
- **参数估计**: 参数值在合理范围内
- **收敛性**: 两种实现都能收敛

**示例**:
```python
def test_no_r_consistency(self):
    """Test NO distribution consistency with R."""
    # Generate data
    data = generate_test_data(n=100, distribution='NO')
    
    # Fit in Python
    py_model = gamlss("y ~ x", family=NO(), data=data)
    
    # Fit in R
    r_result = self.r_bridge.fit_gamlss("y ~ x", "NO", data)
    
    # Compare deviance
    assert abs(py_model.deviance - r_result['deviance']) < 0.01
```

### 3. 特殊测试

- **固定参数测试**: `test_fixed_parameters.py`
- **复杂分布测试**: `test_bug_condition_*.py`
- **保留性测试**: `test_preservation_*.py`


## CI and R-dependent Tests

The default GitHub Actions matrix runs Python 3.10, 3.11, and 3.12 on Linux,
Windows, and macOS without requiring native R. Tests that require `Rscript`,
`gamlss`, or `gamlss.dist` must skip when those dependencies are unavailable.

R-backed consistency remains important, but it belongs in an R-enabled validation
environment or the benchmark gate:

```bash
python benchmarks/run_local_validation.py --quick
```

Python-only benchmark smoke checks do not prove R equivalence.

## 测试状态

测试数量、跳过数量和通过率会随着架构迁移和 R 可用性变化，不再在本文档
中维护静态数字。请以当前环境中的 pytest 输出、CI 结果和 benchmark
validation gate 生成的报告为准。

当前维护重点：

1. **架构契约测试**：`test_core_architecture_contracts.py` 覆盖 canonical
   parameter、distribution protocol、legacy adapter 固定数据参数边界以及
   Optax optimizer adapter。
2. **R 一致性测试**：在缺少 `Rscript`、`gamlss` 或 `gamlss.dist` 的普通
   Python CI 环境中应跳过；发布 R 等价声明前必须在 R-enabled 环境运行。
3. **Benchmark gate**：Python-only smoke check 只能证明代码路径可执行，
   不能证明与 R `gamlss` 等价。

## R桥接

### RBridge工具

`rbus/r_bridge.py` 提供Python-R通信：

```python
from tests.rbus.r_bridge import RBridge

bridge = RBridge()
result = bridge.fit_gamlss("y ~ x", "NO", data)
```

### 要求

- R (>= 4.0)
- gamlss包
- rpy2

## 添加新测试

### 1. 基础功能测试

```python
# tests/test_basic_batchX.py
def test_newdist_basic():
    """Test NEWDIST distribution."""
    family = NEWDIST()
    
    # Test DPQ functions
    y = np.array([1.0, 2.0, 3.0])
    pdf = family.d(y, mu=1.0, sigma=1.0)
    
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf > 0)
```

### 2. R一致性测试

```python
# tests/test_r_consistency_newdist.py
class TestNEWDISTConsistency(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.r_bridge = RBridge()
    
    def test_intercept_only(self):
        """Test intercept-only model."""
        data = generate_test_data(100, 'NEWDIST')
        
        py_model = gamlss("y ~ 1", family=NEWDIST(), data=data)
        r_result = self.r_bridge.fit_gamlss("y ~ 1", "NEWDIST", data)
        
        self.assertAlmostEqual(
            py_model.deviance,
            r_result['deviance'],
            delta=0.1
        )
```

## 调试测试

### 详细输出

```bash
pytest tests/test_r_consistency_no.py -v -s
```

### 只运行失败的测试

```bash
pytest tests/ --lf
```

### 进入调试器

```bash
pytest tests/test_basic_batch1.py --pdb
```

### 查看覆盖率

```bash
pytest tests/ --cov=omnilss --cov-report=html
```

## 持续集成

测试在以下情况自动运行：

- 每次提交
- Pull Request
- 定期调度

## 性能测试

性能测试独立于功能测试：

```bash
# 运行性能测试
python benchmarks/run_performance_tests.py
```

详见 `benchmarks/performance/README.md`

## 故障排除

### R桥接失败

```
RuntimeError: R not available
```

**解决**: 安装R和gamlss包，或跳过R测试

### 测试超时

**解决**: 增加pytest超时或使用更小的测试数据

### 数值差异

**解决**: 调整容差或检查数值稳定性

## 最佳实践

1. **隔离测试**: 每个测试独立，不依赖其他测试
2. **确定性**: 使用固定随机种子
3. **快速**: 保持测试快速（<1秒）
4. **清晰**: 测试名称描述测试内容
5. **文档**: 复杂测试添加注释

## 参考

- [pytest文档](https://docs.pytest.org/)
- [JAX测试指南](https://jax.readthedocs.io/en/latest/notebooks/testing.html)
- [R GAMLSS文档](http://www.gamlss.org/)

---

**维护者**: OmniLSS团队  
**最后更新**: 2026-04-25
