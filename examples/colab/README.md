# OmniLSS Google Colab Notebooks

这个目录包含一系列 Google Colab notebooks，用于测试 OmniLSS 与 R GAMLSS 的功能一致性和性能对比。

---

## 📚 Notebook 列表

### 1. 快速开始

| Notebook | 描述 | Colab 链接 |
|----------|------|-----------|
| [01_quick_start.ipynb](01_quick_start.ipynb) | 快速开始指南，安装和基本使用 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dongfangzhizhu/OmniLSS/blob/main/examples/colab/01_quick_start.ipynb) |

### 2. 功能一致性测试

| Notebook | 描述 | Colab 链接 |
|----------|------|-----------|
| [02_consistency_dpqr.ipynb](02_consistency_dpqr.ipynb) | 测试分布函数（d/p/q/r）的一致性 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dongfangzhizhu/OmniLSS/blob/main/examples/colab/02_consistency_dpqr.ipynb) |
| [03_consistency_fitting.ipynb](03_consistency_fitting.ipynb) | 测试模型拟合的一致性 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dongfangzhizhu/OmniLSS/blob/main/examples/colab/03_consistency_fitting.ipynb) |
| [04_consistency_smoothing.ipynb](04_consistency_smoothing.ipynb) | 测试平滑技术的一致性 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dongfangzhizhu/OmniLSS/blob/main/examples/colab/04_consistency_smoothing.ipynb) |

### 3. 性能测试

| Notebook | 描述 | Colab 链接 |
|----------|------|-----------|
| [05_performance_cpu.ipynb](05_performance_cpu.ipynb) | CPU 性能测试 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dongfangzhizhu/OmniLSS/blob/main/examples/colab/05_performance_cpu.ipynb) |
| [06_performance_gpu.ipynb](06_performance_gpu.ipynb) | GPU 性能测试 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dongfangzhizhu/OmniLSS/blob/main/examples/colab/06_performance_gpu.ipynb) |
| [07_performance_tpu.ipynb](07_performance_tpu.ipynb) | TPU 性能测试 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dongfangzhizhu/OmniLSS/blob/main/examples/colab/07_performance_tpu.ipynb) |

### 4. 综合对比

| Notebook | 描述 | Colab 链接 |
|----------|------|-----------|
| [08_comprehensive_comparison.ipynb](08_comprehensive_comparison.ipynb) | 全面对比 R 和 Python 版本 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dongfangzhizhu/OmniLSS/blob/main/examples/colab/08_comprehensive_comparison.ipynb) |
| [09_device_comparison.ipynb](09_device_comparison.ipynb) | CPU/GPU/TPU 性能对比 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dongfangzhizhu/OmniLSS/blob/main/examples/colab/09_device_comparison.ipynb) |

---

## 🚀 快速开始

### 在 Google Colab 中运行

1. 点击上面的 "Open In Colab" 按钮
2. 在 Colab 中，选择运行时类型：
   - **CPU**: Runtime → Change runtime type → Hardware accelerator: None
   - **GPU**: Runtime → Change runtime type → Hardware accelerator: GPU
   - **TPU**: Runtime → Change runtime type → Hardware accelerator: TPU
3. 运行所有单元格：Runtime → Run all

### 本地运行

```bash
# 克隆仓库
git clone https://github.com/dongfangzhizhu/OmniLSS.git
cd OmniLSS

# 安装依赖
pip install jupyter notebook
pip install -e omnilss

# 启动 Jupyter
cd examples/colab
jupyter notebook
```

---

## 📊 测试内容

### 功能一致性测试

测试 OmniLSS 与 R GAMLSS 的功能一致性：

1. **分布函数（DPQR）**
   - 密度函数（d）
   - 累积分布函数（p）
   - 分位数函数（q）
   - 随机数生成（r）
   - 覆盖 40+ 分布族

2. **模型拟合**
   - RS 算法
   - CG 算法
   - Mixed 算法
   - 参数估计精度
   - 收敛性

3. **平滑技术**
   - P-splines (pb)
   - Cubic splines (ps, cs)
   - 平滑参数选择（GCV, REML）

### 性能测试

测试不同硬件上的性能：

1. **CPU 性能**
   - 不同数据规模（100 - 100,000 样本）
   - 不同分布族
   - 不同模型复杂度
   - 与 R 的性能对比

2. **GPU 性能**
   - GPU 加速效果
   - 不同数据规模的加速比
   - 内存使用
   - 与 CPU 的对比

3. **TPU 性能**
   - TPU 加速效果
   - 大规模数据处理
   - 与 CPU/GPU 的对比

---

## 📈 预期结果

### 功能一致性

- **目标**: 99.9% 的测试通过
- **精度**: 参数估计误差 < 0.01%
- **覆盖**: 40+ 分布族，所有核心功能

### 性能提升

| 硬件 | 预期加速比 | 数据规模 |
|------|-----------|---------|
| **CPU** | 10-30x | 所有规模 |
| **GPU** | 50-100x | 大规模（10K+） |
| **TPU** | 100-200x | 超大规模（100K+） |

---

## 🔧 环境要求

### Google Colab

- **免费版**: 可以运行所有 notebooks（CPU/GPU）
- **Pro 版**: 更快的 GPU，更长的运行时间
- **Pro+ 版**: TPU 访问，更多资源

### 本地环境

```bash
# Python 版本
python >= 3.10

# 核心依赖
jax >= 0.4.20
jaxlib >= 0.4.20
numpy >= 1.24.0
scipy >= 1.11.0
pandas >= 2.0.0

# 可选依赖（用于对比）
rpy2  # R 接口
```

---

## 📝 使用说明

### 1. 选择合适的 Notebook

- **初次使用**: 从 `01_quick_start.ipynb` 开始
- **功能验证**: 使用 `02-04` 系列
- **性能测试**: 使用 `05-07` 系列
- **全面对比**: 使用 `08-09` 系列

### 2. 配置运行时

```python
# 在 notebook 中检查设备
import jax
print(f"可用设备: {jax.devices()}")
print(f"默认设备: {jax.devices()[0]}")

# 指定设备
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # 或 'gpu', 'tpu'
```

### 3. 运行测试

每个 notebook 都包含：
- 环境设置
- 数据生成
- 测试执行
- 结果可视化
- 性能分析

### 4. 保存结果

```python
# 结果会自动保存到
# /content/omnilss_results/  (Colab)
# ./results/                  (本地)
```

---

## 🎯 测试指标

### 一致性指标

- **绝对误差**: |Python - R|
- **相对误差**: |Python - R| / |R|
- **RMSE**: 均方根误差
- **相关系数**: Python 和 R 结果的相关性

### 性能指标

- **执行时间**: 平均、最小、最大
- **加速比**: R_time / Python_time
- **吞吐量**: 样本数/秒
- **内存使用**: MB
- **收敛速度**: 迭代次数

---

## 🐛 故障排查

### 常见问题

#### 1. R 安装失败

```python
# 在 Colab 中安装 R
!apt-get update
!apt-get install -y r-base r-base-dev
!R -e "install.packages('gamlss', repos='https://cran.r-project.org')"
```

#### 2. GPU 不可用

```python
# 检查 GPU
!nvidia-smi

# 如果没有 GPU，切换运行时
# Runtime → Change runtime type → Hardware accelerator: GPU
```

#### 3. TPU 不可用

```python
# TPU 仅在 Colab Pro+ 中可用
# 或使用 Google Cloud TPU
```

#### 4. 内存不足

```python
# 减小数据规模
DATA_SIZES = [100, 500, 1000]  # 而不是 [100, 500, 5000, 50000]

# 或使用批处理
for batch in batches:
    process_batch(batch)
```

---

## 📊 结果示例

### 功能一致性

```
✓ NO distribution: PASSED (error < 1e-10)
✓ GA distribution: PASSED (error < 1e-10)
✓ LOGNO distribution: PASSED (error < 1e-10)
...
Total: 92/92 tests passed (100%)
```

### 性能对比

```
Distribution: NO, n=5000
  R time:      1.02s
  Python time: 0.03s
  Speedup:     34.0x
  
Distribution: GA, n=5000
  R time:      1.07s
  Python time: 0.17s
  Speedup:     6.3x
```

### 设备对比

```
Distribution: NO, n=50000
  CPU:  0.45s (baseline)
  GPU:  0.08s (5.6x faster)
  TPU:  0.04s (11.3x faster)
```

---

## 🤝 贡献

欢迎贡献新的 notebooks：

1. Fork 仓库
2. 创建新的 notebook
3. 遵循现有的格式和风格
4. 提交 Pull Request

---

## 📞 支持

- **Issues**: https://github.com/dongfangzhizhu/OmniLSS/issues
- **Discussions**: https://github.com/dongfangzhizhu/OmniLSS/discussions
- **Documentation**: https://omnilss.readthedocs.io

---

## 📄 许可证

所有 notebooks 遵循项目的 GPL-3.0+ 许可证。

---

**维护者**: OmniLSS 团队  
**最后更新**: 2026-05-08
