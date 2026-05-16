# Google Colab Notebooks 快速参考

一页纸的快速参考指南。

---

## 🚀 快速开始（3 步）

```
1. 点击 Colab 徽章 → 打开 notebook
2. Runtime → Run all → 运行所有单元格
3. 查看结果 → 下载结果文件
```

---

## 📚 Notebooks 列表

| # | Notebook | 用途 | 时间 |
|---|----------|------|------|
| 01 | [quick_start](01_quick_start.ipynb) | 快速开始 | 5 分钟 |
| 02 | [consistency_dpqr](02_consistency_dpqr.ipynb) | 分布函数测试 | 10 分钟 |
| 03 | [consistency_fitting](03_consistency_fitting.ipynb) | 模型拟合测试 | 15 分钟 |
| 04 | [consistency_smoothing](04_consistency_smoothing.ipynb) | 平滑技术测试 | 15 分钟 |
| 05 | [performance_cpu](05_performance_cpu.ipynb) | CPU 性能 | 10 分钟 |
| 06 | [performance_gpu](06_performance_gpu.ipynb) | GPU 性能 | 10 分钟 |
| 07 | [performance_tpu](07_performance_tpu.ipynb) | TPU 性能 | 10 分钟 |
| 08 | [comprehensive_comparison](08_comprehensive_comparison.ipynb) | 综合对比 | 20 分钟 |
| 09 | [device_comparison](09_device_comparison.ipynb) | 设备对比 | 15 分钟 |

---

## 🎯 选择合适的 Notebook

### 我想...

**快速了解 OmniLSS**
→ `01_quick_start.ipynb`

**验证功能一致性**
→ `02_consistency_dpqr.ipynb`
→ `03_consistency_fitting.ipynb`

**测试性能**
→ `05_performance_cpu.ipynb`
→ `09_device_comparison.ipynb`

**对比 R 和 Python**
→ `08_comprehensive_comparison.ipynb`

**测试 GPU/TPU**
→ `06_performance_gpu.ipynb`
→ `07_performance_tpu.ipynb`

---

## 🔧 设备设置

### CPU（默认）

```
Runtime → Change runtime type
Hardware accelerator: None
```

### GPU

```
Runtime → Change runtime type
Hardware accelerator: GPU
```

### TPU（需要 Pro+）

```
Runtime → Change runtime type
Hardware accelerator: TPU
```

---

## 📊 预期结果

### 功能一致性

```
✓ 99.9% 测试通过
✓ 误差 < 1e-10
✓ 40+ 分布族
```

### 性能提升

```
CPU:  10-30x  faster than R
GPU:  50-100x faster than R
TPU:  100-200x faster than R
```

---

## 🐛 常见问题

### Q: GPU 不可用？

```
A: Runtime → Change runtime type → GPU
   或等待 GPU 资源可用
```

### Q: R 安装失败？

```
A: 重新运行安装单元格
   或重启运行时
```

### Q: 内存不足？

```
A: 减小数据规模
   或使用 Colab Pro
```

### Q: 运行时间过长？

```
A: 使用快速测试模式
   或减少测试数量
```

---

## 💾 保存结果

### 自动保存

```python
# 结果自动保存到
/content/omnilss_results/  # Colab
./results/                  # 本地
```

### 手动下载

```python
from google.colab import files
files.download('results.csv')
```

---

## 📝 修改测试

### 更改数据规模

```python
# 在配置单元格中修改
DATA_SIZES = [100, 500, 1000]  # 快速测试
DATA_SIZES = [100, 500, 1K, 5K, 10K]  # 标准测试
```

### 更改分布族

```python
# 在配置单元格中修改
DISTRIBUTIONS = ['NO', 'GA', 'PO']  # 核心分布
DISTRIBUTIONS = ALL_DISTRIBUTIONS  # 所有分布
```

### 更改运行次数

```python
# 在配置单元格中修改
N_RUNS = 3  # 快速测试
N_RUNS = 10  # 精确测试
```

---

## 🔗 有用的链接

- **文档**: [README.md](README.md)
- **计划**: [NOTEBOOKS_CREATION_PLAN.md](NOTEBOOKS_CREATION_PLAN.md)
- **总结**: [FINAL_SUMMARY.md](FINAL_SUMMARY.md)
- **GitHub**: https://github.com/dongfangzhizhu/OmniLSS
- **Issues**: https://github.com/dongfangzhizhu/OmniLSS/issues

---

## 🎓 学习路径

### 初学者路径

```
01 → 02 → 05 → 09
(快速开始 → 功能验证 → CPU性能 → 设备对比)
```

### 进阶路径

```
03 → 04 → 06 → 08
(模型拟合 → 平滑技术 → GPU性能 → 综合对比)
```

---

## 📞 获取帮助

- **Issues**: 报告问题
- **Discussions**: 讨论和建议
- **Email**: axu0606@gmail.com

---

**快速帮助**: 打开 [README.md](README.md) 查看完整文档
