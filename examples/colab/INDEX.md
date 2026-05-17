# Google Colab Notebooks 索引

[中文版本](INDEX_cn.md)

快速导航到所有文档和 notebooks。

---

## 📚 主要文档

| 文档 | 描述 | 推荐阅读顺序 |
|------|------|-------------|
| [README.md](README.md) | 完整的使用指南 | ⭐ 1. 首先阅读 |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | 快速参考卡片 | ⭐ 2. 快速查阅 |
| [NOTEBOOKS_CREATION_PLAN.md](NOTEBOOKS_CREATION_PLAN.md) | 详细创建计划 | 3. 开发参考 |
| [COLAB_NOTEBOOKS_SUMMARY.md](COLAB_NOTEBOOKS_SUMMARY.md) | 项目总结 | 4. 了解项目 |
| [FINAL_SUMMARY.md](FINAL_SUMMARY.md) | 最终总结 | 5. 完整信息 |
| [COLAB_NOTEBOOKS_COMPLETION_REPORT.md](../COLAB_NOTEBOOKS_COMPLETION_REPORT.md) | 完成报告 | 6. 项目状态 |

---

## 📓 Notebooks

### ✅ 已完成

| # | Notebook | 描述 | Colab 链接 | 时间 |
|---|----------|------|-----------|------|
| 01 | [quick_start.ipynb](01_quick_start.ipynb) | 快速开始指南 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dongfangzhizhu/OmniLSS/blob/main/examples/colab/01_quick_start.ipynb) | 5 分钟 |
| 09 | [device_comparison.ipynb](09_device_comparison.ipynb) | CPU/GPU/TPU 对比 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dongfangzhizhu/OmniLSS/blob/main/examples/colab/09_device_comparison.ipynb) | 15 分钟 |

### 📋 待创建（高优先级）

| # | Notebook | 描述 | 状态 | 预计时间 |
|---|----------|------|------|---------|
| 02 | consistency_dpqr.ipynb | 分布函数一致性测试 | 📋 规划完成 | 10 分钟 |
| 05 | performance_cpu.ipynb | CPU 性能详细测试 | 📋 规划完成 | 10 分钟 |

### 📋 待创建（中优先级）

| # | Notebook | 描述 | 状态 | 预计时间 |
|---|----------|------|------|---------|
| 03 | consistency_fitting.ipynb | 模型拟合一致性 | 📋 规划完成 | 15 分钟 |
| 06 | performance_gpu.ipynb | GPU 性能测试 | 📋 规划完成 | 10 分钟 |
| 08 | comprehensive_comparison.ipynb | 综合对比 | 📋 规划完成 | 20 分钟 |

### 📋 待创建（低优先级）

| # | Notebook | 描述 | 状态 | 预计时间 |
|---|----------|------|------|---------|
| 04 | consistency_smoothing.ipynb | 平滑技术测试 | 📋 规划完成 | 15 分钟 |
| 07 | performance_tpu.ipynb | TPU 性能测试 | 📋 规划完成 | 10 分钟 |

---

## 🛠️ 工具

| 工具 | 描述 | 用途 |
|------|------|------|
| [generate_notebooks.py](generate_notebooks.py) | Notebook 生成工具 | 自动化创建 notebooks |

---

## 🎯 快速导航

### 我想...

**快速了解项目**
→ [README.md](README.md)

**快速上手**
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
→ [01_quick_start.ipynb](01_quick_start.ipynb)

**了解项目状态**
→ [COLAB_NOTEBOOKS_COMPLETION_REPORT.md](../COLAB_NOTEBOOKS_COMPLETION_REPORT.md)

**开发新 notebook**
→ [NOTEBOOKS_CREATION_PLAN.md](NOTEBOOKS_CREATION_PLAN.md)
→ [generate_notebooks.py](generate_notebooks.py)

**测试性能**
→ [09_device_comparison.ipynb](09_device_comparison.ipynb)

**验证功能**
→ [02_consistency_dpqr.ipynb](02_consistency_dpqr.ipynb) (待创建)

---

## 📊 项目统计

### 完成度

```
文档:     6/6  (100%) ✅
Notebooks: 2/9  (22%)  📋
工具:     1/1  (100%) ✅
总体:     9/16 (56%)  📋
```

### 代码量

```
文档:     ~4,000 行 Markdown
Notebooks: ~500 行 Python
工具:     ~400 行 Python
总计:     ~4,900 行
```

---

## 🎓 学习路径

### 初学者

```
1. README.md
   ↓
2. QUICK_REFERENCE.md
   ↓
3. 01_quick_start.ipynb
   ↓
4. 09_device_comparison.ipynb
```

### 进阶用户

```
1. NOTEBOOKS_CREATION_PLAN.md
   ↓
2. 02_consistency_dpqr.ipynb
   ↓
3. 05_performance_cpu.ipynb
   ↓
4. 08_comprehensive_comparison.ipynb
```

### 开发者

```
1. NOTEBOOKS_CREATION_PLAN.md
   ↓
2. generate_notebooks.py
   ↓
3. 创建新 notebook
   ↓
4. 提交 PR
```

---

## 📞 支持

- **Issues**: https://github.com/dongfangzhizhu/OmniLSS/issues
- **Discussions**: https://github.com/dongfangzhizhu/OmniLSS/discussions
- **Email**: axu0606@gmail.com

---

## 🔗 相关资源

### 项目资源

- [OmniLSS GitHub](https://github.com/dongfangzhizhu/OmniLSS)
- [OmniLSS Documentation](https://omnilss.readthedocs.io)
- [教程系列](../../tutorials/README.md)
- [宣传文章](../../tutorials/promotional/)

### 外部资源

- [Google Colab](https://colab.research.google.com/)
- [JAX Documentation](https://jax.readthedocs.io/)
- [R GAMLSS](http://www.gamlss.org/)

---

## 📝 更新日志

### 2026-05-08

- ✅ 创建了完整的文档框架
- ✅ 实现了 2 个核心 notebooks
- ✅ 开发了生成工具
- ✅ 建立了质量标准

### 待更新

- 📋 创建剩余 7 个 notebooks
- 📋 收集用户反馈
- 📋 改进文档和工具

---

**最后更新**: 2026-05-08  
**维护者**: OmniLSS 团队  
**版本**: 1.0
