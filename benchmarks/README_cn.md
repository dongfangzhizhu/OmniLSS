# OmniLSS 基准测试

[English version](README.md)

本目录包含用于在发布性能结论前验证 OmniLSS 与原生 R `gamlss` 一致性的基准测试脚本。

## 验证顺序

请始终按照以下顺序运行检查：

1. 先与原生 R `gamlss` 做**数值一致性**验证。
2. 只有一致性通过后，才做**性能对比**。
3. 使用生成的 JSON/Markdown 产物更新文档或报告。

编排脚本会强制执行该顺序：

```bash
python benchmarks/run_local_validation.py --quick
```

默认验证门禁需要 `Rscript` 以及 R 包 `gamlss` 和 `jsonlite`。没有 R 的环境可运行仅 Python 的 smoke 检查：

```bash
python benchmarks/run_local_validation.py --quick --allow-python-only --no-fit --no-smooth
```

仅 Python 的运行**不能**证明与原生 `gamlss` 等价；它只检查基准脚本和 OmniLSS 代码路径可执行。

## 单独脚本

| 脚本 | 目的 | 主要输出 |
|------|------|----------|
| `run_local_validation.py` | 先运行一致性，再在通过后运行性能测试 | 调用下方两个脚本 |
| `comprehensive_r_consistency_test.py` | 与 R `gamlss` 比较 d/p/q、拟合和平滑结果 | `benchmarks/results/raw/r_consistency_*.json` 与 `benchmarks/results/reports/consistency_report_*.md` |
| `comprehensive_performance_test.py` | 测量冷启动时间、warm 稳态时间、Python 堆内存峰值和排除启动成本的进程内 R 计时 | `benchmarks/results/raw/quick_results_*.json` 与 `benchmarks/results/reports/quick_report_*.md` |
| `three_way_comparison.py` | 可选的 OmniLSS/R/ondil 探索性比较 | `benchmarks/results/raw/three_way_*.json` 与 Markdown 报告 |
| `generate_plots.py` | 为基准测试产物绘图 | `benchmarks/results/figures/` |

## 一致性容差

默认容差是显式且保守的：

| 比较项 | 绝对容差 | 相对容差 |
|--------|---------:|---------:|
| d/p/q 函数 | `1e-5` | `1e-5` |
| 拟合和平滑 | `1e-3` | `1e-3` |
| 性能 deviance 检查 | `1e-5` | `1e-5` |

只有在有数值方法说明支持时，才从命令行覆盖容差，例如：

```bash
python benchmarks/comprehensive_r_consistency_test.py --quick \
  --dpqr-abs-tol 1e-6 --dpqr-rel-tol 1e-6
```

## 性能报告规则

报告必须区分：

- OmniLSS cold 时间：第一次拟合，包含 JAX 编译；
- OmniLSS warm 时间：重复稳态拟合时间；
- R 时间：单个 `Rscript` 进程内、包加载和 CSV 准备之后的 elapsed 计时，并包含一次不计时 R warm-up 拟合；
- Python 堆内存峰值：cold 拟合期间的 `tracemalloc` 峰值。

不要用静态营销语句概括结果。必须引用生成的报告、硬件、后端、dtype、数据规模、公式、重复次数，以及 R 是否可用。

## Phase 5 基准测试套件

v1.0 基准测试工作流拆分为三个套件，以便在不隐藏可选依赖的情况下复现实验结论：

| 套件 | 命令 | 依赖预期 | 用途 |
|------|------|----------|------|
| No-R smoke | `python benchmarks/jax_rs_benchmark.py --suite no-r --smoke --families NO --n-values 100 --n-reps 2` | 仅 Python/JAX；不需要 R 或 GPU | CI smoke 与快速本地健康检查 |
| Optional-R | `python benchmarks/jax_rs_benchmark.py --suite optional-r` | 如果 `Rscript` 和所需 R 包可用则使用 R；不可用时在产物中记录 R 比较缺失 | 发布和论文验证 |
| Optional-GPU | `python benchmarks/jax_rs_benchmark.py --suite optional-gpu --smoke` | 测量需要 JAX GPU 设备；没有 GPU 时写入 skipped 产物并成功退出 | GPU crossover 探索 |

`jax_rs_benchmark.py` 现在会在 JSON 和 Markdown 产物中记录重复 warm 计时样本及 95% 置信区间（`*_ci95`）。JAX cold 编译时间单独报告，绝不与 warm 稳态时间混合。

## 需求

- Python 能导入本仓库（`PYTHONPATH=omnilss/src` 或 editable install）。
- 可选但真实验证所需：`PATH` 上存在 `Rscript`，并安装：

```r
install.packages(c("gamlss", "jsonlite"))
```

## 输出位置

生成的基准测试产物写入 `benchmarks/results/`，除非发布流程明确要求冻结参考结果，否则不应提交这些产物。
