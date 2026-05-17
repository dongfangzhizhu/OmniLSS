# 设备感知方法路由指南

OmniLSS 有两条 RS 拟合路径：

- `method="RS"`：NumPy IRLS，当前默认路径。
- `method="RS_JAX"`：JAX JIT 编译 IRLS，仅覆盖核心 JAX 族。
- `method="auto"`：根据设备、观测数 `n` 和配置阈值自动选择 `RS` 或 `RS_JAX`。

## 快速结论

| 场景 | 推荐方法 |
|---|---|
| CPU，任意规模 | `method="RS"`（NumPy，默认） |
| GPU，`n` 小于实测阈值 | `method="RS"` |
| GPU，`n` 大于等于实测阈值 | `method="auto"` 或 `method="RS_JAX"` |
| TPU | `method="auto"`（默认阈值禁用自动切换，当前等同 `RS`） |
| 需要平滑项（`pb`/`cs`/`ps`） | 只能用 `method="RS"` |
| 不在 6 个核心 JAX 族内 | 只能用 `method="RS"` |

当前包内 GPU/TPU 阈值默认保持为 `math.inf`（禁用自动切换），因此在维护者或用户写入可复现的实测阈值之前，`method="auto"` 在 CPU/GPU/TPU 上都会保守地选择 `RS`。

## 为什么 CPU 上 JAX 更慢？

CPU 上 NumPy RS 通常更快，主要原因是：

1. **XLA 编译开销**：`RS_JAX` 首次调用需要编译，冷启动成本无法在小模型中摊销。
2. **Python → JAX 边界成本**：当前 JAX 路径仍需要复用 OmniLSS 的公式解析、初始化和模型包装逻辑。
3. **小矩阵线性代数**：典型 GAMLSS 设计矩阵列数较小（例如 `p=2–10`），CPU LAPACK 对这类 WLS 问题非常高效。
4. **GPU 核启动开销**：在小 `p` 场景中，GPU kernel launch 延迟可能超过矩阵运算本身收益。

这也是默认策略仍保持“CPU 永不自动切到 JAX；GPU/TPU 只有达到实测阈值才切换”的原因。

## 基准测试结果

### 已知 CPU/GPU 初测结论

| 设备 | 结论 |
|---|---|
| CPU：Intel i7-12700K，JAX 0.6.2 CPU backend | `RS_JAX` 在 `n ≤ 1,000,000` 范围内未超过 NumPy `RS`。 |
| GPU：NVIDIA RTX 3060 12 GB，JAX 0.10.0，CUDA 12，`p=2` | `RS_JAX` 在 `n ≤ 500,000` 范围内仍约慢于 NumPy `RS`，默认阈值保持 `math.inf`。 |
| TPU | 尚未完成基准测试，默认阈值保持 `math.inf`。 |

### 运行 GPU 阈值扫描

使用新的 GPU sweep 脚本生成原始 JSON 和 Markdown 报告：

```bash
PYTHONPATH=omnilss/src python benchmarks/gpu_crossover_sweep.py
```

默认矩阵覆盖：

| 变量 | 取值 |
|---|---|
| `n` | 100 · 500 · 1,000 · 5,000 · 10,000 · 50,000 · 100,000 · 500,000 |
| `p` | 2 · 5 · 10 · 20 · 50 |
| 分布族 | NO · GA · PO · BI · WEI · TF |
| 方法 | `RS` vs `RS_JAX`（JAX 预热后计时） |

输出位置：

- `benchmarks/results/gpu_crossover_<timestamp>.json`
- `docs/benchmarks/gpu_crossover_<timestamp>.md`

脚本会记录 `nvidia-smi` 快照、JAX backend、设备列表、每组 p50/p95 计时，并给出保守的阈值建议。

## 如何自定义阈值

### 运行时设置

```python
import omnilss.config as cfg

cfg.set_crossover("gpu", n=50_000, family="NO")
cfg.set_crossover("gpu", n=100_000)          # default fallback
cfg.set_crossover("tpu", n=10_000, family="NO")
cfg.crossover_summary(verbose=True)
```

### 临时实验

```python
import omnilss.config as cfg
from omnilss import NO, gamlss

with cfg.crossover_config(gpu={"NO": 50_000, "default": 100_000}):
    model = gamlss("y ~ x", family=NO(), data=data, method="auto")
```

### YAML / JSON 配置文件

OmniLSS 导入 `omnilss.config` 时会加载第一个存在的配置文件：

1. `OMNILSS_CONFIG_FILE` 指定路径
2. 当前工作目录 `./omnilss_config.yaml`
3. 用户主目录 `~/.omnilss/config.yaml`

示例：

```yaml
auto_method_enabled: true
force_jax: false

gpu_crossover_n:
  NO: 50000
  GA: 80000
  default: .inf

tpu_crossover_n:
  NO: 10000
  default: .inf
```

环境变量优先级高于配置文件：

```bash
export OMNILSS_GPU_CROSSOVER_N='NO=50000,GA=80000,default=inf'
export OMNILSS_TPU_CROSSOVER_N='NO=10000,default=inf'
export OMNILSS_AUTO_METHOD=1
export OMNILSS_FORCE_JAX=0
```

## TPU 配置指南

TPU 阈值尚未完成实测。当前 `TPU_CROSSOVER_N` 已为所有 JAX 支持族提供显式禁用阈值（`math.inf`）：NO、GA、PO、BI、WEI、TF 和 `default`。

推荐流程：

1. 在目标 TPU v2/v3/v4 环境运行与 GPU 相同的 `(n, p, family)` 矩阵。
2. 记录 JAX/JAXLIB/XLA/拓扑信息和 OOM 或编译失败行。
3. 对每个 `(family, p)` 找到 `time(RS_JAX) < time(RS)` 的最小 `n`。
4. 若 `p` 影响小于 2x，取各 `p` 阈值的保守上界写入 `TPU_CROSSOVER_N` 或 YAML；若超过 2x，先完成二维阈值设计再启用自动切换。

在 TPU 数据不足之前，不要把默认阈值改成有限值；需要临时试验时使用运行时配置或 YAML/JSON 配置文件，而不是修改包内默认值。

## 显式方法覆盖

用户始终可以绕过自动路由：

```python
# 始终使用 NumPy RS（安全、默认、支持全部族和平滑项）
model = gamlss("y ~ x", family=NO(), data=data, method="RS")

# 强制 JAX RS（仅支持 NO/GA/PO/BI/WEI/TF，且不支持平滑项）
model = gamlss("y ~ x", family=NO(), data=data, method="RS_JAX")

# 使用设备感知路由（默认禁用阈值时等同 RS）
model = gamlss("y ~ x", family=NO(), data=data, method="auto")
```
