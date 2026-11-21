# 快速上手（Quickstart）

[English version](quickstart.md)

OmniLSS 是 R `gamlss` 生态的 Python/JAX 实现，面向位置、尺度与形状广义加性模型
（GAMLSS）：可同时为分布的每一个参数（`mu`、`sigma`、`nu`、`tau`）建立含线性项、
平滑项与随机效应的回归模型。

## 安装与环境管理

本项目使用 `uv` 管理虚拟环境（Windows CPU 使用 `.venv`，WSL Debian GPU 使用
`debianjaxgpu`）：

```bash
# 以可编辑模式安装核心包
uv pip install -e omnilss

# 可选扩展：gRPC 与 FastAPI 托管服务
uv pip install -e "omnilss[server]"
```

进行 R 数值对齐时，请启用 64 位精度（`JAX_ENABLE_X64=true`）。

## 最小示例

```python
import numpy as np
from omnilss import GA, gamlss

rng = np.random.default_rng(0)
n = 120
x = rng.normal(size=n)
# 模拟：mu = exp(1.0 + 0.5 * x)，sigma = 0.3
mu_true = np.exp(1.0 + 0.5 * x)
y = rng.gamma(shape=1.0 / 0.09, scale=mu_true * 0.09)

model = gamlss("y ~ x", family=GA(), data={"y": y, "x": x})
print("converged:", model.converged, "deviance:", float(model.g_dev))
print("mu 系数:", model.coefficients["mu"])
print("sigma:", float(np.exp(model.coefficients["sigma"][0])))
```

## 预测与条件分位数

```python
newdata = {"x": np.linspace(-2.0, 2.0, 5)}
mu_hat = model.predict(newdata=newdata)
params = model.predict_params(newdata=newdata)          # {"mu": ..., "sigma": ...}
quantiles = model.predict_quantiles(newdata=newdata, quantiles=[0.1, 0.5, 0.9])
```

## 当前范围与核心特性

1. **分布族**：共注册 146 个分布族（0 个占位族），其中 22 个高频族通过全套实时 R
   golden、预测回环与分层路由一致性校验并晋升 `production_safe`（`BE`、`BEINF`、
   `BI`、`EXP`、`GA`、`GEOM`、`GU`、`IG`、`IGAMMA`、`LO`、`LOGNO`、`LOGNO2`、
   `NBI`、`NBII`、`NET`、`NO`、`NO2`、`PE`、`PO`、`RG`、`WEI`、`ZAGA`）。
2. **拟合算法**：`method="RS"`（默认）、`"CG"`（18 个族经实时 R 验证）、`"MIXED"`
   （R `mixed(n1, n2)` 两阶段 RS -> CG）、`"HYBRID"` 及实验性 `"RS_JAX"`。
3. **平滑器与随机效应**：公式内原生支持 `pb(x)`、`ps(x)`、`cs(x)`、`cy(x)`、
   `lo(x)`、`re(random=...)` 与 `random(g)`。
4. **范围内 R 符号覆盖**：11 个范围内 R 包全部 1,036 个符号实现 100% 表面覆盖且
   0 未验证符号；`gamlss2` 明确不纳入 1.0 范围。

## 下一步

- [分布族与能力门禁](tutorial-families_cn.md)
- [拟合算法（RS / CG / MIXED / HYBRID）](tutorial-fitting-algorithms_cn.md)
- [平滑器与随机效应](tutorial-smoothers_cn.md)
