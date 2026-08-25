# Count Distributions — PO, NBI, NBII, GEOM

[中文版本](04_count_distributions_cn.md)

> 本教程校准至当前 OmniLSS 拟合 API（`gamlss(..., method="RS")` 默认路径）。
> 覆盖分布：**Poisson (PO)**、**Negative Binomial I (NBI)**、**Negative Binomial II (NBII)**、**Geometric (GEOM)**。
> 目标读者：需要建模计数/次数响应的用户。

## 范围与目标

| 主题 | 说明 |
|------|------|
| 计数分布族 | `PO`、`NBI`、`NBII`、`GEOM` |
| 参数化 | `mu`（均值）、`sigma`（离散参数，用于 NBI/NBII） |
| 链接函数 | `mu` 用 `log`（R 一致），`sigma` 用 `log` |
| 关键场景 | 过离散（overdispersion）建模、事件次数、索赔频率、通话次数 |

## 家族参数化对照（Python ↔ R）

| 家族 | Python 参数 | 链接（Python = R） | 类型 |
|------|-------------|--------------------|------|
| `PO` | `(mu,)` | `mu: log` | Discrete |
| `NBI` | `(mu, sigma)` | `mu: log, sigma: log` | Discrete |
| `NBII` | `(mu, sigma)` | `mu: log, sigma: log` | Discrete |
| `GEOM` | `(mu,)` | `mu: log` | Discrete |

> 说明：`NBII` 与 `NBI` 采用不同的方差-均值关系，`sigma` 在两者中含义不同；`PO` 与 `GEOM` 均为单参数分布（`GEOM` 可视为 `NBI` 的 `sigma=1` 特例）。

## 数据集构造

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 500

# 线性预测器：计数数据用 log 链接
x = rng.normal(size=n)
eta = 1.0 + 0.5 * x
mu = np.exp(1.0 + 0.5 * x)          # 平均事件数

# 生成计数响应
y = rng.poisson(mu)

data = {"y": y, "x": x}
```

## 拟合模型

```python
from omnilss import gamlss, gamlss_control

model_po = gamlss(
    "y ~ x",
    family="PO",
    data=data,
    method="RS",
    control=gamlss_control(n_cyc=20, c_crit=1e-4),
)

print(f"Global deviance: {model_po.g_dev:.4f}")
print(f"AIC: {model_po.additional_slots['aic']:.2f}")
print("mu coefficients:", model_po.coefficients["mu"])
```

## 过离散检验与 NBI 拟合

```python
# 用 NBI 引入额外的离散参数 sigma，检验过离散
model_nbi = gamlss(
    "y ~ x",
    family="NBI",
    data=data,
    method="RS",
    control=gamlss_control(n_cyc=20, c_crit=1e-4),
)

aic_po = model_po.additional_slots["aic"]
aic_nbi = model_nbi.additional_slots["aic"]
print(f"PO AIC = {aic_po:.2f}, NBI AIC = {aic_nbi:.2f}")

# 拟合后的 sigma（离散度）系数（log 尺度）
print("NBI sigma coefficients:", model_nbi.coefficients["sigma"])
```

## 预测

```python
newdata = {"x": np.array([-1.0, 0.0, 1.0])}

# 预测分布参数
params = model_nbi.predict_params(newdata)
print("predicted mu:", params["mu"])
print("predicted sigma:", params["sigma"])

# 预测响应（均值）
y_pred = model_nbi.predict(newdata, type="response")
print("predicted response:", y_pred)

# 预测分位数
q = model_nbi.predict_quantiles(newdata, quantiles=[0.05, 0.5, 0.95])
print("5%/50%/95% quantiles:", q[0.05], q[0.5], q[0.95])
```

## 模型选择（AIC/BIC）

```python
sbc_po = model_po.additional_slots["sbc"]
sbc_nbi = model_nbi.additional_slots["sbc"]
print(f"PO SBC = {sbc_po:.2f}, NBI SBC = {sbc_nbi:.2f}")

# 若 NBI 的 AIC/SBC 明显更低，说明数据存在过离散，应优先 NBI。
```

## R 一致性要点

- `mu` 的 `log` 链接与 R `gamlss` 一致；`coef(model)["mu"]` ↔ `model.coefficients["mu"]`。
- R 的 `sigma.formula` ↔ Python `sigma_formula`（本教程未使用，保持 `~1`）。
- 全局偏差 `deviance(model)` ↔ `model.g_dev`；AIC ↔ `model.additional_slots["aic"]`。
- 拟合参数值（反链接后）在 `model.fitted_values[parameter]`。

## 练习

1. 把 `n_cyc` 改为 `50`，观察 AIC/全局偏差是否收敛。
2. 用 `NBII` 替换 `NBI` 重跑，比较两者 AIC，说明 `NBII` 的方差-均值关系对过离散的影响。
3. 引入一个无关的预测变量 `z`，比较加入前后 AIC 的变化，验证模型选择。