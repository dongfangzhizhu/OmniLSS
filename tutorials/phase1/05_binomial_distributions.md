# Binomial Family — BI, BB, BNB

[中文版本](05_binomial_distributions_cn.md)

> 本教程校准至当前 OmniLSS 拟合 API（`gamlss(..., method="RS")` 默认路径）。
> 覆盖分布：**Binomial (BI)**、**Beta-Binomial (BB)**、**Beta-Negative Binomial (BNB)**。
> 目标读者：需要建模比例/成败次数响应的用户。

## 范围与目标

| 主题 | 说明 |
|------|------|
| 二项族 | `BI`、`BB`、`BNB` |
| 参数化 | `BI`/`BB` 使用固定参数 `bd`；`BNB` 使用 `(mu, sigma, nu)` |
| 链接函数 | `BI` 的 `mu` 用 `logit`；`BB`/`BNB` 的 `mu` 用 `log` |
| 关键场景 | 比例数据、试验成功次数、过度离散二项数据 |

## 家族参数化对照（Python ↔ R）

| 家族 | Python 参数 | 固定参数 | 链接（Python = R） | 类型 |
|------|-------------|----------|--------------------|------|
| `BI` | `(mu,)` | `bd`（默认 1） | `mu: logit` | Discrete |
| `BB` | `(mu, sigma, bd)` | `bd` | `mu: logit, sigma: log` | Discrete |
| `BNB` | `(mu, sigma, nu)` | — | `mu: log, sigma: log, nu: log` | Discrete |

> 说明：`bd`（试验次数/二项分母）是**固定参数**，不通过模型估计。`BI` 的 `bd` 缺省为 1（Bernoulli）；`BB` 的 `sigma` 引入试验内相关性（过度离散）；`BNB` 用 `(sigma, nu)` 建模额外的离散/长尾结构。

## 数据集构造

```python
import numpy as np
from omnilss import gamlss, gamlss_control

rng = np.random.default_rng(7)
n = 400

x = rng.normal(size=n)
p = 1.0 / (1.0 + np.exp(-(0.2 + 0.6 * x)))   # logit 反链接 → 成功概率
bd = rng.integers(5, 21, size=n)              # 每次试验次数

# 生成二项响应（mu 为成功比例）
y = rng.binomial(bd, p)

data = {"y": y, "x": x, "bd": bd}
```

## 拟合模型（BI）

```python
# bd 作为固定参数，直接从 data 读取
model_bi = gamlss(
    "y ~ x",
    family="BI",
    data=data,
    method="RS",
    control=gamlss_control(n_cyc=20, c_crit=1e-4),
)

print(f"Global deviance: {model_bi.g_dev:.4f}")
print(f"AIC: {model_bi.additional_slots['aic']:.2f}")
print("mu coefficients:", model_bi.coefficients["mu"])
```

## 过度离散与 BB 拟合

```python
# 二项数据常存在试验内相关性 → 用 BB 引入 sigma（bd 同样从 data 读取）
model_bb = gamlss(
    "y ~ x",
    family="BB",
    data=data,
    method="RS",
    control=gamlss_control(n_cyc=20, c_crit=1e-4),
)

aic_bi = model_bi.additional_slots["aic"]
aic_bb = model_bb.additional_slots["aic"]
print(f"BI AIC = {aic_bi:.2f}, BB AIC = {aic_bb:.2f}")
print("BB sigma coefficients:", model_bb.coefficients["sigma"])
```

## 预测

```python
newdata = {"x": np.array([-1.0, 0.0, 1.0]), "bd": np.array([10, 10, 10])}

params = model_bb.predict_params(newdata)
print("predicted mu:", params["mu"])
print("predicted sigma:", params["sigma"])

y_pred = model_bb.predict(newdata, type="response")
print("predicted response (成功比例):", y_pred)
```

## 模型选择

```python
sbc_bi = model_bi.additional_slots["sbc"]
sbc_bb = model_bb.additional_slots["sbc"]
print(f"BI SBC = {sbc_bi:.2f}, BB SBC = {sbc_bb:.2f}")

# 若 BB 的 AIC/SBC 更低，说明存在过度离散，BB 更合适。
```

## R 一致性要点

- `mu` 的 `logit` 链接与 R `gamlss` 一致；`bd` 固定参数与 R `bi()`/`BB()` 家族一致。
- R 的 `coef(model)["mu"]` ↔ `model.coefficients["mu"]`。
- 全局偏差 `deviance(model)` ↔ `model.g_dev`；AIC ↔ `model.additional_slots["aic"]`。

## 练习

1. 比较 `BI`、`BB`、`BNB` 三个家族的 AIC，判断数据中是否真的存在过度离散。
2. 将 `bd` 固定为常量（如 `10`），观察拟合结果与逐行 `bd` 的差异。
3. 使用 `predict_quantiles` 生成 5%/50%/95% 的成功次数分位数。