# Zero-Inflated Distributions — ZIP, ZINBI, ZAGA, ZAIG

[中文版本](06_zero_inflated_cn.md)

> 本教程校准至当前 OmniLSS 拟合 API（`gamlss(..., method="RS")` 默认路径）。
> 覆盖分布：**Zero-Inflated Poisson (ZIP)**、**Zero-Inflated Negative Binomial (ZINBI)**、**Zero-Adjusted Gamma (ZAGA)**、**Zero-Adjusted Inverse Gaussian (ZAIG)**。
> 目标读者：需要建模含大量零的计数/连续非负响应的用户。

## 范围与目标

| 主题 | 说明 |
|------|------|
| 零膨胀/零调整族 | `ZIP`、`ZINBI`、`ZAGA`、`ZAIG` |
| 参数化 | `ZIP`: `(mu, sigma)`；其余: `(mu, sigma, nu)` |
| 链接函数 | `nu` 为 `logit`（零膨胀概率）；`mu`/`sigma` 为 `log` |
| 关键场景 | 索赔次数、购买次数、降雨量、费用支出（大量 0 值） |

## 家族参数化对照（Python ↔ R）

| 家族 | Python 参数 | 链接（Python = R） | 类型 |
|------|-------------|--------------------|------|
| `ZIP` | `(mu, sigma)` | `mu: log, sigma: logit` | Discrete |
| `ZINBI` | `(mu, sigma, nu)` | `mu: log, sigma: log, nu: logit` | Discrete |
| `ZAGA` | `(mu, sigma, nu)` | `mu: log, sigma: log, nu: logit` | Mixed |
| `ZAIG` | `(mu, sigma, nu)` | `mu: log, sigma: log, nu: logit` | Mixed |

> 说明：`nu` 为 logit 链接，代表**非零（或零膨胀）概率**，与 R `gamlss.dist` 一致。`ZIP` 无单独的离散参数，过离散场景应改用 `ZINBI`。

## 数据集构造

```python
import numpy as np
from omnilss import gamlss, gamlss_control

rng = np.random.default_rng(11)
n = 500

x = rng.normal(size=n)
# 零膨胀概率（logit 反链接）
p_zero = 1.0 / (1.0 + np.exp(-(-1.0 + 0.4 * x)))
mu = np.exp(1.2 + 0.5 * x)

# 生成零膨胀泊松数据
y = np.where(
    rng.random(size=n) < p_zero,
    0.0,
    rng.poisson(mu),
)

data = {"y": y, "x": x}
```

## 拟合模型（ZIP）

```python
model_zip = gamlss(
    "y ~ x",
    family="ZIP",
    data=data,
    method="RS",
    control=gamlss_control(n_cyc=20, c_crit=1e-4),
)

print(f"Global deviance: {model_zip.g_dev:.4f}")
print(f"AIC: {model_zip.additional_slots['aic']:.2f}")
print("mu coefficients:", model_zip.coefficients["mu"])
print("sigma coefficients:", model_zip.coefficients["sigma"])
```

## 过离散与 ZINBI

```python
# 若零值之外仍有过度离散，用 ZINBI 增加 sigma 离散参数
model_zinbi = gamlss(
    "y ~ x",
    family="ZINBI",
    data=data,
    method="RS",
    control=gamlss_control(n_cyc=20, c_crit=1e-4),
)

aic_zip = model_zip.additional_slots["aic"]
aic_zinbi = model_zinbi.additional_slots["aic"]
print(f"ZIP AIC = {aic_zip:.2f}, ZINBI AIC = {aic_zinbi:.2f}")
```

## 预测与分位数

```python
newdata = {"x": np.array([-1.0, 0.0, 1.0])}

params = model_zinbi.predict_params(newdata)
print("predicted mu:", params["mu"])
print("predicted sigma:", params["sigma"])
print("predicted nu:", params["nu"])

q = model_zinbi.predict_quantiles(newdata, quantiles=[0.5, 0.95])
print("50%/95% quantiles:", q[0.5], q[0.95])
```

## R 一致性要点

- `nu` 的 `logit` 链接与 R `gamlss.dist` 一致；`sigma.formula` 可建模非零部分的离散。
- R 的 `coef(model)["nu"]` ↔ `model.coefficients["nu"]`。
- 全局偏差 `deviance(model)` ↔ `model.g_dev`；AIC ↔ `model.additional_slots["aic"]`。

## 练习

1. 用 `ZAGA` 替换 `ZIP`（对非负连续数据），比较两者 AIC 与拟合参数。
2. 引入一个预测变量进 `nu` 公式（`parameter_formulas={"nu": "~ x"}` 或按签名传入），观察零膨胀概率如何随 x 变化。
3. 比较 `ZIP` 与 `ZINBI`，判断数据是否存在额外的过度离散。