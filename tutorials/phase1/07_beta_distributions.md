# Beta and Bounded Distributions — BE, BEINF, BEZI, BEOI

[中文版本](07_beta_distributions_cn.md)

> 本教程校准至当前 OmniLSS 拟合 API（`gamlss(..., method="RS")` 默认路径）。
> 覆盖分布：**Beta (BE)**、**Beta Inflated (BEINF)**、**Beta Zero-Inflated (BEZI)**、**Beta One-Inflated (BEOI)**。
> 目标读者：需要建模 (0,1) 区间的比例/成分数据，或带边界堆积的比例数据用户。

## 范围与目标

| 主题 | 说明 |
|------|------|
| Beta 族 | `BE`、`BEINF`、`BEZI`、`BEOI` |
| 参数化 | `BE`: `(mu, sigma)`；`BEINF`: `(mu, sigma, nu, tau)`；`BEZI`/`BEOI`: `(mu, sigma, nu)` |
| 链接函数 | `mu` 用 `logit`（R 一致） |
| 关键场景 | 满意率、市场份额、得病率、投入比例 |

## 家族参数化对照（Python ↔ R）

| 家族 | Python 参数 | 链接（Python = R） | 类型 |
|------|-------------|--------------------|------|
| `BE` | `(mu, sigma)` | `mu: logit, sigma: logit` | Continuous |
| `BEINF` | `(mu, sigma, nu, tau)` | `mu: logit, sigma: logit, nu: log, tau: log` | Mixed |
| `BEZI` | `(mu, sigma, nu)` | `mu: logit, sigma: log, nu: logit` | Mixed |
| `BEOI` | `(mu, sigma, nu)` | `mu: logit, sigma: log, nu: logit` | Mixed |

> 说明：`BE` 的 `sigma` 为精度参数（R 参数化 `a=mu*sigma`, `b=(1-mu)*sigma`）。`BEINF`/`BEZI`/`BEOI` 在 0 或 1 处有概率堆积，`nu`（`BEZI`/`BEOI`）或 `(nu, tau)`（`BEINF`）控制堆积概率。

## 数据集构造

```python
import numpy as np
from omnilss import gamlss, gamlss_control

rng = np.random.default_rng(3)
n = 400

x = rng.normal(size=n)
# mu 用 logit 反链接 → (0,1) 比例
mu = 1.0 / (1.0 + np.exp(-(0.0 + 0.7 * x)))
sigma = 3.0                                   # 精度参数（常量）
a = mu * sigma
b = (1.0 - mu) * sigma

y = rng.beta(a, b)                            # (0,1) 区间的 Beta 响应

data = {"y": y, "x": x}
```

## 拟合模型（BE）

```python
model_be = gamlss(
    "y ~ x",
    family="BE",
    data=data,
    method="RS",
    control=gamlss_control(n_cyc=30, c_crit=1e-5),
)

print(f"Global deviance: {model_be.g_dev:.4f}")
print(f"AIC: {model_be.additional_slots['aic']:.2f}")
print("mu coefficients:", model_be.coefficients["mu"])
print("sigma coefficients:", model_be.coefficients["sigma"])
```

## 边界堆积：BEINF

```python
# 若数据在 0 或 1 处有堆积，用 BEINF 引入 (nu, tau) 边界概率
# 首先生成带堆积的数据
y_infl = y.copy()
n_zero = int(n * 0.15)
n_one = int(n * 0.05)
y_infl[:n_zero] = 0.0
y_infl[n_zero:n_zero+n_one] = 1.0

data_infl = {"y": y_infl, "x": x}

model_beinf = gamlss(
    "y ~ x",
    family="BEINF",
    data=data_infl,
    method="RS",
    control=gamlss_control(n_cyc=30, c_crit=1e-5),
)

aic_be = model_be.additional_slots["aic"]
aic_beinf = model_beinf.additional_slots["aic"]
print(f"BE AIC = {aic_be:.2f}, BEINF AIC = {aic_beinf:.2f}")
print("BEINF nu coefficients:", model_beinf.coefficients["nu"])
print("BEINF tau coefficients:", model_beinf.coefficients["tau"])
```

## 预测与分位数

```python
newdata = {"x": np.array([-1.0, 0.0, 1.0])}

params = model_be.predict_params(newdata)
print("predicted mu:", params["mu"])
print("predicted sigma:", params["sigma"])

q = model_be.predict_quantiles(newdata, quantiles=[0.05, 0.5, 0.95])
print("5%/50%/95% quantiles:", q[0.05], q[0.5], q[0.95])
```

## R 一致性要点

- `mu` 的 `logit` 链接、`sigma` 为精度参数，与 R `gamlss.dist` 的 `BE()` 家族一致。
- R 的 `coef(model)["mu"]` ↔ `model.coefficients["mu"]`。
- 全局偏差 `deviance(model)` ↔ `model.g_dev`；AIC ↔ `model.additional_slots["aic"]`。

## 练习

1. 用 `BEZI` 和 `BEOI` 分别拟合堆积在 0 或 1 的数据，比较 AIC 与边界概率估计。
2. 改变 `sigma`（精度）的值，观察拟合的 `sigma` 系数与数据离散度之间的关系。
3. 使用 `model.predict_quantiles` 生成条件中位数曲线，检查比例数据的预测合理性。