# 分布族与参数

OmniLSS 实现了 R `gamlss.dist` 的分布族。每个族有若干**分布参数**
（如 `NO` 有 `mu`/`sigma`），每个参数可以有自己的线性预测子。

## 八核心族（R 一致性已验证）

| 族 | 参数 | 典型用途 |
|---|---|---|
| `NO` | mu, sigma | 正态 |
| `GA` | mu, sigma | Gamma（正偏斜） |
| `PO` | mu | Poisson 计数 |
| `BI` | mu | 二项（0/1） |
| `NBI` | mu, sigma | 负二项（过离散计数） |
| `BE` | mu, sigma | Beta（(0,1) 比例） |
| `ZAGA` | mu, sigma, nu | 零膨胀 Gamma |
| `GU` | mu, sigma | Gumbel（极值） |

## 参数建模

每个分布参数可单独建模（与 R 相同）：

```python
from omnilss import gamlss
import numpy as np

rng = np.random.default_rng(0)
n = 100
x = rng.normal(size=n)
y = rng.normal(loc=1.0 + 0.3 * x, scale=0.5)

model = gamlss("y ~ x", sigma_formula="~1", family="NO", data={"y": y, "x": x})
```

- `mu_formula` / `sigma_formula` / `nu_formula` / `tau_formula`：各参数公式
- `family` 可传字符串（如 `"GA"`）或族对象（如 `GA()`）

## 族的能力门禁

```python
from omnilss.family_capabilities import capability_matrix, check_capability
capability_matrix()          # 全族 × 方法能力矩阵
check_capability("ZAGA", "rs_fit")
```

未实现/占位的族在 `rs_fit` 等路径上会被**拒绝**（`FamilyCapabilityError`），
避免静默错误结果。

## 参数与链接

| 参数 | 默认链接 |
|---|---|
| mu | log（计数/正支持）或 identity |
| sigma | log |
| nu / tau | logit |

## 下一步

- [拟合算法 RS/CG/MIXED](tutorial-fitting-algorithms.md)
