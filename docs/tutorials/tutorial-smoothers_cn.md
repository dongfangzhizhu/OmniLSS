# 平滑器与随机效应

[English version](tutorial-smoothers.md)

非参数平滑项与随机效应可按 R `gamlss` 语法直接写入模型公式，并在 RS 反向拟合循环内部求解。

## 惩罚 B 样条（`pb`、`ps`、`cy`）

```python
model_pb = gamlss("y ~ pb(x)", family="NO", data=data)
model_ps = gamlss("y ~ ps(x, df=5)", family="NO", data=data)
model_cy = gamlss("y ~ cy(x, df=4)", family="NO", data=data)
```

- `pb(x)` 支持自动选择平滑参数（ML / GAIC / GCV）或指定固定 `df` / `lambda`。
- `ps(x)` 与 R `ps()` 对齐（等间距 B 样条基 + 差分惩罚）。
- `cy(x)` 施加周期性循环边界条件。
- `pb()` 已在 `NO` 与 `GA` 族上通过实时 R 一致性验证（`smooth_terms=validated`）。

## 三次样条（`cs`）与局部多项式回归（`lo`）

```python
from omnilss import gamlss, gamlss_lo, lo, lo_control

model_cs = gamlss("y ~ cs(x, df=4)", family="NO", data=data)
lo_fit = gamlss_lo(x, y, w=np.ones_like(y), span=0.5, degree=1)
```

- `cs(x)` 完整保存样条状态，预测 `newdata` 时精确重建平滑曲线。
- `lo()` / `gamlss_lo()` 实现精确加权局部多项式 LOESS（0/1/2 阶、tricube 核、双权稳健迭代及
  精确帽子矩阵 `trace_hat` / `nl_df`），与 R `stats::loess` 对齐至 `1e-10`。

## 随机效应（`re`、`random`）

```python
from omnilss import gamlss_re, re

re_fit = gamlss_re(x_design, y, w=weights, groups=group_ids, method="REML")
```

- `re()` / `gamlss_re()` 基于 Woodbury + Cholesky 精确求解加权线性混合效应模型的边际 ML/REML
  （支持随机截距与随机斜率），并提供总体级（`level=0`）与分组级（`level=1`）预测。

## 参数级平滑项

不同分布参数可分别指定平滑公式：

```python
model = gamlss("y ~ pb(x)", sigma_formula="~ pb(x, df=4)", family="NO", data=data)
```

## 下一步

- [预测与序列化](tutorial-prediction_cn.md)
