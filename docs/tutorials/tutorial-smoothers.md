# 平滑器：pb / ps / cs

平滑项通过公式语法加入（与 R `gamlss` 相同），在 IRLS 内部求解。

## pb（P-spline，惩罚 B 样条）

```python
model = gamlss("y ~ pb(x)", family="NO", data=data)
```

- `pb(x, df=...)`：按有效自由度选择；`pb(x, lambda=...)`：固定惩罚。

## ps（P-spline，R `ps()` 语义）

```python
model = gamlss("y ~ ps(x)", family="NO", data=data)
model = gamlss("y ~ ps(x, df=5)", family="NO", data=data)
```

- 与 R `ps()` 对齐：等间距基 + 差分惩罚；`df`/`lambda` 两种选择方式。

## cs（三次样条）

```python
model = gamlss("y ~ cs(x)", family="NO", data=data)
```

- `cs()` 在 R 中返回拟合值向量（单列表示），OmniLSS 采用相同语义，
  预测时基于保存的样条状态重建。

## 参数级平滑

每个分布参数可用不同公式：

```python
model = gamlss("y ~ pb(x)", sigma_formula="~ pb(x)", family="NO", data=data)
```

## 数值注意

- 平滑基是"隐式"的（设计矩阵内部扩展），`model.design_matrices`
  中可见实际列。
- 奇异惩罚 Gram 自动稳定（Hessian ridge），记录 safeguard 事件。

## 下一步

- [预测与 newdata](tutorial-prediction.md)
