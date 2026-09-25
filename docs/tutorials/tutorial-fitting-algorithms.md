# 拟合算法：RS / CG / MIXED / HYBRID

OmniLSS 实现 R `gamlss` 的三种拟合算法，通过 `method=` 选择。

## RS（Rigby & Stasinopoulos，默认）

- 对每个分布参数**轮流**迭代，固定其余参数，用修改后的 IRLS 更新。
- 数值地基：显式 NumPy 后端（CPU 恒走 NumPy，GPU/大样本经路由到 JAX）。
- 用法：

```python
model = gamlss("y ~ x", family="GA", data=data)                    # 默认 RS
model = gamlss("y ~ x", family="GA", data=data, method="RS")
```

## CG（Cole & Green）

- 对各参数的得分向量与期望信息矩阵联合处理（完整牛顿型）。
- 对参数相关性强的模型（如 `BE` 的 mu/sigma）通常收敛更稳。

```python
model = gamlss("y ~ x", family="BE", data=data, method="CG")
```

## MIXED（R 兼容两阶段算法）

- 严格执行 R `mixed(n1, n2)` 的顺序：先运行 RS，再把全部可估参数状态作为 CG 的 warm start。
- 默认 `mixed_n1=1`、`mixed_n2=20`；最终 `iter` 和 `converged` 采用 CG 阶段语义，两个阶段的记录保存在 `model.additional_slots`。
- CG 尚不能保留惩罚平滑状态，因此含平滑项时会明确拒绝；此时使用 RS 或 HYBRID。

```python
model = gamlss(
    "y ~ x",
    family="NBI",
    data=data,
    method="MIXED",
    mixed_n1=1,
    mixed_n2=20,
)
```

## HYBRID（自动单后端调度）

HYBRID 保留原有的自动策略：根据样本量、family 和平滑项，在一次拟合中选择 RS 或 L-BFGS。它不是 R `mixed()`。

```python
model = gamlss("y ~ pb(x)", family="NO", data=data, method="HYBRID")
```

## 收敛控制

```python
from omnilss.controls import gamlss_control

control = gamlss_control(n_cyc=100, ccrit=1e-3, trace=True)
model = gamlss("y ~ x", family="GA", data=data, control=control)
```

- `n_cyc`：最大迭代轮数；`ccrit`：收敛判据（全局偏差相对变化）。
- 未收敛会给出 `ConvergenceWarning`，请检查 `model.converged`。

## 数值守卫

- **eta 裁剪**（`[-15, 15]`）：防止对数/逻辑链接发散。
- **Hessian 稳定**：奇异 Gram 自动加 ridge，记录到 safeguard 事件。
- 所有守卫触发均记录在 `model.additional_slots["safeguard_events"]`，
  可审计。

## 下一步

- [平滑器 pb/ps/cs](tutorial-smoothers.md)
