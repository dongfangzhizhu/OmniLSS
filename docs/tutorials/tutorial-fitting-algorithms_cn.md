# 拟合算法：RS / CG / MIXED / HYBRID / RS_JAX

[English version](tutorial-fitting-algorithms.md)

OmniLSS 实现了 R `gamlss` 的核心估计算法及实验性 XLA 编译 RS 后端，通过 `method=` 选择。

## RS（Rigby & Stasinopoulos，默认）

- 按分布参数交替迭代，固定其余参数并对当前参数执行与 R `glim.fit` 对齐的修正 IRLS 更新。
- 已在全部 22 个 `production_safe` 族与 `ZIP` 上通过实时 R 验证（`rs_fit=validated`）。

```python
model = gamlss("y ~ x", family="GA", data=data)              # 默认 RS
model = gamlss("y ~ x", family="GA", data=data, method="RS")
```

## CG（Cole & Green）

- 利用跨参数期望信息块进行联合牛顿型更新，对参数相关性较强的模型收敛更稳健。
- 在 `docs/development/layered-parity-2026-11-15.json` 中 18 个直接与实时 R 通过对拍的
  族（`BE`、`BEINF`、`BI`、`EXP`、`GA`、`GEOM`、`GU`、`IG`、`IGAMMA`、`LO`、`LOGNO`、
  `LOGNO2`、`NET`、`NO`、`NO2`、`PO`、`WEI`、`ZAGA`）已晋升 `cg_fit=validated`。

```python
model = gamlss("y ~ x", family="BE", data=data, method="CG")
```

## MIXED（R `mixed(n1, n2)` 两阶段算法）

- 严格按先 `mixed_n1` 轮 RS、再以完整 RS 状态热启动 `mixed_n2` 轮 CG 的顺序运行。
- 最终 `iter` 与 `converged` 采用 CG 阶段语义，双阶段诊断均写入 `model.additional_slots`。
- CG/MIXED 路径遇到平滑项时会显式拒绝；含平滑项的模型请使用 `RS` 或 `HYBRID`。

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

## HYBRID 与实验性 RS_JAX

- `method="HYBRID"` 根据样本量、分布族与平滑项自动在 RS 与 L-BFGS 之间选择（不等同于 R `mixed()`）。
- `method="RS_JAX"` 将整个 RS 循环编译到 XLA，支持 6 个核心族（`NO`、`GA`、`PO`、`BI`、
  `WEI`、`TF`，当前为 `experimental` 层级）。

```python
model_h = gamlss("y ~ pb(x)", family="NO", data=data, method="HYBRID")
model_j = gamlss("y ~ x", family="NO", data=data, method="RS_JAX")
```

## 收敛控制与数值守卫

```python
from omnilss.controls import gamlss_control

control = gamlss_control(n_cyc=100, c_crit=1e-3, trace=False)
model = gamlss("y ~ x", family="GA", data=data, control=control)
```

- 参数固定（`mu_fix` / `sigma_fix` / `nu_fix` / `tau_fix`）与分类因子对比编码
  （`contr.treatment` / `contr.sum` / `contr.helmert` / `contr.poly`）在 `RS`、`CG`、
  `MIXED` 全路由生效。
- 所有数值守卫（线性预测子裁剪、Hessian ridge 稳定化）均记录在
  `model.additional_slots["safeguard_events"]` 中以供审计。

## 下一步

- [平滑器与随机效应](tutorial-smoothers_cn.md)
