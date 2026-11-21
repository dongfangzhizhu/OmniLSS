# 分布族与能力门禁

[English version](tutorial-families.md)

OmniLSS 完整移植了 R `gamlss.dist` 的分布族体系。每个分布族包含 1 至 4 个分布参数
（`mu`、`sigma`、`nu`、`tau`）、默认链接函数、解析或自动微分得分/Hessian 槽位以及
`d`/`p`/`q`/`r` 核函数。

## 22 个生产准入（`production_safe`）分布族

以下 22 个分布族全部通过了 `docs/development/m3-admission-evidence-2026-11-15.json`
的 9 项硬门禁（实时 R 导数/DPQR/RS 偏差一致性、R 一致性测试模块、golden 拟合夹具、
`newdata` 预测误差 `<= 2e-4`、JSON 往返漂移 `<= 1e-10`、重拟合收敛以及分层路由零失配）：

| 支撑集 | 生产准入分布族 | 典型场景 |
|---|---|---|
| 实数轴 `(-inf, inf)` | `NO`、`NO2`、`GU`、`RG`、`LO`、`PE`、`NET` | 对称、厚尾或偏斜连续响应 |
| 正半轴 `(0, inf)` | `EXP`、`GA`、`IG`、`IGAMMA`、`LOGNO`、`LOGNO2`、`WEI` | 正连续、持续时间、赔付金额 |
| 单位区间 `(0, 1)` / `[0, 1]` | `BE`、`BEINF` | 比例、比率及含 0/1 膨胀的比例数据 |
| 零调整 `[0, inf)` | `ZAGA` | 含零点质量的正连续数据 |
| 计数 `{0, 1, ...}` / 二项 | `PO`、`GEOM`、`NBI`、`NBII`、`BI` | 计数、过离散计数及二项试验 |

另有 104 个已注册族处于 `experimental` 层级，20 个 `gen.Family` 骨架构造器显式标记为
`compat_alias`（当前 `placeholder` 数量为 0）。

## 多参数联合建模

```python
import numpy as np
from omnilss import NO, gamlss

rng = np.random.default_rng(0)
x = rng.normal(size=120)
y = rng.normal(loc=1.0 + 0.5 * x, scale=np.exp(-0.4 + 0.25 * x))

model = gamlss(
    "y ~ x",
    sigma_formula="~ x",
    family=NO(),
    data={"y": y, "x": x},
)
```

对于 3/4 参数族，可通过 `parameter_formulas={"nu": "~ x", "tau": "~ 1"}` 指定各参数公式；
也可通过 `sigma_fix=True, sigma_start=0.5` 固定任意参数。

## Fail-Closed 能力门禁

OmniLSS 严格区分“已注册”与“已验证”。可通过 `omnilss.family_capabilities` 查询能力等级
或执行 fail-closed 生产路由校验：

```python
from omnilss.family_capabilities import (
    capability_matrix,
    get_family_capability,
    method_route_capability_report,
    require_production_family_route,
)

cap = get_family_capability("ZAGA")
print(cap.implementation_status, cap.status("rs_fit"), cap.status("cg_fit"))

# 对 experimental 或 compat_alias 路由将抛出 ProductionFamilyRejectedError
require_production_family_route("GA", "RS", route="service")
report = method_route_capability_report("GA", "CG", strict=True)
matrix = capability_matrix()
```

在 `gamlss(..., strict_capabilities=True)` 下，未验证的实验性族/路由将被硬拒绝。

## 变换族、删失族与截断族

- `gen_Family("NO", type="log")` / `Family("NO", "logit")` 构造对数/logit 变换族并自动包装导数。
- `omnilss.cens`（`gen_cens`、`cens`）与 `omnilss.tr`（`gen_trun`、`trun`）支持删失与截断分布建模。

## 下一步

- [拟合算法（RS / CG / MIXED / HYBRID）](tutorial-fitting-algorithms_cn.md)
