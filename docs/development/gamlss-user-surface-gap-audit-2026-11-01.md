# gamlss() 用户面调用面缺口审计（D2.130）

> 日期：2026-11-01　活 R：R 4.6.0 + gamlss 5.5.0（`names(formals(gamlss))` 实测）。
> 目的：回答“R 用户习惯的入口，我们缺哪些”，并把缺口拆成可交付任务。
> 结论先行：**拟合主路径（formula / sigma / nu / tau / family / data / weights / method / control）齐备；
> 缺的是“从哪里开始”和“把某个参数钉住”这两类能力（start.*、*.fix、start.from、contrasts）。**

## 1. 入口参数对照

R 的 `gamlss()` 形参（实测顺序）：

```
formula, sigma.formula, nu.formula, tau.formula, family, data, weights, contrasts, method,
start.from, mu.start, sigma.start, nu.start, tau.start, mu.fix, sigma.fix, nu.fix, tau.fix,
control, i.control, ...
```

我们的 `OmniLSS.gamlss()`：

```
formula, sigma_formula, family, data, weights, parameter_formulas, method, control, i_control,
verbose, strict_capabilities, allow_compat_alias, optimizer, learning_rate, max_iter,
history_size, optimizer_kwargs
```

| R 参数 | 状态 | 我们的对应路径 / 证据 |
|---|---|---|
| `formula` | ✅ | 同名；mu 公式 |
| `sigma.formula` | ✅ | `sigma_formula`（含 `~1` 默认） |
| `nu.formula` / `tau.formula` | ✅ 等价 | `parameter_formulas={"nu": "~1", "tau": "~1"}`，键按小写参数名（`_fitting_utils._resolve_parameter_formulas`） |
| `family` | ✅ | 接受族对象或注册表名 |
| `data` | ✅ | dict / DataFrame；族固定参数（BI/DBI 的 `bd`）也从这个 dict 取 |
| `weights` | ✅ | 同名 |
| `method` | ✅ | `RS` / `CG` / `MIXED` / `joint`；RS 在 GPU 后端另有分支 |
| `control` / `i.control` | ✅ | 同名（`control` 管外层，`i_control` 管内层） |
| `...`（`trace`、`n.cyc` …） | ⚠️ 部分 | 我们暴露 `verbose` / `max_iter`，R 的 `n.cyc` 需要转写 |
| `mu.start` / `sigma.start` / `nu.start` / `tau.start` | ❌ 缺 | 初值只能由 `_fitting_init.py` 按族决定，用户无法覆盖 |
| `mu.fix` / `sigma.fix` / `nu.fix` / `tau.fix` | ❌ 缺（用户面） | 内核已支持**族级**固定参数（`family.fixed_parameters` + `data` 里的同名列，`fitting._resolve_fixed_parameter_values`），但没有“把 mu 固定在常数/formula” 的用户入口 |
| `start.from` | ❌ 缺 | 无热启动入口；`operations.py` 只有个未接线的 `start_from` 类型标注 |
| `contrasts` | ❌ 缺 | 全库无实现；因子编码走公式解析器的默认处理 |

## 2. 影响面（为什么值得补）

1. **`start.*` 是 R 用户处理难收敛族的标准手段**。我们的 D2.38 已证明“初值决定轨迹”（Box-Cox/JSU/NBII/SI 全靠对齐 R 初值才 pass），
   但用户拿到难收敛数据时**没有任何旋钮**——只能改族或改数据。
2. **`start.from` 决定了“重拟合/滚动窗口”的成本**。缺它意味着每次都要从零开始，且 D2.117 的“逐族固定初值”无法被用户复用。
3. **`mu.fix`/`sigma.fix` 是 offset/固定尺度类建模的入口**（R 里常用 `mu.fix=TRUE, mu.start=rep(c, n)` 做已知偏移）。
   内核已有 `fixed_parameter_values` 通道（CG/RS/JAX 三条路径都接了），缺的只是 API 层解析，**性价比最高**。
4. `contrasts` 只在有因子的公式上生效；我们的平滑/因子路径本来就有限，优先级最低。

## 3. 拆解为任务（建议顺序）

| ID | 任务 | 工作量估计 | 验收标准 |
|---|---|---|---|
| D2.131 | `mu.fix/sigma.fix/nu.fix/tau.fix` + `*.start` 用户入口 | 中 | R 的三种典型写法各一条 live-R 对照：①`mu.fix=TRUE, mu.start=c`（截距固定）②`sigma.fix=TRUE, sigma.start=c` ③`mu.start` 只给起点不改模型；三条 deviance 对上活 R 1e-6 |
| D2.132 | `start.from` 热启动 | 中 | 用 R 的 `start.from` 结果（系数表）作起点，两个模型的第二次拟合迭代数下降且 deviance 与一次拟合一致（≤1e-8） |
| D2.133 | `n.cyc` / `trace` 兼容别名 | 小 | `gamlss(..., n.cyc=5)` 与 `max_iter=5` 等价；`trace=TRUE` 走 `verbose` |
| D2.134 | `contrasts` | 大（依赖公式解析器） | 因子响应/因子协变量在 `contrasts=list(factor="contr.sum")` 下与 R 系数布局一致；无因子时报错不静默 |

## 4. 不在本轮范围

- `gamlssML` / `gamlssCG` 的独立入口名（R 里是薄壳，我们有自己的 `method` 路由）。
- R 的 `predict.gamlss` 的 `type="terms"` / `se.fit` 细节（属 Phase 3 预测面，已有 schema 契约）。
