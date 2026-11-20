# P5-1：OmniLSS ↔ ondil 分布契约（v0.1，文档阶段）

> 状态：**契约草案，已核对两侧源码**（D2.269）。本文只定义契约与差异，不含适配层实现。
> 后续：P5-1b 适配层 + 一致性扫描工件，P5-1c 独立流式 golden。
> 轨道说明：ondil 是**在线层参考实现**，只作设计 oracle，不改变 R 优先级；L1 的 R golden
> 只证明 kernel，不证明在线 estimator —— 流式部分的验收必须独立。

## 1. 两侧现状（事实）

| | OmniLSS | ondil（`ondil/`，vendored） |
|---|---|---|
| 分发形态 | 146 个 `FamilyDefinition`，dict 参数 + R 风格 `d/p/q/r` | 22+ 个 `Distribution` 子类，`theta` 矩阵 + 方法 |
| 契约入口 | `omnilss/core/distributions/protocol.py:25`（`DistributionProtocol`） | `ondil/src/ondil/base/distribution.py:16`（`Distribution` ABC） |
| 适配现状 | `core/distributions/adapters.py:86`（`FamilyDistributionAdapter`，迁移桥） | 各分布直接继承 ABC / `ScipyMixin` |
| 参数表达 | `Mapping[str, ArrayLike]`，按名字、可广播 | `theta: (n_rows, n_params)`，按列序 |
| 约束/链接 | `Constraint`（validate/project）+ `Link`（transform/inverse/derivative） | `parameter_support`/ `link_support` 元组 + `LinkFunction` 五方法 |
| 导数 | `score()` / `hessian()` 返回嵌套 dict | `dl1_dp1` / `dl2_dp2` / `dl2_dpp` 按参数下标 |
| 在线估计 | 无 | `partial_fit` / `update` / forget factor（`base/estimator.py:34`） |
| **可导入性** | 是 | **否**：`ondil/pyproject.toml:39` 硬依赖 `scikit-learn>=1.5` + `numba`，本仓库 venv 两者都没有 → ondil 只能作**源码参考**，不能作运行时 oracle |

**重叠面（已逐族核对）**：ondil 的 22 个 `corresponding_gamlss` 名称在 OmniLSS 全部可解析，
且**参数名与顺序 1:1 一致**（无一处错位）：

`BE BEINF BEINF0 EXP GA GU IG IGAMMA JSUo LO LOGNO LOGNO2 NO NO2 PE PO RG ST3 SST TF WEI ZAGA`

（`JSU` 在 OmniLSS 有、ondil 只有 `JSUo`；ondil 的 `mv_*` 多元分布无 GAMLSS 名，不在本表。）

## 2. 数据模型：`theta` ↔ `params`

契约固定为**列序 = 族参数声明序**：

```
theta[:, j]  ↔  params[family.parameters[j]]     j = 0 .. p-1
```

- OmniLSS 侧唯一真源是 `FamilyDefinition.parameters`（并且 `estimable_parameters` 可能更短，
  例如 `BB` 的 `bd` 是数据参数、`NET` 的 `nu/tau` 固定不估）。**theta 只承载 estimable 参数。**
- adapter 交换 `theta` 时必须逐名比对 `family.estimable_parameters == tuple(ondil.parameter_names.values())`，
  **不允许按位置直接塞**：22 个族今天一致不代表第 23 个也一致，错位会静默给出错误梯度。
- 广播：OmniLSS 参数可以是标量/向量/矩阵；进入 `theta` 前统一广播到 `(n_rows,)`，
  多元参数（矩阵/LowerTriangular）在 P5-1b 之前**明确不支持**。

## 3. 字段映射表

| ondil 成员（`base/distribution.py`） | OmniLSS 对应 | 语义 | 差异处理 |
|---|---|---|---|
| `parameter_names` | `family.parameters` | 列序 | 名字集合必须相等；顺序必须相等 |
| `parameter_shape` | 无 | 每参数结构 | 标量族一律 `SCALAR`；多元推迟 |
| `distribution_support` | **无对应字段** | 响应支撑 | **需新增**（见 §9、§15-Q1） |
| `parameter_support` | `Constraint` | 参数域 | 只比较开闭/有限性，不比字面 eps（§9） |
| `links`（`dict[int, LinkFunction]`） | `Parameter.link` / `family.links`（名字） | 链接 | 4 个方法的方向差异见 §4 |
| `logpdf`/`logpmf` | `d(..., log=True)` | 对数密度/质量 | OmniLSS 对数空间是权威，禁止 `log(density)` |
| `pdf`/`pmf` | `d(...)` | 密度/质量 | 仅用于展示；golden 只比对数 |
| `cdf` | `p(y, ...)` | 下尾概率 | — |
| `logcdf` | `p(y, ..., log_p=True)` | 对数下尾 | OmniLSS 原生支持 `log_p`/`lower_tail`；ondil 侧若无非对数实现则标 unsupported，不得 `log(cdf)` 回退 |
| `ppf`/`quantile` | `q(p, ...)` | 分位数 | `quantile` 是 `ppf` 的别名（`distribution.py:181`） |
| `rvs(size, theta)` | `r(key, n, ...)` | 抽样 | RNG 语义不同（§8） |
| `dl1_dp1(y, theta, param)` | `score()[name]` | ∂ℓ/∂θ_j，**逐观测** | OmniLSS 现实现有缺陷（§6、§10-D2） |
| `dl2_dp2(y, theta, param)` | `hessian()[name][name]` | ∂²ℓ/∂θ_j² | 同上 |
| `dl2_dpp(y, theta, (i,j))` | `hessian()[name_i][name_j]` | 交叉二阶导 | ondil 传 `i == j` 直接报错 |
| `initial_values(y)` | `init_params(y)` | 初始值 | ondil 返回 `(n, p)` 平铺；OmniLSS 返回逐参数标量 |
| `calculate_conditional_initial_values` | 无 | 顺序 warm start | P5-1c 再定 |
| `is_discrete` | `family.type` | 离散标记 | `Discrete`→`True`；`Continuous`→`False`；**`Mixed`→`False`**（与 ondil 一致：`BEINF`/`ZAGA` 只重写 `pdf`/`logpdf` 的原子质量，只有 `PO` 设 `is_discrete=True`；`pmf`/`logpmf` 对 Mixed 标 unsupported） |
| `corresponding_gamlss` | `family.name` | 互查键 | 用**族名**互查，不用类名 |
| `crps` | 无 | 评分规则 | 可选，依赖 `scoringrules`，**不进入 golden** |

## 4. 链接函数：方向陷阱（本契约最容易写错的地方）

两侧都有"链接导数"，但**方向相反**：

```
ondil  Log:  link(x)=log x, inverse(η)=exp η,
             inverse_derivative(η) = exp η   = dμ/dη
             link_derivative(x)    = 1/x     = dη/dμ
OmniLSS LogLink: transform(v)=log v, inverse(η)=exp η,
             derivative(η) = exp η          = dμ/dη   ← 对应 inverse_derivative
```

（`ondil/src/ondil/links/loglinks.py:21-33`、`omnilss/core/links/__init__.py:60`）

因此：

| 契约方向 | ondil | OmniLSS |
|---|---|---|
| μ = g⁻¹(η) | `inverse` | `inverse` |
| η = g(μ) | `link` | `transform` |
| **dμ/dη** | `inverse_derivative` | `derivative` |
| dη/dμ | `link_derivative` | 不存在，需 `1/derivative` |
| d²η/dμ² | `link_second_derivative` | 不存在，需 `-derivative'(η)/derivative(η)³` |

链接名映射（OmniLSS 族声明 → 规范 `Parameter` → ondil 类）：

| OmniLSS 名字 | `core.params` 规范对象 | ondil 类 | 备注 |
|---|---|---|---|
| `identity` | `IdentityLink` / `Real` | `Identity` | 双向可逆，无差异 |
| `log` | `LogLink` / `Positive` | `Log` | 见上面的方向陷阱 |
| `logit` | `LogitLink` / `UnitInterval` | `Logit` | `derivative`=`p(1-p)` ↔ `inverse_derivative` |
| `softplus` | `SoftplusLink` | **无** | 显式标 unsupported，禁止用 Log 顶替 |
| — | 无 | `LogShiftValue`/`LogShiftTwo`/`LogIdent` | ondil 独有（`TF` 的 `nu>2` 等），OmniLSS 侧若需要另开能力项 |

**强制校验**：映射后必须满足 `link_support ⊆ parameter_support`（ondil 只在构造时 warn，
不报错 —— 契约要求 adapter 直接 raise）。

## 5. 密度与分布函数约定

1. **对数空间优先**：golden 只比 `logpdf`/`logpmf`/`logcdf`；密度值仅用于绘图。
   原因见 D2.53：`log K_ν` 的钳位曾把 SI 密度放大 1.04e4 倍，而密度值看起来"只是大一点"。
2. **尾参数**：OmniLSS `p/q` 的 `lower_tail`/`log_p` 是契约的一部分，映射 `logcdf` 用
   `p(..., log_p=True)`；上尾用 `lower_tail=False, log_p=True`（避免 `log(1-p)` 抵消）。
3. **支撑集外**：离散族在非整数点返回 0（密度）/ `-inf`（对数），**不返回 NaN**；
   `NaN` 只用于参数非法（如 `sigma<=0`）。两侧必须一致，golden 显式覆盖支撑集外点。
4. **离散支撑**：`distribution_support` 的下界是**闭**的（`PO` 是 `[0, ∞)`），
   连续族上下界开。`Mixed`（零膨胀/膨胀）的响应支撑仍是 `[0, ∞)` 或 `[0, 1]`。
5. **dtype**：交换数组一律 `float64`（x64 已 pin）；整数响应保留整数用于 `pmf` 分支判断。

## 6. 导数约定（最需要修的一处）

契约：**score/hessian 一律逐观测、按参数名、对对数似然求导，不求和**。

```
score[name]        shape (n,)      = ∂ℓ_i/∂θ_name
hessian[a][b]      shape (n,)      = ∂²ℓ_i/∂θ_a∂θ_b     （a、b 均可取全部 estimable 参数）
```

现状偏差（`omnilss/core/distributions/adapters.py`）：

- `score()` 的 `score_functions` 分支返回逐观测数组 ✓；**autodiff 分支先 `jnp.sum(...)` 再求导，
  返回的是标量**（:105-122）→ 形状随族实现路径变化，属于**契约违反**，必须在 P5-1b 修掉。
- `hessian()` 的 `hessian_functions` 分支只填对角 `{name: {name: arr}}`，交叉项缺失（:124-143）；
  autodiff 分支同样求和成标量 → `dl2_dpp` 无法映射。

数值判据（P5-1b 的 golden）：

| 比较项 | 容差 | 说明 |
|---|---|---|
| `logpdf`/`logpmf` | `1e-12` 绝对（对 ℓ 量级 `|ℓ|<1e3`） | 两侧都走解析式 |
| `cdf`/`ppf` | `1e-10` 相对 | `ppf` 用二分/牛顿，误差放大 |
| `score` | `1e-10` 相对 | 与 `jax.grad` 对拍 + 与 ondil 对拍 |
| `hessian` | `1e-8` 相对 | 二阶导条件数差，容差放宽一档 |

## 7. 初始化

- OmniLSS `init_params(y)` 走 `Parameter.initializer`（`mu`→均值、`sigma`→1 等），逐参数标量；
  ondil `initial_values(y)` 返回 `(n, p)` 平铺矩阵。
- 契约：adapter 提供 `initial_theta(y) -> (n, p)`，由 `init_params` 广播得到；
  反向不提供（不把 ondil 的初值塞回 OmniLSS 拟合）。
- 初值**不是**拟合结果，golden 里只做形状/有限性/支撑检查，不比数值相等。

## 8. 随机数与可复现性

- OmniLSS：`r(key, n, ...)` 走 JAX PRNG，显式 key，可 `jit`/`vmap`。
- ondil：`rvs(size, theta)` 走 scipy 全局 `np.random`（`distribution.py:469`），无 key 参数。
- 契约：**抽样不比原始样本**。golden 用概率积分变换（PIT）比较：
  `cdf(rvs(...)) ~ U(0,1)` 的 KS 统计量 + 固定 seed 下的均值/方差区间；
  需要逐样本对齐的场合一律用 OmniLSS 侧 key（ondil 不参与）。

## 9. 支持集与约束

| OmniLSS `Constraint` | 数值域 | ondil `parameter_support` | 差异 |
|---|---|---|---|
| `Real` | `(-inf, inf)` | `(-np.inf, np.inf)` | 一致 |
| `Positive(eps=1e-12)` | `(1e-12, inf)` | `(np.nextafter(0,1), inf)` ≈ `5e-324` | **eps 不同**，只比"开区间+正" |
| `UnitInterval(eps=1e-12)` | `(1e-12, 1-1e-12)` | `(nextafter(0,1), nextafter(1,0))` | 同上 |
| 无 | — | 位移链接的下界（`LogShiftTwo` → `(2, inf)`） | OmniLSS 无对应，映射时标 unsupported |

契约：比较支持集用**规范化描述** `(closed_lower, closed_upper, finite_lower, finite_upper)`，
不比较 eps 字面值；投影行为（`project`）不进入 golden（它是数值稳定手段，不是分布定义）。

## 10. 已知缺陷清单（两侧，均已定位）

| # | 侧 | 位置 | 问题 | 影响 | 处置 |
|---|---|---|---|---|---|
| D1 | ondil | `base/estimator.py:34-60` | `OnlineEstimator.partial_fit(sample_weight=...)` 转发给 `fit(sample_weight=...)`，而 `Estimator.fit` 形参是 `sample_weights`（:90） | 基类路径 `TypeError`；子类各自用 `sample_weight` 才没暴露 | 契约统一 **`sample_weights`**；适配层不改 ondil，调用时用子类签名 |
| D2 | OmniLSS | `core/distributions/adapters.py:105-143` | `score`/`hessian` 的 autodiff 分支 `jnp.sum` 后求导 → 标量；hessian 只填对角 | 形状随实现路径漂移，`dl2_dpp` 无法映射 | P5-1b 必修（§6） |
| D3 | ondil | `base/distribution.py:430` | `ScipyMixin.logpmf` 里调的是 `super().cdf(...)` 而非 `super().logpmf(...)` | 离散/连续守卫失效：连续族调 `logpmf` 不会报错 | 契约要求 adapter 自行守卫；上游 bug 记录在案 |
| D4 | ondil | `base/distribution.py:469` | `rvs` 用 scipy 全局 RNG，无 key/random_state | 不可复现，不能进 golden | §8：抽样只做 PIT/矩检验 |
| D5 | 两侧 | `Positive` eps vs `nextafter` | 参数下界差 300 个数量级 | 极端参数下"合法/非法"判定不同 | §9：只比规范化开闭性，golden 点避开 eps 邻域 |
| D6 | OmniLSS | `core/links` | `Link.derivative` 与 ondil `link_derivative` 同名不同向 | 直接把 `link_derivative` 接到 `derivative` 会静默错梯度 | §4 表；命名建议见 §15-Q4 |
| D7 | ondil | `ondil/pyproject.toml:39` | 硬依赖 `scikit-learn>=1.5` + `numba`，本仓库 venv 都没有 → `import ondil` 直接 `ModuleNotFoundError: sklearn` | **ondil 不能作运行时 oracle**；若把"对 ondil 对拍"写成硬门禁，CI 上永远是 skip 或红 | 契约改为 scipy + 解析导数基线，ondil 直连对拍标 optional（§11、§15-Q6） |

## 11. 适配层设计草案（P5-1b，本次不实现）

```python
# omnilss/src/omnilss/ondil_contract.py（草案）
def theta_from_params(family, params) -> np.ndarray:        # (n, p)，列序 = estimable_parameters
def params_from_theta(family, theta) -> dict[str, np.ndarray]
def contract_report(family) -> dict:                        # 逐字段：mapped / unsupported / mismatch
def as_ondil(family) -> "OndilDistributionView"             # 只读视图，numpy 侧，不依赖 ondil 安装
```

- `as_ondil(...)` 返回**本包自己的 numpy 视图**，方法签名照 ondil ABC；ondil 若未安装也能跑契约扫描。
- 扫描工具 `tools/ondil_contract_scan.py` 输出 `docs/development/p5-ondil-contract-<date>.json`：
  每族一行的 `mapped/unsupported/mismatch` + 22 族的参数序一致性 + 支撑集规范化比较。
- **对拍基线（D7 决定）**：ondil 在本 venv 不可导入，所以**不能**把 "对 ondil 对拍" 当硬门禁。
  基线分两层：
  1. **解析/独立基线**（硬门禁）：scipy.stats 对应分布（ondil 的 `ScipyMixin` 同源）比
     `logpdf/cdf/ppf`；`score/hessian` 与 `jax.grad`/`jax.hessian` 对拍；`BEINF`/`ZAGA` 等
     scipy 无对应的族用**手写闭式**（与 R golden 无关，独立推导）比。
  2. **ondil 直连对拍**（optional）：仅当环境能 `import ondil` 时运行，否则 `pytest.skip`，
     并在工件里记 `ondil_available: false` —— 不允许把 skip 当成通过。
- 一致性测试 `test_ondil_contract.py`：对 22 个重叠族，在 3 组参数点 × 支撑集内/外，
  比 `logpdf/cdf/ppf/score/hessian`（容差见 §6）。

## 12. 流式 golden 契约（P5-1c，本次只定义）

**目标**：证明"在线估计器"的正确性，而不是再证一遍 kernel。

| 项 | 约定 |
|---|---|
| 入口 | `partial_fit(X, y, sample_weights=None)`；首次调用等价 `fit`，其后等价 `update` |
| 遗忘因子 | `forget ∈ [0, 1)`，权重 `γ = (1-forget)^(N-n)`（`ondil/gram.py:6`）；`forget=0` 即等权累积 |
| 分批不变性 | `update(A); update(B)` 与 `update(concat(A,B))` 的系数差 `<= 1e-10`（同一 `forget`） |
| 顺序敏感性 | `forget>0` 时**不要求**顺序不变；只要求"漂移方向正确"（后验权重更大）的定性检查 |
| warm start | `partial_fit` 首块用 `max_iter=k`，与一次性 `fit(max_iter=k)` 的系数差 `<= 1e-8` |
| 数据 | **自造 DGP**（固定 seed、解析真值），**不复用 R golden**；R 侧最多用来确认族参数化一致 |
| 缺失/漂移 | 缺失率 0/5/20% 下的偏差上界；分布漂移（参数跳变）后的追赶半衰期 |
| 性能/内存 | 单块 `update` 的 O(p²) 内存、每观测耗时不随 N 增长（斜率检验），区分 cold/warm |

判据写死为**可复跑的命令 + 容差**，工件由工具生成；不接受手写常数（D2.268 的教训：
写死在工具里的"实测值"等于誊抄）。

## 13. 验收拆分

| 项 | 内容 | 判据 | 状态 |
|---|---|---|---|
| P5-1a | 本契约文档 | 映射表逐项有源码出处；22 族参数序经脚本核对；缺陷清单可定位 | **本次交付** |
| P5-1b | 适配层 + 扫描工件 + 一致性测试 | 22 族 `logpdf/logcdf/score/hessian` 对**解析/scipy 基线**达 §6 容差（ondil 直连对拍为 optional）；工件可重生成；D2 修复 | 待做 |
| P5-1c | 流式契约 + 独立 golden | §12 七项判据全部可复跑 | 待做 |

P5-1 的整体勾选条件是 **a+b+c**；只有 a 完成时在 plan 里标 `P5-1a [x]`，不勾总项。

## 14. 未决问题（需要拍板）

- **Q1 响应支撑从哪来？** 方案 A：给 `FamilyDefinition` 加 `support: tuple[float, float]`
  （显式、可校验、146 族要补数据）；方案 B：从 `d()` 在若干点探测（无侵入、可能误判 Mixed）。
  **建议 A**，但只对 22 个重叠族先补，其余保持 `None`（映射时标 unsupported）。
- **Q2 `theta` 要不要成为 OmniLSS 公共 API？** 建议**不要**：`theta` 只存在于 adapter 边界，
  内部继续用 dict（可读、可广播），避免两套参数表示长期并存。
- **Q3 ondil 的上游 bug（D1/D3）修不修？** 建议**不修 vendored 代码**，在适配层规避并记录；
  若后续把 ondil 变成正式依赖，再单独提 issue。
- **Q4 `Link.derivative` 要不要改名？** 建议在 P5-1b 加**别名** `inverse_derivative`
  （不破坏现有调用），并在文档标 `derivative` = dμ/dη；不做全局重命名。
- **Q5 流式实现在哪一侧？** 建议 OmniLSS 侧只做**契约与验证**（P5-1c 的 golden 驱动 ondil），
  不自研在线估计器 —— 除非 Q2 决定要 `theta` 公共 API。
- **Q6 要不要为对拍把 scikit-learn + numba 装进 venv？** 建议**不装**：两个重依赖只服务于
  "对 ondil 直连对拍"，而 ondil 的数值内容在 22 个重叠族里绝大多数就是 scipy 本身（§11 基线 1）。
  若确需直连对拍，单独建一个 `ondil-oracle` 环境，不污染主 venv。
