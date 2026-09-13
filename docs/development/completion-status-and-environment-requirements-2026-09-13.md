# OmniLSS 完成度与环境需求说明

更新时间：2026-09-13

## 1. 本轮交付

- 修复 `SI`（Sichel 简化分布）公开 DPQR 不完整的问题。此前只有 log-PMF，公开 `p/q/r` 走 placeholder 并返回 `NaN`。
- 增加基于 R-SI 递推核的离散概率网格、CDF、首个交叉点分位数和逆变换采样。
- 补齐概率边界、upper-tail、log-probability、参数广播和采样测试：`tests/test_si_dpqr_contract.py`。
- 刷新代码派生的覆盖数据：`r-export-coverage.json`、`r-export-coverage.md` 和 `placeholder-migration-queue-2026-09-13.md`。

验证结果：

```text
python -m pytest tests/test_si_dpqr_contract.py tests/test_family_capabilities.py -q
21 passed
```

另已通过 Python 编译检查和 `git diff --check`。

## 2. 当前完成度

覆盖统计必须以 `docs/development/r-export-coverage.json` 为准，而不是历史手工文档。当前注册表共 145 个家族：

| 状态 | 数量 | 含义 |
|---|---:|---|
| implemented | 1 | 当前生产级证据最完整的基线家族 |
| experimental | 93 | 有真实代码/接口，但仍需更完整 R golden、拟合或性能证据 |
| compat_alias | 25 | 兼容层/别名，不计为独立 R 数学实现 |
| placeholder | 26 | 生产路由应拒绝，仍需真实数值实现或明确不支持理由 |

`SI` 与 `NBF` 已从 placeholder 队列移出并进入 experimental；这不等于已达到 production-safe。

## 3. 距离全部完成仍缺失的功能

### 3.1 直接影响分布正常使用

以下 26 个家族仍是 placeholder，当前不应宣称可用于生产拟合或预测：

`BB`、`DPO1`、`EGB2`、`EXGAUS`、`GAF`、`GB1`、`GB2`、`GIG`、`GPO`、`IGA`、`LOGSHASH`、`LQNO`、`MN3`、`MN4`、`MN5`、`NO2`、`NOF`、`PARETO`、`PARETO1`、`PE2`、`RGE`、`WEI2`、`ZALG`、`ZIP`、`ZIP2`、`ZIPF`。

其中应优先处理：

1. `NBF/GPO/NBII` 等离散计数族及 `ZIP/ZIP2/ZIPF` 零膨胀族；
2. `BCCG/BCT/BCPE` 及其变体的截断归一化；
3. 正实数族 `GB1/GB2/GIG/WEI2/PARETO`；
4. `MN3/MN4/MN5` 等尚无完整概率契约的族。

### 3.3 本轮新增交付

- `NBF` 已接入精确三参数 DPQR：按 R 源码计算 `sigma1=sigma*mu**(nu-2)`，映射为 `size=1/sigma1`，并实现 R 的小离散度 Poisson 分支、upper-tail、log-probability、广播和逆变换采样。
- 新增 `tests/test_nbf_dpqr_contract.py`；NBF/SI 契约测试共 6 项通过，综合 DPQR 测试 100% 完成（预期 placeholder 跳过项除外）。

### 3.2 影响拟合可靠性的功能

- 绝大多数 experimental 家族尚未完成固定 R 源码生成的 d/p/q/r golden 验收。
- 各家族在 RS、CG、MIXED 下的拟合一致性尚未全部闭环；已知部分 CG/MIXED case 仍存在次优收敛问题。
- ST1–ST4 已完成 DPQR 契约，但仍属于 compat/experimental 证据层，不应误报为生产级全量迁移。
- ST1/ST2 的数值积分和逐元素根求解仍较慢，需要 profile、缓存/向量化和稳定性优化。
- `SIMPLEX` 等部分家族虽然有接口，但仍需检查是否存在近似、NaN fallback 或不完整 tail 语义。

### 3.3 核心生态功能

- Phase 6 在线学习层（`ondil`）尚未完成。
- Phase 7 性能规模化仍在进行：JIT 上提、后端对齐、真实 GPU/大样本 benchmark 尚未全部完成。
- Phase 8 用户文档与教程（API 参考、Python-R 参数对照、8 篇教程校准）尚未完成。
- `gamlss()` method 名称、实现路径和历史兼容命名仍需最终审计。
- 全库 lint/type check 尚未形成严格的本地验收记录；CI 中 mypy 仍为允许失败。
- 发布检查、跨 Python 3.10/3.11/3.12、Windows/Linux/macOS 的完整矩阵仍需实际执行。

## 4. 需要外部环境或资源的需求

### 必需或强烈建议

- Python 3.10–3.12。
- JAX/JAXLIB、NumPy、SciPy、Pandas；数值一致性建议启用 `JAX_ENABLE_X64=true`。
- R 4.x、`gamlss`、`gamlss.dist`、`jsonlite`：没有这些只能运行 Python smoke test，不能声称 R 等价。
- 可联网环境或已构建的离线 devcontainer，用于安装 Python/R 依赖和生成 golden artifact。

### 按功能可选

- CUDA GPU、匹配 CUDA/JAX wheel：仅用于 GPU 性能与规模化验证，不是 CPU 核心功能的必要条件。
- `flax`/`optax`：深度 GAMLSS。
- `scikit-learn`：`GAMLSSRegressor`。
- `scoringrules`：概率评分扩展。
- `grpcio`/`protobuf`/`grpcio-tools`：gRPC 服务与 stub 生成。
- Sphinx 文档依赖：构建 API 文档与教程站点。

### CI/发布资源

- GitHub Actions runner：执行分组测试、R 一致性、lint 和跨平台构建。
- PyPI trusted publishing 权限及仓库环境：发布 Python wheel 时需要。
- 若使用离线模式，需要预先构建 `.devcontainer` 并缓存 R source tarball；离线会话不能临时访问 CRAN 或 apt。

详细安装步骤见 `docs/development/r-environment-setup.md`、
`docs/development/offline-dev-bootstrap-2026-05-17.md` 和 `README.md`。

## 5. 建议后续顺序

1. 为 `NBF/GPO/ZIP/ZIP2/ZIPF` 建立真实 d/p/q/r 与边界契约，逐族单独验收。
2. 优化 ST1/ST2 及离散族概率计算，加入 profile 基准，避免全套 R 测试超时。
3. 完成 27 个 placeholder 的迁移并为每族生成 R golden artifact。
4. 逐族完成 RS/CG/MIXED 拟合对照，再晋级 capability 状态。
5. 收尾在线层、性能/GPU 基准、method 审计、lint/type、API 文档和发布矩阵。

因此，当前项目已经具备可用的 Release 1 核心和较宽的实验性分布覆盖，但距离“R `gamlss` 全部功能完成、所有分布 production-safe”仍不是完成状态。