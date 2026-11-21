# Changelog

[中文版本](CHANGELOG_cn.md)

All notable changes to Omni GAMLSS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-11-21 (M0–M3 Closed / 100% In-Scope R Surface Coverage)

### Added
- **100% In-Scope R Symbol Surface Coverage (1,036 / 1,036 across 11 R packages)**:
  - Completed all 8 surrounding R packages (`gamlss.cens`, `gamlss.tr`, `gamlss.mx`, `gamlss.add`, `gamlss.spatial`, `gamlss.foreach`, `gamlss.prepdata`, `gamlss.ggplots`) with dedicated evidence modules and live-R consistency tests (D2.246–D2.266).
  - Closed the final 11 core `gamlss` symbols with live-R parity tests: `get_rqres`, `rqres_plot`, `fittedPlot`, `bp`, `gamlssNews` (D2.276), weighted local polynomial `gamlss_lo` and exact marginal ML/REML `re`/`gamlss_re` (D2.277), plus `calibration`, `stepGAIC_CH`, and `stepTGDAll_A` (D2.278).
- **M0 / M1 / M2 / M3 Milestones & Admission Gates**:
  - Expanded `production_safe` admission from 1 family (5 symbols) to **22 families (110 symbols)** backed by live-R golden fit, `newdata` prediction, JSON round-trip serialization, and 66-cell RS/CG/MIXED layered parity (D2.227–D2.245).
  - Implemented `gamlss.family` / `checklink` / `make.link.gamlss` / `as.family` / `gen.Family` with live-R fittable `logNO`/`logitNO`, and extended `*.fix` + categorical prediction reconstruction across RS/CG/MIXED/JOINT/LBFGS.
- **Distribution Contract & Opt-in Native Bessel**:
  - Added `omnilss/ondil_contract.py` (22-family parameter/support/link/derivative contract view with per-observation `score`/`hessian` fixes, self-contained with zero external runtime dependency) and opt-in JAX native `omnilss/bessel_native.py` (5.7e-14 accuracy, archived opt-in pending GPU benchmarking).

### Fixed
- Synchronized drifted M1 assessment and layered-parity artifacts, tightened coverage ratchet baseline (`missing`: 173 → 41, all remaining out-of-scope `gamlss2`), and wired M1/M2/M3/M4/P2/layered-parity gates into CI and Release workflows (D2.272).
- Fixed `omnilss-pro/tests/test_client_tls.py` failure under real `grpcio` and enabled live execution of `omnilss-pro` and `omnilss-server` test tiers in CI/Release (D2.273).
- Eliminated internal `_parse_formula` `DeprecationWarning` across 5 core modules, guarded `inf - inf` `RuntimeWarning` in `rs_algorithm.py`, and fixed `JSUo`/`JSUO` case mismatch in `family_capabilities.py` (D2.274).
- Added fail-closed `require_production_family_route` admission, bounded rate-limit client tracking, and 11-test suite to `omnilss-server/`, and fixed doubled `-q` output suppression in `tools/tiered_collection.py --mode run` (D2.275).

## [0.4.0] - 2026-08-12 (Release 1)

### Release 1 能力范围（覆盖范围 = 已验证能力）

**核心分布（E3 晋级，八核心）**：NO / GA / PO / BI / NBI / BE / ZAGA / GU 的
d/p/q/r、属性测试（hypothesis 100 examples/族）、R 一致性（rtol ≤ 1e-3）与
固定 R golden artifact（replay rtol 2e-4）全部通过。

**拟合算法**：RS（NumPy 恒走）、CG（含 CG_IRLS_CROSS 交叉导数实验后端闭环：
eta-scale derivative kernel、observed-information block extraction、IRLS-cross
global line search、CG backend diagnostics 与验证矩阵）、MIXED（真混合路由，
修复 NBI 次优收敛）三算法隔离 R 对照；路由诊断（Route Diagnostic 含
accelerator_available）。

**开发与发布**：离线开发环境构建文档补充一次性 build-time 依赖清单、镜像
构建期依赖校验、发布/架构/R/gRPC 验证命令和维护规则（避免缺 optax/build/
twine/R 包/protobuf tooling 阻塞）。

**数值地基**：x64 全局精度、IRLS working-response 无硬 clip、WLS 动态 ridge、
ZAGA 密度无形状 clip、族感知初始化、safeguard 结构化诊断（eta_clip /
hessian_ridge / hessian_floor / nan_replacement）。

**平滑与预测（pb/ps/cs）**：单一 managed 表示 + lambda origin 生命周期
（user_fixed / df_target / method_selected）、RS 平滑器更新诊断、saved-state
newdata 重建契约（cs 由拒绝改为重建成功）、严格 PredictionSchemaError
（term id + expected/actual schema + state version）、smoother admission
fail-closed（UnsupportedSmootherError）。

**验证体系**：`--require-r` fail-closed 门禁（stage1/stage2 gate suite）、
容差体系重建（无 rtol > 0.01 残留，审计门禁）、golden runner 验证 OmniLSS
本体、真实 R 参考结果。

### Added
- Phase 1 核心数学正确性修复（qTF/pIG/qIG/WEI3/BCCG-BCT-BCPE 截断归一化等）。
- Phase 2 最小可信核心（八核心 E3 晋级、三算法 R 对照、safeguard 诊断、
  RS 路由、ps/cs 基线、stage1 门禁）。
- Phase 3 可信平滑与预测（3.1–3.8 全部完成）。
- Phase 4 验证体系（4.2 容差、4.3 golden 全流程、4.4 伪测试清理、
  4.1/4.5 CI 与发布门禁）。

### Changed
- MIXED 由内联阻尼 IWLS 改为真混合 dispatcher（修复收敛缺陷）。
- 八核心族 R 对照容差收紧至 ≤ 1e-3（原 0.02–0.25）。
- golden runner python 侧由 scipy 改为 OmniLSS 本体。
- ZASICHEL/ZISICHEL/ZABNB/ZIBNB 参数元组去重（zero 参数改用 tau）。

### Notes
- 已知收敛缺陷（CG-ZAGA/CG-GU/Mixed-GU）与 14 个非核心族拟合差异已文档化
  例外清单，列入拟合引擎后续工作。
- ZASICHEL/ZISICHEL dpq NaN 与 fixed-R vendored 源包安装（Windows）为
  预先存在/环境阻塞项。

## [0.3.0] - 2026-05-17

### Added
- R 运行环境安装与验证文档：`docs/development/r-environment-setup.md`（包含 apt/conda 安装路径与 R 侧最小验证命令）。
- 发布与 CG 交叉导数状态已收敛到维护中的验证文档：`docs/validation/cg-algorithm-integrity.md`；一次性收尾清单与开发路径草案不再作为长期文档发布。
- `fitting.py` 拆分辅助模块：`_fitting_utils.py`、`_fitting_init.py`、`_fitting_residuals.py`，用于承载公式工具、初始化与残差/rqres 逻辑。

### Changed
- `fitting.py` 改为委托共享辅助实现，减少内联重复逻辑并收敛架构边界。
- `FamilyDefinition` 文档补充与 `core.distributions.DistributionProtocol` 的迁移关系说明。
- `core/likelihood` 与 `core/losses` 包级文档补充迁移期用途说明。

### Fixed
- CG 算法命名和描述修正：`_apply_method_step` 中的 CG 分支明确标注为“阻尼步”。
- 版本号统一：`omnilss/pyproject.toml` 与 `omnilss/src/omnilss/__init__.py` 对齐为 0.3.0。
- `__init__.py` 中 5 处静默 `except ImportError: pass` 改为 `ImportWarning` 诊断。
- README 性能声明补充测试条件并移除无条件绝对性能表述。

### Validation
- 新增发布检查工具 `omnilss/tools/release_check.py`，用于本地构建、twine 检查与 wheel 安装 smoke check。
- 明确 R-backed consistency tests 是发布与 benchmark claims 的必要门禁；Python-only 模式只能作为 smoke check。

### Known limitations
- 当前开发容器缺少 `build` / `twine` / `optax` / R `gamlss` 相关依赖，因此打包检查、架构 smoke test 与 R 一致性门禁需要在依赖齐全的发布环境中复跑。
- `CG_FULL_HESSIAN` 是默认 correctness backend；`CG_IRLS_CROSS` 保持显式 opt-in 实验 backend，发布声明以验证文档和 benchmark artifact 为准。

### Installation diagnostics
- `omnilss.check_installation()` 安装健康检查函数。

## [0.1.0] - 2024-XX-XX

### Added
- Initial implementation of GAMLSS in JAX
- Support for 47+ distributions
- Basic d/p/q/r functions for common distributions
- Model fitting with `gamlss()` and `gamlss_ml()` functions
- Smooth term support (pb, cs)
- R consistency tests
- Basic performance testing

### Distributions Implemented
**Continuous:**
- NO, GA, LOGNO, WEI, EXP, IG, LO, TF, BE
- GU, RG, IGAMMA, PARETO2
- NO2, LOGNO2, PE, SIMPLEX, EXGAUS
- SHASH, SN1, SN2, GT
- GG, GB2, NET

**Discrete:**
- PO, BI, GEOM, NBI, NBII
- ZIP, ZIP2, ZINBI, ZAP
- BB, BNB, PIG, SICHEL, DPO, DEL, YULE, WARING

**Mixed:**
- BEINF, ZAGA, ZAIG

## Release Notes

### Version 0.1.0 Notes

This is the initial release of Omni GAMLSS, providing:
- Core GAMLSS functionality
- 47+ distribution families
- Automatic differentiation via JAX
- Smooth term support
- R compatibility

**Known Limitations:**
- 32 distributions have only d function (p/q/r return NaN)
- Some discrete distributions need consistency improvements
- Performance optimization ongoing for complex distributions

**Performance:**
- 10-150x faster than R for simple distributions
- 3-10x faster for complex distributions (with JIT optimization)

## Upgrade Guide

### From Development to 0.1.0

No breaking changes. This is the first release.

## Future Plans

### Version 0.2.0 (Planned)
- [ ] Complete p/q/r functions for all distributions
- [ ] Integrate JIT optimization into main fitting code
- [ ] Add more smooth term types
- [ ] Improve documentation and tutorials
- [ ] Add more examples

### Version 0.3.0 (Planned)
- [ ] Add model diagnostics
- [ ] Add prediction functions
- [ ] Add centile plots
- [ ] Add model selection tools

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to this project.

## Links

- [Documentation](docs/README.md)
- [GitHub Repository](https://github.com/dongfangzhizhu/OmniLSS)
- [Issue Tracker](https://github.com/dongfangzhizhu/OmniLSS/issues)
