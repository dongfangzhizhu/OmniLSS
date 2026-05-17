# Changelog

All notable changes to Omni GAMLSS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 完成 CG full-Hessian correctness backend 的交叉导数实现闭环，并将 `CG_IRLS_CROSS` 推进为显式可选实验后端：新增 eta-scale derivative kernel、observed-information block extraction helpers、CG backend diagnostics、IRLS-cross global line search 与回归测试、验证矩阵与 smoke/reference artifact，确保 `method="CG"` 可审计地使用 cross-derivative 信息而不是静默退化为 RS。
- 完成 CG full-Hessian correctness backend 的交叉导数实现闭环，并将 `CG_IRLS_CROSS` 推进为显式可选实验后端：新增 eta-scale derivative kernel、observed-information block extraction helpers、CG backend diagnostics、IRLS-cross global line search 与回归测试、验证矩阵与 smoke/reference artifact，确保 `method="CG"` 可审计地使用 cross-derivative 信息而不是静默退化为 RS。

### Changed
- 离线开发环境构建文档与 devcontainer bootstrap 补充一次性 build-time 依赖清单、镜像构建期依赖校验、发布/架构/R/gRPC 验证命令和维护规则，避免后续离线会话因缺少 `optax`、`build`、`twine`、R 包或 protobuf tooling 阻塞开发。
- 离线开发环境构建文档与 devcontainer bootstrap 补充一次性 build-time 依赖清单、镜像构建期依赖校验、发布/架构/R/gRPC 验证命令和维护规则，避免后续离线会话因缺少 `optax`、`build`、`twine`、R 包或 protobuf tooling 阻塞开发。


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
