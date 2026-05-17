# Changelog

All notable changes to Omni GAMLSS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- JIT optimization framework for performance improvement (`fitting_jit.py`)
- Comprehensive DPQR function tests (`test_dpqr_comprehensive.py`)
- Performance testing framework in `performance/` directory
- Documentation structure in `docs/` directory

### Fixed
- GU (Gumbel) distribution d function formula (was using RG formula)
- RG (Reverse Gumbel) distribution d function formula and parameter order
- PowerShell test script module loading warning
- `test_dpqr_comprehensive.py` AttributeError during test collection

### Changed
- Reorganized project documentation into `docs/` directory
- Improved test script organization
- Enhanced performance testing capabilities

### Performance
- ZAGA IWLS step: 275x faster with JIT compilation
- BE IWLS step: 202x faster with JIT compilation
- Overall expected improvement: 5-10x for ZAGA, 3-8x for BE

## [0.3.0] - 2026-05-17

### Fixed
- CG 算法命名和描述修正：`_apply_method_step` 中的 CG 分支明确标注为“阻尼步”。
- 版本号统一：`omnilss/pyproject.toml` 与 `omnilss/src/omnilss/__init__.py` 对齐为 0.3.0。
- `__init__.py` 中 5 处静默 `except ImportError: pass` 改为 `ImportWarning` 诊断。
- README 性能声明补充测试条件并移除无条件绝对性能表述。

### Added
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
