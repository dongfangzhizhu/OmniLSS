# OmniLSS 交接文档（2026-11-21，D2.278 基线）

> 用途：新会话入口。先读本文件，再读 `DEVELOPMENT_PLAN.md` 第二十三节与
> `docs/development/full-project-audit-2026-11-21.md`。历史 2026-10-31 交接细节已归档于 git 历史与 `.workbuddy/memory/`。

## 0. 环境与硬性约束

- 仓库：`d:\AI\my\githubprojs\OmniLSS`，分支 `main`。
- **Windows 运行环境**：`.venv/`（Python 3.10.16 / JAX 0.6.2 CPU x64 / NumPy 2.2.6 / SciPy 1.15.3），使用 **`uv`** 管理。
- **WSL Debian GPU 环境**：`debianjaxgpu/`，使用 **`uv`** 管理（受保护，切勿误删）。
- **R 环境**：R 4.6.0 + `gamlss` 5.5.0 + `gamlss.dist` 6.1.1 + `jsonlite`；vendored 源码在 `R/`（12 个 R 包）。
- **参考项目红线**：`gamlss-python/` 与 `ondil/` **仅作设计与算法参考，严禁引入其代码或作为依赖**。

## 1. 当前状态（D2.278 实测）

- **范围内 R 符号覆盖率**：**1,036 / 1,036（100.00%）**——11 个范围内 R 包（`gamlss`、`gamlss.dist`、`gamlss.cens`、`gamlss.tr`、`gamlss.mx`、`gamlss.add`、`gamlss.spatial`、`gamlss.foreach`、`gamlss.prepdata`、`gamlss.ggplots`、`gamlss.data`）全部 0 缺失；剩余 41 个 `surface_missing` 全部属于明确 out-of-scope 的 `gamlss2`。
- **里程碑状态**：
  - **M0 / M1 / M2 / M3**：全部达成，对应门禁脚本全部 **PASS** 且已接入 CI / Release。
  - **Phase 1（Track A：A1–A4）**：全部完成（漂移门禁修复、覆盖棘轮基线收紧至 41、`omnilss-pro` TLS 测试修复、`omnilss-server` 能力门禁与有界限流加固、内部 `_parse_formula` / `inf-inf` 告警清零、`JSUO` 大小写统一）。
  - **Phase 2（Track B1：B1a/B1b/B1c）**：全部完成（补齐最后 11 个核心符号，全部配活 R 一致性测试）。
- **运行时能力矩阵（146 族）**：`implemented` 22（`production_safe` 22 族 / 110 符号）、`experimental` 104、`compat_alias` 20、`placeholder` 0；`r_consistency=validated` 40 族。
- **测试规模**：4 层共 624+ 文件 / 5,840+ 用例；`omnilss-server`(11) + `omnilss-pro`(25) 经 `tools/tiered_collection.py --mode run` 实跑 0 失败。

## 2. 下一阶段待办（按优先级，详见 `DEVELOPMENT_PLAN.md` §23）

1. **B2（扩大证据与能力准入）**：压降范围内 `unverified` 符号，打通 `cg_fit`/`smooth_terms`/`sampling` 能力矩阵准入，扩展第二批 `production_safe` 高频族与 `cens`/`tr` golden 语料。
2. **B3b & B4（M4 / 1.0.0 正式收官）**：统一官方教程到 `docs/tutorials/` 并接入 `mkdocs.yml`，建立 `tools/m4_assessment.py` 门禁，升级版本号并跑 8 分片全量回归与 `release_check.py`。
3. **Track C（P5 自研在线与多元层）**：纯 JAX/NumPy 实现 `omnilss/online/`（`OnlineGAMLSS` + 7 项独立流式 golden）与多元 `MVN`/`MVT`（Cholesky/LowRank/ModifiedCholesky）。
4. **Track D（P6 GPU 与内核加速）**：在 WSL `debianjaxgpu` 上实测 GPU/CPU parity、原生 Bessel 与 `RS_JAX` 性能，优化 `SICHEL` 动态步长循环。

## 3. 常用验证命令

```powershell
$py = ".venv\Scripts\python.exe"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:JAX_PLATFORMS = "cpu"
$env:JAX_ENABLE_X64 = "true"
$env:PYTHONPATH = "omnilss/src"

& $py tools/generate_symbol_coverage.py --check
& $py tools/coverage_diff_report.py
& $py tools/milestone_manifest.py --check
& $py tools/m1_assessment.py --check
& $py tools/m2_assessment.py --check
& $py tools/m3_assessment.py --check
& $py tools/m4_scope_inventory.py --check
& $py tools/contract_freeze.py --check
& $py tools/tiered_collection.py --tier omnilss-server --tier omnilss-pro --mode run
```

## 4. 关键纪律（防踩坑）

1. **改任何 `omnilss/src` 或 `omnilss/tests` 后必须同步工件**：`tools/generate_symbol_coverage.py` + `tools/coverage_diff_report.py` + `tools/milestone_manifest.py --write`，否则 provenance 门禁失败。
2. **按族名分支用 `resolve(key).name` 或统一 `.upper()`**：注册表键为全大写（如 `JSUO`），R 族名含小写 `o`（如 `JSUo`）。
3. **`tmp/` 保留 5 个承重 R 参照文件**（`bi_dbi_fixture_r.json`、`bi_dbi_ref.R`、`d223_c_ref.csv`、`d223_full_ref.csv`、`d223_ref.R`），勿删。
4. **全量测试必须 8 分片运行**（`tools/pytest_shards.py --shards 8`），性能基准单进程单独复核。
