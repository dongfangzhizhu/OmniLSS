# R 运行环境安装与测试指南（OmniLSS）

[中文版本](r-environment-setup_cn.md)

更新时间：2026-05-17

## 目标

为 `benchmarks/**/*.R`、R-backed consistency tests、`run_r_tests.ps1`、benchmark validation gate 以及发布前 R 等价性检查提供可用的 R 运行时（`R` / `Rscript`）和必需 R 包。

## 离线开发环境要求

用于离线开发的 devcontainer 必须在镜像构建阶段一次性安装 R 和所有必需 R 包。不要等到离线开发会话中再执行 `apt-get install r-base` 或 `install.packages()`，否则会重现“缺少 R / 缺少 gamlss 包导致发布验证被阻塞”的问题。

当前离线镜像使用 `.devcontainer/Dockerfile` 中的 `rocker/r-ver:4.4.1` 基础镜像，并在 build 阶段安装：

```r
install.packages(c(
  "renv",
  "gamlss",
  "gamlss.dist",
  "jsonlite",
  "languageserver"
))
```

完整离线开发环境清单见 `docs/development/offline-dev-bootstrap-2026-05-17.md`。若未来 R 侧 benchmark 或 validation 新增包依赖，必须同步更新该文档和 `.devcontainer/Dockerfile`。

## Ubuntu/Debian（手动备选）

优先使用 devcontainer；仅在本地手动调试且系统包管理器可用时使用以下方案：

```bash
sudo apt-get update
sudo apt-get install -y r-base
R --version
Rscript --version
```

安装必需 R 包：

```bash
Rscript -e "options(repos=c(CRAN='https://cloud.r-project.org')); install.packages(c('renv','gamlss','gamlss.dist','jsonlite','languageserver'))"
```

## Conda / Mamba（手动备选）

当系统包管理器不可用时可使用：

```bash
mamba install -y -c conda-forge r-base
# 或
conda install -y -c conda-forge r-base
R --version
```

随后仍需安装 R 包：

```bash
Rscript -e "options(repos=c(CRAN='https://cloud.r-project.org')); install.packages(c('renv','gamlss','gamlss.dist','jsonlite','languageserver'))"
```

## 最小验证

在仓库根目录执行：

```bash
Rscript -e "missing <- setdiff(c('renv','gamlss','gamlss.dist','jsonlite','languageserver'), rownames(installed.packages())); if (length(missing)) stop(paste('missing R packages:', paste(missing, collapse=', '))); cat('R package inventory: ok\n')"
```

```bash
Rscript benchmarks/diagnostic/test_r_likelihood.R
```

如需运行更完整的 R 侧对比测试，可按需执行：

```bash
Rscript benchmarks/diagnostic/check_r_convergence.R
Rscript benchmarks/diagnostic/load_r_data_and_test.R
```

发布前的完整门禁仍应使用：

```bash
python benchmarks/run_local_validation.py --quick
```

## 常见问题

1. **`apt-get` 403 / 仓库不可达**：通常是网络代理或镜像访问受限。不要在离线会话中继续补装；应在可联网环境重新构建 devcontainer。
2. **`Rscript: command not found`**：说明当前环境不是完整离线开发镜像，或 R 未正确安装。优先重新构建 devcontainer。
3. **缺少 `gamlss` / `gamlss.dist` / `jsonlite`**：说明镜像 build 阶段的 R 包安装未成功；更新 `.devcontainer/Dockerfile` 或 CRAN 镜像后重新构建。
4. **R-backed tests 被 skip**：这只能作为 Python-only smoke 情况处理，不能用于发布 R 等价性或 benchmark claims。
