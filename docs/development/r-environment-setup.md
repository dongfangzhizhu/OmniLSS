# R 运行环境安装与测试指南（OmniLSS）

更新时间：2026-05-17

## 目标
为 `benchmarks/**/*.R` 以及 `run_r_tests.ps1` 等脚本提供可用的 R 运行时（`R` / `Rscript`）。

## Ubuntu/Debian（首选）
```bash
sudo apt-get update
sudo apt-get install -y r-base
R --version
Rscript --version
```

## Conda / Mamba（备选）
当系统包管理器不可用时可使用：
```bash
mamba install -y -c conda-forge r-base
# 或
conda install -y -c conda-forge r-base
R --version
```

## 最小验证
在仓库根目录执行：
```bash
Rscript benchmarks/diagnostic/test_r_likelihood.R
```

如需运行更完整的 R 侧对比测试，可按需执行：
```bash
Rscript benchmarks/diagnostic/check_r_convergence.R
Rscript benchmarks/diagnostic/load_r_data_and_test.R
```

## 常见问题
1. **`apt-get` 403 / 仓库不可达**：通常是网络代理或镜像访问受限。可在可联网 CI 环境安装，或切换到 Conda 方案。
2. **`Rscript: command not found`**：说明 `r-base` 未正确安装，检查 PATH 或重装。
3. **缺少 R 包**：根据错误信息执行 `install.packages("<pkg>")` 后重试。
