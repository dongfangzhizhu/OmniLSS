# OmniLSS P0-P4 可用性冲刺记录（2026-05-17）

本文档记录本阶段将 OmniLSS 从工程实验推进到可安装、可运行、可验证、可引用状态的任务执行情况。

## 目标

- 修复阻塞性运行时问题，确保核心 `import omnilss` 与基础 `gamlss()` 可用。
- 让 gRPC 边界具备可生成 stubs、可启动服务、可 smoke test 的基础设施。
- 补齐发布、双进程边界验证、benchmark 诚实化、R/GPU 条件验证、JOSS 草稿等可开发资产。

## 完成项

### P0 阻塞修复

- `omnilss.__init__` 的 optional scoring / sklearn 顶层导出改为 lazy resolution，避免 optional extras 影响核心 import。
- gRPC `Sample` 改为基于拟合分布参数调用 family `r()`，并保留 `q()` + uniform fallback。
- `algorithms.__init__` 明确区分历史 L-BFGS 路径 `joint_lbfgs_fit` 与真实 Cole-Green `cole_green_fit`。
- JSON 序列化持久化 `formulas`，HTTP `/predict` 对缺失变量与预测失败返回明确错误。

### P1 gRPC 基础设施

- 扩展 `fit.proto` 支持完整 fit 参数和拟合摘要返回。
- 提交 `fit/predict/sample` 的 generated stubs，打包时包含 proto 和 generated Python 文件。
- `tools/generate_grpc_stubs.py` 会修正 grpcio-tools 生成文件中的 sibling import，使其可作为 package module 导入。
- `omnilss-server/grpc_main.py` 提供独立 gRPC 启动入口。
- Docker Compose 已包含 Core HTTP、Core gRPC 和 Pro demo 三服务拓扑。

### P2 发布准备

- `pyproject.toml` 包含 `api/grpc/proto/*.proto`、`api/grpc/generated/*.py` 与 `py.typed`。
- Release workflow 在构建前安装 `grpcio-tools` 并重新生成 gRPC stubs。
- PyPI 发布 job 使用 `pypi` environment 与 OIDC trusted publishing。

### P3 双层架构验证

- 新增 `omnilss-pro/` 原型包，不依赖 `omnilss`，只依赖 `grpcio`、`protobuf` 和 `numpy`。
- Pro client 使用独立复制的 proto/generated stubs 通过 gRPC 调用 Core 的 fit/predict/sample。
- Pro integration test 在 Core 服务未运行时 skip，在 Docker Compose 中可作为 demo service 运行。

### P4 质量与可信度

- `benchmarks/comprehensive_performance_test.py` 将 warm-up 提升为 3 次，warm time 使用多次运行中位数，并保留 cold-start 单独报告。
- 新增 `honest_benchmark()` 小型 helper，供 README/performance 表格刷新使用。
- 新增 `benchmarks/r_reference_results.json`、`benchmarks/generate_r_reference_results.R` 与 `benchmarks/validate_against_r.py`；当前 R reference 标记为待外部 R 环境填充。
- 新增 GPU 条件测试 `omnilss/tests/test_device_gpu.py`，无 GPU 时 skip。
- 新增 JOSS 草稿 `paper/paper.md` 与 `paper/paper.bib`。
- README 性能说明改为方法论和复现命令，不再保留未经当前环境重新生成的静态 speedup 表。

## 环境受限 / 跳过项

- 当前环境无法从 PyPI 安装 `grpcio-tools`（网络 tunnel 403），因此本轮无法用 `python tools/generate_grpc_stubs.py` 端到端再生成 stubs；已用系统 `protoc` 生成 pb2，并保留 workflow 中的正式生成步骤。
- 当前环境有 `grpcio` 但无 `google.protobuf`，gRPC pytest 用例会按依赖不完整路径 skip。
- 当前环境无 Rscript / R `gamlss`，R reference JSON 只能提交待填充占位，不能声称 R 一致性已验证。
- 当前环境无 GPU backend，GPU 条件测试只能验证 skip 行为。
- 当前环境无 `build` 包，不能本地执行 `python -m build`；Release workflow 与 Environment Validation workflow 已保留正式构建路径。

## 继续推进（依赖闭环）

- P1/P2：将 `protobuf>=3.20.0` 显式加入 Core `grpc` extra、devcontainer 依赖、server requirements 与 Pro prototype 依赖；generated `*_pb2.py` 在运行时直接依赖 `google.protobuf`，不能只依赖 `grpcio-tools` 的传递依赖。
- P0/P1：`omnilss.check_installation()` 的 `grpc` 项现在检查 generated gRPC stub runtime，而不是只检查 server module 是否可导入，从而能暴露 `protobuf` 缺失等真实 gRPC 运行时问题。
- P1：Core gRPC server 在 runtime/stub 依赖不完整时给出 `pip install 'omnilss[grpc]'` 与 generated stubs 路径提示，避免把缺少 protobuf 误诊为单纯未生成 stubs。
- P1/P4：新增 gRPC runtime diagnostics 单元测试，不依赖可启动 gRPC server 即可验证缺失依赖识别与错误提示。
- P4：新增 R reference 生成脚本，并在 devcontainer/Codex 配置中加入 `OMNILSS_REFRESH_R_REFERENCE=1` 开关，供下次具备 R/gamlss/jsonlite 的构建自动刷新 R fixture。
- P4：新增 `.devcontainer/devcontainer.gpu.json` 可选 GPU devcontainer 配置，使用 `--gpus all`、`JAX_PLATFORMS=cuda` 与 `jax[cuda12]`，供下次具备 NVIDIA runtime/网络的环境执行 GPU 条件测试。
- P2/P4：新增 `.github/workflows/environment-validation.yml` 手动/定时验证工作流，覆盖 gRPC stub/package build、R reference 生成与可选 self-hosted GPU smoke；GPU job 默认不运行，需要 workflow_dispatch 时勾选 `enable_gpu`，避免无 self-hosted GPU runner 时阻塞常规验证。

## 人工检查前状态

- P0-P4 代码与文档资产已闭环到当前可自动推进范围。
- 当前执行环境仍因 PyPI tunnel 403 无法安装 `protobuf` / `grpcio-tools` / `build` 等缺失包，因此 gRPC smoke 在本地以缺少 `google.protobuf` 的 skip 形式验证依赖诊断；相关依赖已写入 devcontainer/packaging/release workflow/环境验证 workflow 配置，具备网络的维护者环境应重新运行完整检查建议。
- R reference 与 GPU 后端仍属于外部环境验证项；已补充 R fixture 生成开关与 GPU devcontainer 配置，但不能在本容器中声称已完成数值一致性或 GPU 性能验证。

## 检查建议

维护者在具备网络、R、grpcio 和可选 GPU 的环境中建议执行：

```bash
cd omnilss && python tools/generate_grpc_stubs.py
cd omnilss && python -m pytest tests/test_grpc_server.py -v
python benchmarks/comprehensive_performance_test.py --quick --require-r --n-repeats 10
OMNILSS_REFRESH_R_REFERENCE=1 bash .codex/setup.sh
python benchmarks/validate_against_r.py
python -m build
# or run GitHub Actions: Environment Validation
```

## 后续环境初始化优化（追加）

- `.devcontainer/` 现在在镜像构建阶段安装 Python/R/protobuf/system libraries，并预填充 `/opt/omnilss-wheelhouse` 以支持后续离线开发。
- `.codex/setup.sh` 现在支持 online refresh + offline fallback，安装 Core/Pro editable packages，生成 gRPC stubs，并输出 `omnilss.check_installation()` 健康检查。
