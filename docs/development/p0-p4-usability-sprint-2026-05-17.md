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

- 新增 `omnilss-pro/` 原型包，不依赖 `omnilss`，只依赖 `grpcio` 和 `numpy`。
- Pro client 使用独立复制的 proto/generated stubs 通过 gRPC 调用 Core 的 fit/predict/sample。
- Pro integration test 在 Core 服务未运行时 skip，在 Docker Compose 中可作为 demo service 运行。

### P4 质量与可信度

- `benchmarks/comprehensive_performance_test.py` 将 warm-up 提升为 3 次，warm time 使用多次运行中位数，并保留 cold-start 单独报告。
- 新增 `honest_benchmark()` 小型 helper，供 README/performance 表格刷新使用。
- 新增 `benchmarks/r_reference_results.json` 与 `benchmarks/validate_against_r.py`；当前 R reference 标记为待外部 R 环境填充。
- 新增 GPU 条件测试 `omnilss/tests/test_device_gpu.py`，无 GPU 时 skip。
- 新增 JOSS 草稿 `paper/paper.md` 与 `paper/paper.bib`。
- README 性能说明改为方法论和复现命令，不再保留未经当前环境重新生成的静态 speedup 表。

## 环境受限 / 跳过项

- 当前环境无法从 PyPI 安装 `grpcio-tools`（网络 tunnel 403），因此本轮无法用 `python tools/generate_grpc_stubs.py` 端到端再生成 stubs；已用系统 `protoc` 生成 pb2，并保留 workflow 中的正式生成步骤。
- 当前环境无 `grpcio`，gRPC pytest 用例会按预期 skip。
- 当前环境无 Rscript / R `gamlss`，R reference JSON 只能提交待填充占位，不能声称 R 一致性已验证。
- 当前环境无 GPU backend，GPU 条件测试只能验证 skip 行为。
- 当前环境无 `build` 包，不能本地执行 `python -m build`；Release workflow 已保留正式构建路径。

## 检查建议

维护者在具备网络、R、grpcio 和可选 GPU 的环境中建议执行：

```bash
cd omnilss && python tools/generate_grpc_stubs.py
cd omnilss && python -m pytest tests/test_grpc_server.py -v
python benchmarks/comprehensive_performance_test.py --quick --require-r --n-repeats 10
python benchmarks/validate_against_r.py
python -m build
```

## 后续环境初始化优化（追加）

- `.devcontainer/` 现在在镜像构建阶段安装 Python/R/protobuf/system libraries，并预填充 `/opt/omnilss-wheelhouse` 以支持后续离线开发。
- `.codex/setup.sh` 现在支持 online refresh + offline fallback，安装 Core/Pro editable packages，生成 gRPC stubs，并输出 `omnilss.check_installation()` 健康检查。
