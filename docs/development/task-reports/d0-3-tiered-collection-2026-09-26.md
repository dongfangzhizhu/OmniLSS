# D0.3 四组分层 collection 门禁固化到 CI（2026-09-26）

## 目标

`docs/development/deliverable-plan-2026-09-26.md` D0.3：把分层 collection 命令固化到 CI，
各组 collection 零错误，并统计 collected/error/skipped。

## 分组口径（按用户指示调整）

计划原表写“omnilss / ondil / gamlss-python 分别收集”，清单写“四套”。用户明确：
**ondil 与 gamlss-python 是两个参考项目，不在本项目内使用，可在 README 做对比**。
因此门禁改为覆盖**本项目自有的四个测试面**：

| Tier | 目录 | 范围 | 本机 collected |
|---|---|---|---|
| `omnilss` | `omnilss/tests` | 主套件（`omnilss/pyproject.toml` 配置） | 3384 |
| `omnilss-server` | `omnilss-server` | L3 FastAPI 服务层 | 1 |
| `omnilss-pro` | `omnilss-pro` | 商业扩展原型（src 布局） | 25 |
| `benchmarks` | `benchmarks` | 基准/诊断脚本 | 10 |

## 实现

- 工具 `tools/tiered_collection.py`：
  - 逐组以子进程运行 `python -m pytest ...`，`--mode collect`（默认，零错误门禁）或 `--mode run`；
  - 汇总 `collected / errors / skipped / failed`，退出码 0 仅当每组都通过；
  - fail-closed 规则：存在 error、存在 failed、收集为 0 项或 pytest 退出码 5 → 失败；
  - `--strict-skips`：把收集期 skip（模块级 `importorskip`/`skip`）判为失败（对齐 D7“无隐性 skip”）；
  - `--json` 机器可读报告，`--tier`/`--timeout` 可选；`--list` 列出分组与“参考项目永不为组”的声明；
  - 探针可注入（`runner=`）以便测试无需真实子进程。
- 解析细节：pytest 配置叠加 `-q` 会抑制 “N tests collected” 汇总行，工具优先解析汇总行、
  回退解析逐文件 `path: count` 行；skip 同时支持汇总 `N skipped` 与 `SKIPPED [N]` 两种形态。
- 修复：`omnilss-pro/pyproject.toml` 补 `pythonpath = ["src"]`。此前 src 布局使
  `tests/test_automl.py` 在收集期 `ModuleNotFoundError: No module named 'omnilss_pro'`
  （1 error，阻断整组），现整组 25 项干净收集。
- CI：`.github/workflows/ci.yml` 新增 job `collection-gate`（在 `tests` 与 `lint` 之间）：
  安装 `.[server,dev]` → `python tools/tiered_collection.py --mode collect --strict-skips --json | tee tiered-collection.json`
  → 上传 artifact → `tools/golden_manifest_gate.py --check`；
  D0.1 产物新鲜度检查在 `R/` 存在时执行（`R/` 是 vendored 且 gitignored，CI 无 R 源码），
  并给出一条显式说明步骤而不是静默跳过。
- README：`## Testing` 增补四组门禁用法与分组表；`## Comparison with Similar Tools` 增补
  “Vendored reference projects (compared, not gated)”小节，说明 `ondil/`（0.4.8）与
  `gamlss-python/`（0.2.0）仅作对比、不入门禁，且有回归测试保证它们永远不会变成组。

## 证据与结果

- `python tools/tiered_collection.py --mode collect --strict-skips`（221s）：
  `omnilss 3384 / omnilss-server 1 / omnilss-pro 25 / benchmarks 10`，合计 **3420 项，
  0 error / 0 skip**，`result: OK`，exit 0。
- 修复前实测：`omnilss-pro` 1 error（`omnilss_pro` 导入失败）、`benchmarks` 5 error
  （缺 `scripts/diagnostic/*.csv` 与 `rbus` 桥）、`omnilss-server` 0 collected（汇总行被抑制而误判）。
- 5 个 `benchmarks/diagnostic/` 独立脚本以 `--ignore` 显式排除，清单固化在
  `BENCHMARK_KNOWN_EXCLUSIONS`，并有测试断言这些路径真实存在（防陈旧排除），新增破坏仍会失败。
- 无测试孤岛：回归测试断言四个组根目录覆盖本项目 400+ 个 `test_*.py`，仓库其它顶层目录
  （除既有忽略目录与参考项目）不得再有 `test_*.py`。

## 验证

- `cd omnilss; python -m pytest tests/test_tiered_collection.py -m 'not slow' -q` → 21 项通过（10.6s）。
- `cd omnilss; python -m pytest tests/test_tiered_collection.py -m slow -q` → 端到端真实四组收集断言通过。
- `python tools/tiered_collection.py --list` → 列出四组 + `reference projects (never tiers): ondil, gamlss-python`。
- `.github/workflows/ci.yml` YAML 解析通过，jobs = tests / collection-gate / lint / benchmark-smoke / tls-smoke。

## 已知边界

- `--mode run` 全库执行仍超出单次时限，本任务只固化“收集零错误 + 可选的严格 skip”，
  运行选定组由 `--tier ... --mode run` 按需使用（CI 继续按 Group A/B/C 跑主套件）。
- 收集期 skip 默认只报告不失败（D0.3 验收是“零错误”）；CI 以 `--strict-skips` 收紧，
  本地缺少可选依赖时会如实报出 skip 数量而不是静默通过。
- `omnilss-server` 只有 1 个测试文件（L3 覆盖主要在 `omnilss/tests/test_grpc_server.py` 等），
  组内数量少属现状，不是门禁缺陷。

## 后续

- D0.4：在 D0.1 符号矩阵之上输出 missing/extra/alias/placeholder 差异报告并阻断回归。