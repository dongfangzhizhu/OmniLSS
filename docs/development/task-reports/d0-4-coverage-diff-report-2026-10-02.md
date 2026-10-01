# D0.4 registry/R 源码变化自动差异报告（2026-10-02）

## 目标

`docs/development/deliverable-plan-2026-09-26.md` D0.4：在 D0.1 逐符号矩阵之上输出
**missing / extra / alias / placeholder / evidence 缺失**差异报告，并在 registry 或 R 源码
变化时阻断回归。

## 实现

- 工具 `tools/coverage_diff_report.py`：直接复用 D0.1 的 `build_coverage()`（不重复解析），
  把逐符号矩阵分到五类，并与入库的 ratchet 基线比对。

| 类别 | 判定依据（矩阵层） | 是否阻断 |
|---|---|---|
| `missing` | `surface == missing`（R 有导出，Python 无 surface） | 是 |
| `placeholder` | `kernel == placeholder` | 是 |
| `compat_alias` | `kernel == compat_alias` | 是 |
| `evidence_gap` | `kernel == implemented` 且 `parity == unverified` | 是 |
| `extra` | `omnilss.__all__` 中无 R 对应项的公开导出 | 否（仅报告） |

- **ratchet 语义**（`docs/development/symbol-coverage-baseline.json`，schema_version 1）：
  - 任一阻断类别**新增**符号 → 回归，`exit 1` 并逐条打印符号；
  - 符号**离开**阻断类别 → 记为 improvement，只报告不失败；
  - `ceiling_ok` 额外要求当前计数不超过基线计数；
  - `--update-baseline` 是承认一次回归的唯一途径（可评审的提交）；
  - `--allow-regressions` 仅供有意的、经评审的步骤使用，报告仍打印 `FAIL-CLOSED`。
- **fail-closed**：baseline 缺失 / 不可读 / schema 不符 / 缺类别 → 报告 `failure` 并退出 1，
  绝不退化成 skip（对齐 D7“无隐性 skip”）。
- **可移植性**：报告里的 `baseline` 字段落库为**仓库相对路径**（`_display_path`），
  避免提交物带上本机绝对路径而在 CI 上必然失配。有回归测试断言已入库报告不含盘符
  或 `/home/runner` 之类的机器路径。
- 流水线接入：`tools/generate_coverage_artifacts.py` 在写完 D0.1 矩阵后，用**同一个内存快照**
  生成 `symbol-coverage-diff.json`；若 ratchet 失败，流水线打印报告并返回 1，
  保证“矩阵与差异报告不会互相打脸”。
- CI：`.github/workflows/ci.yml` 的 `collection-gate` job 新增
  `python tools/coverage_diff_report.py --json --out coverage-diff.json`，
  并上传 `coverage-difference-report` artifact（`if: always()`）。
- 顺带修好：`omnilss/pyproject.toml` 的 `norecursedirs` 增补 `.hypothesis`
  （hypothesis 示例数据库目录，避免收集期被扫描）。

## 证据与结果

本机实测（Python 3.10.19 / JAX 0.6.2 / R 4.6.0，仓库根执行）：

```
coverage difference report (D0.4)
baseline: docs/development/symbol-coverage-baseline.json
R exports: 1080

| category | current | baseline | new | resolved |
|---|---|---|---|---|
| missing | 402 | 402 | 0 | 0 |
| placeholder | 75 | 75 | 0 | 0 |
| compat_alias | 11 | 11 | 0 | 0 |
| evidence_gap | 36 | 36 | 0 | 0 |
| extra (informational) | 300 | 300 | 0 | 0 |

no symbol-level change relative to the ratchet baseline
result: OK
```

- 基线即本次首次生成，**尚未经过任何“下降”**，因此当前 ratchet 只是**冻结现状**：
  402 missing / 75 placeholder / 11 compat_alias / 36 evidence_gap 是 1.0 之前必须逐项
  削减的债务清单，不是已达成的覆盖率。
- 数字口径说明：这里的 `placeholder` 75 是**按 R 符号计数**（一个 family 通常对应
  `d/p/q/r` 等多个导出符号），与交付计划第 2 节“registry family 146（18 placeholder）”
  按**族**计数的口径不同，两者不可直接互相比大小。

## 验证

- `cd omnilss; python -m pytest tests/test_coverage_diff_report.py -q` → **17 项通过**
  （含 1 项 `slow`：重跑真实矩阵，断言入库 baseline 与 diff 快照均新鲜）。
- `cd omnilss; python -m pytest tests/test_symbol_coverage_tool.py -q` → 14 项通过。
- `python tools/golden_manifest_gate.py --check` → `validation: ok (9 cases)`，`result: OK`。
- `python tools/generate_coverage_artifacts.py` → 6 个生成物全部重写，exit 0
  （`r-export-coverage.{json,md}`、`r-symbol-coverage.{json,md}`、
  `symbol-coverage-diff.json`；占位队列内容与已入库的 2026-09-26 版仅日期不同，
  按 D8“同主题只保留最新版”不产生新的重复副本）。
- `.github/workflows/ci.yml` 新增步骤位于既有 `collection-gate` job 内，未新增 job。

## 已知边界

- **不检测数值回归**：本门禁只管符号层与证据层的增减，d/p/q/r 数值是否仍与 R 一致
  由 D1 的 parity 测试负责。
- **R 源码缺失时口径收窄**：`R/` 是 vendored 且 gitignored，CI 无 R 源码，矩阵在 CI 上
  会退化为不含 R 定义证据的形态；本报告在 CI 上仍然可信（比较的是同一环境下的矩阵
  与基线），但“R 侧新增导出”这类回归只有在有 `R/` 的本地/开发环境中才会暴露。
  D0.1 的新鲜度检查已在 CI 显式说明这一点（而非静默跳过）。
- `extra` 300 项是 Python 独有导出，只报告不阻断；其中可能有命名不规范需要回收的，
  不在本任务范围内。

## 后续

- D1-A：ST/SEP/SST 的 R golden 与独立 kernel —— 完成后 `missing`/`placeholder`/
  `evidence_gap` 三张清单都应下降，届时用 `--update-baseline` 把新水位固化。
