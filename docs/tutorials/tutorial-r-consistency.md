# R 一致性验证与 golden artifact

OmniLSS 的可靠性建立在**逐族、逐符号 validated 的 R 一致性证据**之上（Phase 1-6），
而非「全库与 R `gamlss` 数值等价」的 blanket 声明。任何数值对齐宣称都只适用于
已由 R-backed 测试 / golden artifact / 能力矩阵覆盖的范围；当前范围与证据层级见
`docs/development/r-symbol-coverage.json`（`stats.per_layer`）与
`docs/development/symbol-coverage-diff.json`。`gamlss2` 为明确 OUT OF SCOPE
（`tools/coverage_scope.py`），不属本承诺，也不计入 1.0 交付目标。

## 验证体系

1. **R 一致性测试**：对固定 seed 数据，OmniLSS 拟合结果与 R `gamlss`
   结果在 `rtol ≤ 1e-3` 内一致（系数、偏差）。
2. **固定 R golden artifact**：R 侧预生成的 fixture（`tests/fixtures/
   r_golden/`），CI 无需 R 即可回归——任何数值回归立即暴露。
3. **能力矩阵门禁**：未实现/占位族被硬拒绝，绝不静默给出错误数值。

## 本机运行

```bash
# R 一致性测试（需要本机 R + gamlss）
cd omnilss
PYTHONPATH=src python -m pytest tests/test_r_consistency_*.py -q

# golden 回归（无需 R）
PYTHONPATH=src python -m pytest tests/test_r_golden*.py -q
```

## 重新生成 golden fixture（升级 R/gamlss 后）

```bash
Rscript tests/fixtures/r_golden/generate_golden.R
```

生成物写入 `tests/fixtures/r_golden/`（含 manifest），数值由 R 侧固定，
提交前请人工核对 diff。

## 八核心族验证状态

| 族 | 验证 | 例外 |
|---|---|---|
| NO/GA/PO/BI/NBI | ✅ 全通过 | — |
| BE/GU | ⚠️ 部分 | GU 容差边缘（Phase 2.2 收敛处理） |
| ZAGA | ✅ 拟合一致 | — |
| 其余 ~20 族 | 🔄 进行中 | 见 docs/development 审计 |

## 下一步

- [gRPC 托管服务](tutorial-grpc-serving.md)
