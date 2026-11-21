# R 一致性验证与 Golden 工件体系

[English version](tutorial-r-consistency.md)

OmniLSS 的数值可靠性建立在**逐族、逐符号、工件可机检的实测证据**之上，而非笼统声明。
符号范围与对齐状态由 `docs/development/r-symbol-coverage.json` 自动生成，并由
`docs/development/symbol-coverage-baseline.json` 棘轮门禁锁定；`gamlss2` 在
`tools/coverage_scope.py` 中明确标记为 OUT OF SCOPE，不计入 1.0 目标。

## 三层验证体系

1. **逐符号覆盖与一致性矩阵（`tools/generate_symbol_coverage.py`）**：
   11 个范围内 R 包全部 1,036 / 1,036（100%）符号完成表面覆盖，且范围内未验证符号清零
   （`evidence_gap = 0`、`placeholder = 0`）。
2. **固定 R Golden 夹具（`omnilss/tests/fixtures/r_golden/regenerated/`）**：
   覆盖全部 22 个生产准入族的 R 侧偏差、系数、`newdata` 预测与收敛元数据，CI 无需安装
   R 即可回归验证。
3. **里程碑与准入审计门禁（`tools/m1_assessment.py`、`m2_assessment.py`、
   `m3_assessment.py`、`m3_admission_evidence.py`、`layered_parity.py`、
   `p2_defect_classification.py`、`m4_scope_inventory.py`）**：
   能力矩阵中的每一条 `validated` 声明（`production_safe`、`rs_fit`、`cg_fit`、
   `prediction`、`sampling`、`smooth_terms`）均在 CI 中按 fail-closed 方式回溯审计。

## 本地运行验证

```bash
# 运行实时 R 一致性测试（需要本机 R + gamlss + jsonlite）
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=omnilss/src \
  .venv/Scripts/python.exe -m pytest -p no:cacheprovider omnilss/tests/test_r_consistency*.py

# 运行无 R 依赖的 golden 回归测试
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=omnilss/src \
  .venv/Scripts/python.exe -m pytest -p no:cacheprovider omnilss/tests/test_r_golden*.py

# 校验全部治理工件与当前源码树无漂移
python tools/generate_symbol_coverage.py --check
python tools/m1_assessment.py --check
python tools/m2_assessment.py --check
python tools/m3_assessment.py --check
python tools/m3_admission_evidence.py --check
python tools/m4_scope_inventory.py --check
```

## 重新生成 Golden 夹具

```bash
Rscript omnilss/tests/fixtures/r_golden/generate_golden.R
```

升级 R 或 `gamlss` 后重新生成夹具时，请在提交前逐项核对数值 diff。

## 下一步

- [gRPC 与 HTTP 托管服务](tutorial-grpc-serving_cn.md)
