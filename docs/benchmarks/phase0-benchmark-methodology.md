# Phase 0 Benchmark Methodology

[中文版本](phase0-benchmark-methodology_cn.md)

## Fixed scenarios
- small / medium / pathological datasets
- RS baseline + optional joint optimizer comparison

## Metrics
- loglikelihood/deviance
- convergence status and iterations
- runtime
- peak memory
- JIT compile time (when applicable)
- gradient stability diagnostics

## Repro command
```bash
PYTHONPATH=omnilss/src python benchmarks/phase0_benchmark_harness.py --n 1000 10000 --method RS --family NO
```

## Output artifact
Results are persisted as JSON under `docs/benchmarks/`.
