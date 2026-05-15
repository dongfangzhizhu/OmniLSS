# Benchmarking Principles

OmniLSS benchmarks must be transparent scientific measurements, not marketing
claims.

## Required Metrics

| Metric | Required |
| --- | --- |
| JIT compile time | Yes |
| Warm runtime | Yes |
| Cold runtime | Yes |
| Memory usage | Yes |
| GPU speed | Yes |
| Batch scaling | Yes |

## Reporting Rules

- Report hardware, backend, dtype, input shape, and number of repetitions.
- Separate compile/setup time from execution time. Python reports cold and warm times separately; R timings must exclude `Rscript` startup, package loading, and CSV parsing when used for speed ratios.
- Avoid claims like `149x faster!!!` without context, methodology, and baseline.
- Include variance or confidence intervals when practical. Use synchronized JAX timings (`jax.block_until_ready()`) for Python paths.
