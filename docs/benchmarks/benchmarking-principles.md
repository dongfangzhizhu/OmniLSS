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
- Separate compile time from execution time.
- Avoid claims like `149x faster!!!` without context, methodology, and baseline.
- Include variance or confidence intervals when practical.
