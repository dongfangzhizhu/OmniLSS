# CG cross-derivative reference smoke artifact (2026-05-17)

[中文版本](cg-cross-derivative-reference-2026-05-17_cn.md)

This artifact records a small deterministic smoke comparison for the CG roadmap.
It is **not** a performance claim and must not be used as evidence that one
backend is faster in general.  Its purpose is to keep a checked-in reference for
backend diagnostics and deviance trajectories after the full-Hessian and
eta-level cross-derivative implementations were connected.

Raw machine-readable results are stored in
`docs/benchmarks/cg-cross-derivative-reference-2026-05-17.json`.

## Environment

- Python: 3.14.4
- Platform: Linux-6.12.47-x86_64-with-glibc2.39
- JAX: 0.10.0
- `jax_enable_x64`: true
- Devices: `cpu:0`

## Dataset

- Family: `NO`
- `n`: 30
- Formula: `y ~ x`
- Sigma formula: `~ x`
- Data: deterministic grid with sinusoidal residual pattern
- Control: `n_cyc=8`, `c_crit=1e-4`

## Results

| Backend | Final deviance | Iterations | Converged | Cross-derivative status |
|---|---:|---:|---:|---|
| `RS` | -56.815030 | 8 | false | n/a |
| `CG_FULL_HESSIAN` | -31.994175 | 8 | false | `full_hessian` |
| `CG_IRLS_CROSS` | -49.176226 | 8 | false | `eta_correction` |

## Interpretation

- Both CG backends record explicit cross-derivative diagnostics.
- The checked-in trajectories are smoke references only; all three backends were
  stopped by the same short iteration budget and did not converge.
- Broader benchmark claims still require generated benchmark artifacts with
  hardware, backend, sample sizes, repetitions, and R availability recorded.
