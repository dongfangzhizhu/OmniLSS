# Cross Derivatives Verification Report (Week 1 Day 5)

> Date: 2026-05-19  
> Chinese version: [CROSS_DERIVATIVES_VERIFICATION_2026_05_19_cn.md](./CROSS_DERIVATIVES_VERIFICATION_2026_05_19_cn.md)

## Scope

This report verifies the newly introduced cross-parameter derivative infrastructure:

- `omnilss/src/omnilss/derivatives/cross_derivatives.py`
- `cross_hessian(...)` generic AD mixed-Hessian utility
- `cross_hessian_from_family(...)` eta-scale mixed derivatives for OmniLSS families

## Verification Items

1. **Family-level structure checks (NO / GA / WEI)**
   - shape correctness for each `(param_i, param_j)` result
   - finiteness checks
   - Hessian cross-term symmetry checks

2. **Generic AD correctness checks**
   - analytic quadratic case (`ll_i = x_i^2 + 3 x_i y_i + 2 y_i^2`) with exact Hessian blocks
   - finite-difference second-order gradient check via `jax.test_util.check_grads`

## Results

- All Week 1 derivative unit tests passed locally.
- The analytic quadratic case matched expected exact Hessian values.
- Finite-difference gradient checking passed for second-order derivatives (`order=2`).

## Acceptance Mapping (Week 1)

- Cross-derivative infrastructure implemented: **Done**.
- Initial numerical verification for NO distribution and companion families: **Done (local AD checks)**.
- R internal derivative direct comparison (`d2ldmds`) remains a follow-up task when R bridge internals are wired for direct derivative calls.

## Commands Executed

```bash
python -m pytest omnilss/tests/test_cross_derivatives.py -q
```
