# Hessian Audit (Phase-1 T-02)

Date: 2026-05-15

This report tracks hand-written Hessian usage in `omnilss/src/omnilss/distributions.py` for core families.

| Family | Parameter | Current Hessian style | Status |
|---|---|---|---|
| NO | mu | Expected | ✅ |
| NO | sigma | **Expected (updated in this patch)** | ✅ |
| GA | mu | Expected | ✅ |
| GA | sigma | Expected (R formula with trigamma) | ✅ |
| LO | mu/sigma | Expected (explicit override) | ✅ |
| BE | mu/sigma | Expected (explicit override) | ✅ |
| TF | mu/sigma/nu | Observed (autodiff Hessian) | ⚠️ pending decision |

## Notes
- In this patch, NO sigma second derivative was switched from observed Hessian form to expected Hessian form: `-2/sigma^2`.
- TF still uses autodiff observed Hessian and requires a dedicated design decision before conversion.
- AD-generated families in `distributions_b*.py` are out of scope for this audit iteration.
