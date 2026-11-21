# Distribution Families & Capability Gates

[中文版本](tutorial-families_cn.md)

OmniLSS ports the `gamlss.dist` family system. Each family defines one to four
distribution parameters (`mu`, `sigma`, `nu`, `tau`) with default link
functions, analytic or automatic-differentiation score/Hessian slots, and
`d`/`p`/`q`/`r` kernels.

## 22 Production-Safe Admitted Families

Every family below passed all nine hard admission gates in
`docs/development/m3-admission-evidence-2026-11-15.json` (live-R derivative,
DPQR, and RS deviance parity; validated R-consistency tests; golden fit
fixture; live `newdata` predictions within `2e-4`; JSON round-trip drift
`<= 1e-10`; converged refit; and zero OmniLSS-side layered route mismatches):

| Support | Production-Safe Families | Typical Use |
|---|---|---|
| Real line `(-inf, inf)` | `NO`, `NO2`, `GU`, `RG`, `LO`, `PE`, `NET` | Symmetric, heavy-tailed, or skewed continuous responses |
| Positive `(0, inf)` | `EXP`, `GA`, `IG`, `IGAMMA`, `LOGNO`, `LOGNO2`, `WEI` | Positive continuous / duration / claim-severity data |
| Unit interval `(0, 1)` / `[0, 1]` | `BE`, `BEINF` | Proportions, rates, and 0/1-inflated proportions |
| Zero-adjusted `[0, inf)` | `ZAGA` | Positive continuous responses with point mass at 0 |
| Counts `{0, 1, ...}` / Binomial | `PO`, `GEOM`, `NBI`, `NBII`, `BI` | Count data, overdispersed counts, and binomial trials |

An additional 104 registered families are available at `experimental` tier, and
20 `gen.Family` skeleton constructors remain explicitly marked `compat_alias`
(0 `placeholder` families remain).

## Modeling Multiple Parameters

```python
import numpy as np
from omnilss import NO, gamlss

rng = np.random.default_rng(0)
x = rng.normal(size=120)
y = rng.normal(loc=1.0 + 0.5 * x, scale=np.exp(-0.4 + 0.25 * x))

model = gamlss(
    "y ~ x",
    sigma_formula="~ x",
    family=NO(),
    data={"y": y, "x": x},
)
```

Use `parameter_formulas={"nu": "~ x", "tau": "~ 1"}` for 3- and 4-parameter
families, or pass `sigma_fix=True, sigma_start=0.5` to hold a parameter fixed.

## Fail-Closed Capability Gates

Registration is intentionally separated from validation. Inspect capability
tiers or enforce fail-closed production routing via `omnilss.family_capabilities`:

```python
from omnilss.family_capabilities import (
    capability_matrix,
    get_family_capability,
    method_route_capability_report,
    require_production_family_route,
)

cap = get_family_capability("ZAGA")
print(cap.implementation_status, cap.status("rs_fit"), cap.status("cg_fit"))

# Raises ProductionFamilyRejectedError for experimental or compat_alias routes
require_production_family_route("GA", "RS", route="service")
report = method_route_capability_report("GA", "CG", strict=True)
matrix = capability_matrix()
```

Passing `strict_capabilities=True` to `gamlss(...)` rejects any experimental
family/route rather than silently fitting unvalidated kernels.

## Transformed, Censored, and Truncated Families

- `gen_Family("NO", type="log")` / `Family("NO", "logit")` build log- and
  logit-transformed families with wrapped response derivatives.
- `omnilss.cens` (`gen_cens`, `cens`) and `omnilss.tr` (`gen_trun`, `trun`)
  wrap base families for censored and truncated responses.

## Next Steps

- [Fitting Algorithms (RS / CG / MIXED / HYBRID)](tutorial-fitting-algorithms.md)
