import numpy as np

from omnilss.fitting import gamlss
from omnilss.distributions import resolve_family


MVP_FAMILIES = ("NO", "GA", "TF", "BE", "ZIP", "ZINBI")


def _make_data(name: str, n: int = 128):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    if name == "NO":
        y = 1.0 + 0.4 * x + rng.normal(scale=0.5, size=n)
    elif name == "GA":
        y = np.exp(0.8 + 0.2 * x + rng.normal(scale=0.2, size=n))
    elif name == "TF":
        y = 0.5 + 0.3 * x + rng.standard_t(df=4, size=n)
    elif name == "BE":
        z = -0.2 + 0.4 * x + rng.normal(scale=0.5, size=n)
        y = 1.0 / (1.0 + np.exp(-z))
        y = np.clip(y, 1e-5, 1.0 - 1e-5)
    elif name == "ZIP":
        lam = np.exp(0.6 + 0.1 * x)
        y = rng.poisson(lam)
        y[rng.uniform(size=n) < 0.2] = 0
    elif name == "ZINBI":
        mu = np.exp(0.6 + 0.1 * x)
        y = rng.poisson(mu)
        y[rng.uniform(size=n) < 0.25] = 0
    else:
        raise ValueError(name)
    return {"x": x, "y": y}


def test_phase0_mvp_families_rs_smoke_stable():
    for fam in MVP_FAMILIES:
        data = _make_data(fam)
        model = gamlss("y ~ x", family=fam, data=data, method="RS")
        assert np.isfinite(float(model.g_dev))


def test_phase0_mvp_family_definitions_present():
    for fam in MVP_FAMILIES:
        family = resolve_family(fam)
        assert family.name == fam
        assert len(family.estimable_parameters) >= 1
        assert "mu" in family.parameters
