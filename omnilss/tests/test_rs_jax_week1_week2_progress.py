import numpy as np

from omnilss import gamlss
from omnilss.distributions import NO


def test_rs_jax_emits_week1_week2_observability_slots():
    rng = np.random.default_rng(42)
    n = 128
    x = rng.normal(size=n)
    y = 2.0 + 0.5 * x + rng.normal(scale=0.3, size=n)

    model = gamlss(
        formula="y ~ x",
        family=NO(),
        data={"x": x, "y": y},
        method="RS_JAX",
    )

    slots = dict(model.additional_slots)
    split = slots.get("timing_split")
    assert isinstance(split, dict)
    for k in ("warm_start", "core_rs", "host_device_sync"):
        assert k in split
        assert split[k] is not None
        assert float(split[k]) >= 0.0

    assert "beta_norm_first_step" in slots
    assert float(slots["beta_norm_first_step"]) >= 0.0
    assert "beta_explosion_threshold" in slots
    assert float(slots["beta_explosion_threshold"]) > 0.0
    assert "beta_fallback_applied" in slots
    assert isinstance(slots["beta_fallback_applied"], bool)

    assert "eta_clip_ratio" in slots
    ratio = float(slots["eta_clip_ratio"])
    assert 0.0 <= ratio <= 1.0

    assert slots.get("device_aware_suggested_backend") in {"numpy", "jax"}
    assert int(slots.get("p_total", 0)) >= 1
