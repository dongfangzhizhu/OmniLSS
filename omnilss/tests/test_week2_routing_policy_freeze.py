import numpy as np

import omnilss.config as cfg
from omnilss import gamlss
from omnilss.distributions import NO


def test_auto_routing_prefers_rs_for_small_p_on_gpu_threshold_hit(monkeypatch):
    # Force backend detection to GPU and crossover reached.
    monkeypatch.setattr(cfg, "_current_backend", lambda: ("gpu", ["fake-gpu"]))
    monkeypatch.setitem(cfg.GPU_CROSSOVER_N, "NO", 1)

    rng = np.random.default_rng(7)
    n = 64
    x = rng.normal(size=n)
    y = 1.0 + 0.3 * x + rng.normal(scale=0.2, size=n)

    model = gamlss(
        formula="y ~ x",  # estimated p = 2 (intercept + x)
        family=NO(),
        data={"x": x, "y": y},
        method="auto",
    )

    routing = model.get_method_routing(validate=False)
    assert routing is not None
    assert routing["selected_method"] == "RS"
    assert routing["reason"] == "small_p_prefers_numpy_rs"


def test_estimate_formula_width_handles_simple_forms():
    from omnilss.fitting import _estimate_formula_width

    assert _estimate_formula_width("y ~ 1") == 1
    assert _estimate_formula_width("y ~ x1") == 2
    assert _estimate_formula_width("y ~ x1 + x2") == 3
    assert _estimate_formula_width("y ~ 0 + x1 + x2") == 2
    # Unknown/complex smooth forms intentionally return None to avoid bad routing assumptions.
    assert _estimate_formula_width("y ~ pb(x1)") is None


def test_auto_routing_prefers_rs_jax_for_large_p_on_gpu(monkeypatch):
    # Force backend detection to GPU and keep crossover threshold low enough to trigger JAX route.
    monkeypatch.setattr(cfg, "_current_backend", lambda: ("gpu", ["fake-gpu"]))
    monkeypatch.setitem(cfg.GPU_CROSSOVER_N, "NO", 2)

    rng = np.random.default_rng(17)
    n = 64
    data = {"y": None}
    formula_terms = []
    signal = 0.2
    for i in range(1, 9):
        name = f"x{i}"
        vals = rng.normal(size=n)
        data[name] = vals
        formula_terms.append(name)
        signal += 0.05 * vals
    data["y"] = signal + rng.normal(scale=0.2, size=n)

    model = gamlss(
        formula="y ~ " + " + ".join(formula_terms),
        family=NO(),
        data=data,
        method="auto",
    )

    routing = model.get_method_routing(validate=False)
    assert routing is not None
    assert routing["selected_method"] == "RS_JAX"
    assert routing["reason"] == "gpu_crossover_reached"

