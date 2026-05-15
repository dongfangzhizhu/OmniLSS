import pytest
pytest.importorskip("cloudpickle")
import numpy as np

from omnilss import gamlss, save_model, load_model


def _roundtrip_for_family(family: str):
    rng = np.random.default_rng(0)
    n = 80
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.3, size=n)
    if family == "GA":
        y = np.exp(y / 3.0)
    data = {"y": y, "x": x}
    model = gamlss("y ~ x", family=family, data=data)

    path = "/tmp/omnilss_model.pkl"
    save_model(model, path)
    loaded = load_model(path)

    newdata = {"x": np.array([-1.0, 0.0, 1.0])}
    p1 = model.predict_params(newdata)
    p2 = loaded.predict_params(newdata)
    for k in p1:
        np.testing.assert_allclose(np.asarray(p1[k]), np.asarray(p2[k]), atol=1e-10, rtol=1e-10)


def test_save_load_no():
    _roundtrip_for_family("NO")


def test_save_load_ga():
    _roundtrip_for_family("GA")
