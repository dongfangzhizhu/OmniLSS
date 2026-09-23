import numpy as np
import pytest

from omnilss import gamlss, load_model, save_model


def _roundtrip_for_family(family: str, path):
    rng = np.random.default_rng(0)
    n = 80
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.3, size=n)
    if family == "GA":
        y = np.exp(y / 3.0)
    data = {"y": y, "x": x}
    model = gamlss("y ~ x", family=family, data=data)

    save_model(model, path)
    loaded = load_model(path)

    newdata = {"x": np.array([-1.0, 0.0, 1.0])}
    p1 = model.predict_params(newdata)
    p2 = loaded.predict_params(newdata)
    for k in p1:
        np.testing.assert_allclose(
            np.asarray(p1[k]), np.asarray(p2[k]), atol=1e-10, rtol=1e-10
        )


def test_save_load_no(tmp_path):
    _roundtrip_for_family("NO", tmp_path / "no-model.omnilss")


def test_save_load_ga(tmp_path):
    _roundtrip_for_family("GA", tmp_path / "ga-model.omnilss")


def test_explicit_pickle_roundtrip_for_trusted_objects(tmp_path):
    """The opt-in pickle API remains available for trusted Python objects."""
    pytest.importorskip("cloudpickle")
    from omnilss import load_model_pickle, save_model_pickle

    path = tmp_path / "trusted-object.pkl"
    expected = {"values": np.array([1.0, 2.0, 3.0]), "label": "trusted"}

    save_model_pickle(expected, path)
    actual = load_model_pickle(path)

    assert actual["label"] == expected["label"]
    np.testing.assert_array_equal(actual["values"], expected["values"])
