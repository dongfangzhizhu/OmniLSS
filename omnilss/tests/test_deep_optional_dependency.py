import importlib


def test_deep_module_importable():
    mod = importlib.import_module("omnilss.deep")
    assert hasattr(mod, "fit_deep_gamlss")
