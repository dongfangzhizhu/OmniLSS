import warnings

from omnilss.algorithms.cg_algorithm import cg_fit


def test_cg_fit_deprecated_alias_warns():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            cg_fit(formula="y ~ x", family="NO", data={"y": [1.0,2.0], "x":[1.0,2.0]})
        except Exception:
            pass
    assert any(isinstance(w.message, DeprecationWarning) for w in rec)
