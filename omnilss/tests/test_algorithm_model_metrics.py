import numpy as np

from omnilss.algorithms import _model_metrics


class _SmoothFit:
    basis_columns = (1, 4)


class _SmoothInfo:
    smooth_fits = [_SmoothFit()]


def test_df_fit_with_smooth_edf_replaces_nominal_basis_columns(monkeypatch):
    def fake_compute_smooth_edf(X, w, smooth_fits):
        assert X.shape == (5, 4)
        assert w.shape == (5,)
        assert len(smooth_fits) == 1
        return 1.5

    monkeypatch.setattr(_model_metrics, "compute_smooth_edf", fake_compute_smooth_edf)

    df_fit, smooth_edf = _model_metrics.df_fit_with_smooth_edf(
        coefficients={"mu": np.zeros(4), "sigma": np.zeros(1)},
        estimable_parameters=("mu", "sigma"),
        design_matrices={"mu": np.ones((5, 4)), "sigma": np.ones((5, 1))},
        weights=np.ones(5),
        smooth_infos={"mu": _SmoothInfo()},
    )

    assert df_fit == 3.5
    assert smooth_edf == {"mu": 1.5, "sigma": 0.0}
