import numpy as np

from omnilss import plotBEINF0


def test_plot_beinf0_returns_vectorized_curves():
    result = plotBEINF0(mu=[0.3, 0.7], sigma=0.4, nu=[0.2, 0.8], n=41)
    assert result.family == "BEINF0"
    assert len(result.entries) == 2
    assert all(entry.y.shape == (41,) for entry in result.entries)
    assert all(np.all(np.isfinite(entry.density)) for entry in result.entries)
    assert result.entries[0].parameters["nu"] == 0.2
    assert result.entries[1].parameters["nu"] == 0.8
    assert result.entries[0].point_mass_zero == 0.2 / 1.2
    assert result.entries[0].point_mass_one == 0.0