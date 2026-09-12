import numpy as np

from omnilss import plotBEINF1


def test_plot_beinf1_returns_vectorized_curves():
    result = plotBEINF1(mu=[0.3, 0.7], sigma=0.4, tau=[0.2, 0.8], n=41)
    assert result.family == "BEINF1"
    assert len(result.entries) == 2
    assert all(entry.y.shape == (41,) for entry in result.entries)
    assert all(np.all(np.isfinite(entry.density)) for entry in result.entries)
    assert result.entries[0].parameters["tau"] == 0.2
    assert result.entries[1].parameters["tau"] == 0.8