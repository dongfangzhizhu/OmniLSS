import numpy as np
import pytest

from omnilss import plotBEINF


def test_plot_beinf_returns_curve_and_parameters():
    result = plotBEINF(mu=0.4, sigma=0.3, nu=0.2, tau=0.6, n=51)
    entry = result.entries[0]
    assert result.family == "BEINF"
    assert entry.y.shape == (51,)
    assert entry.density.shape == (51,)
    assert np.all(np.isfinite(entry.density))
    assert entry.parameters == {"mu": 0.4, "sigma": 0.3, "nu": 0.2, "tau": 0.6}
    assert entry.point_mass_zero == pytest.approx(0.2 / 1.8)
    assert entry.point_mass_one == pytest.approx(0.6 / 1.8)


def test_plot_beinf_validates_grid():
    with pytest.raises(ValueError, match="no_points"):
        plotBEINF(n=1)
    with pytest.raises(ValueError, match="greater"):
        plotBEINF(from_value=1.0, to_value=0.0)