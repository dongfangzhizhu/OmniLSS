# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.ggplots bivariate surfaces (D2.261): bivar_fun / bivar_fun_plotly /
bivar_pdf.

The grids follow R: expand.grid(x, y) with x varying fastest, so the Z matrix
has rows by x and columns by y (matrix(z, nrow = n)).  bivar_pdf's 50% HDR
threshold is R's algorithm verbatim: sort by density, accumulate z * dx * dy,
normalise by the maximum, take the first cell reaching 0.5.

Analytic check for a standard bivariate normal: the 50% HDR level is
exp(-log 2) / (2 pi) = 1 / (4 pi) = 0.0795775.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from omnilss import BivarSurface, bivar_fun, bivar_fun_plotly, bivar_pdf


def _normal_density(x, y):
    return np.exp(-(x**2 + y**2) / 2.0) / (2.0 * np.pi)


def test_bivar_fun_grid_and_orientation():
    result = bivar_fun(_normal_density, xlim=(-2, 2), ylim=(-1, 1), n=21, plot=False)
    assert isinstance(result, BivarSurface)
    assert result.x.size == 21 and result.y.size == 21
    assert result.z.shape == (21, 21)
    assert result.frame.shape == (21 * 21, 3)
    # expand.grid order: x varies fastest.
    assert result.frame["x"].iloc[0] == pytest.approx(-2.0)
    assert result.frame["y"].iloc[0] == pytest.approx(-1.0)
    assert result.frame["x"].iloc[1] > result.frame["x"].iloc[0]
    # Z[i, j] = f(x_i, y_j), like R's matrix(z, nrow = n).
    assert result.z[0, 0] == pytest.approx(_normal_density(-2.0, -1.0))
    assert result.z[0, 5] == pytest.approx(_normal_density(-2.0, result.y[5]))
    assert result.z[7, 0] == pytest.approx(_normal_density(result.x[7], -1.0))
    assert result.z.max() == pytest.approx(1.0 / (2.0 * np.pi), rel=0.02)


def test_bivar_fun_falls_back_to_scalar_evaluation():
    def scalar_only(x, y):
        if np.ndim(x) or np.ndim(y):
            raise TypeError("scalar inputs only")
        return float(np.exp(-(x**2 + y**2)))

    result = bivar_fun(scalar_only, xlim=(-1, 1), ylim=(-1, 1), n=9, plot=False)
    assert result.z.shape == (9, 9)
    assert result.z[4, 4] == pytest.approx(1.0, rel=1e-6)


def test_bivar_fun_validates_its_function():
    with pytest.raises(ValueError, match="bivariate function"):
        bivar_fun()
    with pytest.raises(ValueError, match="bivariate function"):
        bivar_fun("not callable")


def test_bivar_fun_plotly_returns_the_surface_matrix():
    result = bivar_fun_plotly(_normal_density, xlim=(-2, 2), ylim=(-2, 2), n=15, plot=False)
    assert result.z.shape == (15, 15)
    assert result.z[0, 0] == pytest.approx(_normal_density(-2.0, -2.0))
    assert result.z[-1, -1] == pytest.approx(_normal_density(2.0, 2.0))
    assert result.figure is None  # plot=False
    with pytest.raises(ValueError, match="bivariate function"):
        bivar_fun_plotly()


def test_bivar_pdf_recovers_the_analytic_hdr_level():
    pdf = lambda x, y: float(stats.multivariate_normal.pdf([x, y], [0, 0], [[1, 0], [0, 1]]))
    result = bivar_pdf(pdf, xlim=(-3, 3), ylim=(-3, 3), n=120, plot=False)
    assert result.mass == pytest.approx(1.0, abs=0.01)
    # 50% HDR of a standard bivariate normal: 1 / (4 pi).
    assert result.level == pytest.approx(1.0 / (4.0 * np.pi), rel=2e-3)
    # The region above the threshold carries just over half the mass.
    dx = result.x[1] - result.x[0]
    dy = result.y[1] - result.y[0]
    inside = float(np.sum(result.z[result.z >= result.level])) * dx * dy / result.mass
    assert 0.45 < inside < 0.60


def test_bivar_pdf_grid_matches_the_function_and_validates():
    pdf = lambda x, y: float(np.exp(-(x**2 + y**2) / 2.0) / (2.0 * np.pi))
    result = bivar_pdf(pdf, xlim=(-2, 2), ylim=(-2, 2), n=31, plot=False)
    assert result.z.shape == (31, 31)
    assert result.z[0, 0] == pytest.approx(pdf(-2.0, -2.0))
    assert result.z[15, 15] == pytest.approx(pdf(result.x[15], result.y[15]))
    assert result.level > 0.0
    with pytest.raises(ValueError, match="density function"):
        bivar_pdf()
    with pytest.raises(ValueError, match="density function"):
        bivar_pdf(42)
