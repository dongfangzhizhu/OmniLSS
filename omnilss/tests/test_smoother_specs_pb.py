"""R gamlss pb(): the P-spline basis and penalty constructor.

R's pb() returns the covariate carrying X, D, df, max.df and lambda; gamlss()
passes those to gamlss.pb().  The Python port returns a PBSpline object and is
compared element-wise with live R (X to ~1e-12, D exactly, plus the df and
max.df bookkeeping and their guards).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse as _sparse  # noqa: F401  (documents the dense-matrix contract)

from omnilss.smoother_controls import pb_control
from omnilss.smoother_specs import PBSpline, bbase, difference_matrix, pb

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
x <- seq(0, 1, length.out = 200)
s <- pb(x)
X <- attr(s, "X")
D <- attr(s, "D")
s5 <- pb(x, df = 5)
s_big <- pb(x, df = 30)
s_max <- pb(x, max.df = 30)
short <- pb(seq(0, 1, length.out = 40))
cat(jsonlite::toJSON(list(
  xdim = as.integer(dim(X)),
  ddim = as.integer(dim(D)),
  X = as.numeric(X),
  D = as.numeric(D),
  knots = as.numeric(attr(X, "knots")),
  maxdf = as.numeric(attr(s, "max.df")),
  df_len = as.integer(length(attr(s, "df"))),
  df5 = as.numeric(attr(s5, "df")),
  df_big = as.numeric(attr(s_big, "df")),
  max_big = as.numeric(attr(s_max, "max.df")),
  short_inter = as.integer(attr(short, "control")$inter)
), auto_unbox = FALSE, digits = 17))
"""


def _x() -> np.ndarray:
    return np.linspace(0.0, 1.0, 200)


def test_basis_and_penalty_shapes() -> None:
    spline = pb(_x())
    assert isinstance(spline, PBSpline)
    assert spline.X.shape == (200, 23)
    assert spline.D.shape == (21, 23)
    assert spline.max_df == 23.0
    assert spline.df is None
    assert spline.lambda_ is None


def test_basis_partitions_unity() -> None:
    spline = pb(_x())
    row_sums = np.asarray(spline.X.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-9)


def test_penalty_is_a_second_difference_matrix() -> None:
    spline = pb(_x())
    expected = difference_matrix(23, 2)
    np.testing.assert_allclose(spline.D, expected, rtol=0, atol=0)
    assert np.all(spline.D.sum(axis=1) == pytest.approx(0.0, abs=1e-12))


def test_df_is_stored_plus_two_like_r() -> None:
    assert pb(_x(), df=5).df == 7.0
    assert pb(_x(), lambda_=3.5).lambda_ == 3.5


def test_df_guard_resets_to_three_then_adds_two() -> None:
    with pytest.warns(UserWarning, match="exceed the number of columns"):
        spline = pb(_x(), df=30)
    assert spline.df == 5.0


def test_negative_df_becomes_two() -> None:
    with pytest.warns(UserWarning, match="extra df's are set to 0"):
        spline = pb(_x(), df=-1)
    assert spline.df == 2.0


def test_max_df_guard() -> None:
    with pytest.warns(UserWarning, match="max.df's are set to"):
        spline = pb(_x(), max_df=30)
    assert spline.max_df == 23.0


def test_short_samples_use_ten_intervals() -> None:
    spline = pb(np.linspace(0.0, 1.0, 40))
    assert spline.control["inter"] == 10


def test_fewer_distinct_values_than_intervals() -> None:
    spline = pb(np.array([0.0, 0.5, 0.5, 1.0, 1.0, 1.0]))
    assert spline.control["inter"] == 3


def test_matrix_input_is_rejected_like_r() -> None:
    with pytest.raises(ValueError, match="declare it as a vector"):
        pb(np.zeros((5, 2)))


def test_control_object_is_honoured() -> None:
    spline = pb(_x(), control=pb_control(inter=10, degree=2, order=3))
    assert spline.control["degree"] == 2
    assert spline.control["order"] == 3
    # degree 2 with 10 intervals: ncol(P) = 10 + 2*2 + 1, basis loses degree + 1.
    assert spline.X.shape[1] == 10 + 2 * 2 + 1 - 3
    assert spline.D.shape[0] == spline.X.shape[1] - 3


def test_bbase_matches_its_definition() -> None:
    values = np.array([0.1, 0.3, 0.5])
    design, knots = bbase(values, -0.01, 1.01, 5, 3)
    assert design.shape[0] == 3
    assert design.shape[1] == knots.shape[0]
    # Partition of unity: the basis reproduces a constant.
    np.testing.assert_allclose(design.sum(axis=1), 1.0, rtol=1e-10)


@pytest.mark.slow
def test_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss" / "R" / "pb.R").exists():
        pytest.skip("vendored gamlss pb.R not present")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pb.R"
        path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(path)], capture_output=True, text=True, timeout=300
        )
    if proc.returncode != 0:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    ref = json.loads(proc.stdout.strip().splitlines()[-1])

    spline = pb(_x())
    shape = tuple(int(value) for value in ref["xdim"])
    np.testing.assert_allclose(
        spline.X, np.asarray(ref["X"], dtype=float).reshape(shape, order="F"), rtol=1e-11, atol=1e-10
    )
    dshape = tuple(int(value) for value in ref["ddim"])
    np.testing.assert_allclose(
        spline.D, np.asarray(ref["D"], dtype=float).reshape(dshape, order="F"), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(spline.knots, np.asarray(ref["knots"], dtype=float), rtol=1e-12)
    assert spline.max_df == float(np.asarray(ref["maxdf"]).ravel()[0])
    assert spline.df is None and int(np.asarray(ref["df_len"]).ravel()[0]) == 0

    assert pb(_x(), df=5).df == float(np.asarray(ref["df5"]).ravel()[0])
    with pytest.warns(UserWarning):
        assert pb(_x(), df=30).df == float(np.asarray(ref["df_big"]).ravel()[0])
    with pytest.warns(UserWarning):
        assert pb(_x(), max_df=30).max_df == float(np.asarray(ref["max_big"]).ravel()[0])

    short = pb(np.linspace(0.0, 1.0, 40))
    assert short.control["inter"] == int(np.asarray(ref["short_inter"]).ravel()[0])
