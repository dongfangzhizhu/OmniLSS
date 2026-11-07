"""R gamlss.prepdata scaling / model-matrix / interaction helpers.

data_scale(), data_vars2data() and data_form2X() all end up in model.matrix();
the port reproduces R's column names (treatment contrasts, first level dropped,
":" for interactions) so the results can be compared element by element.
data_inter() is the pairwise likelihood-ratio interaction screen.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omnilss.data_prep import data_form2X, data_inter, data_scale, data_vars2data

_REPO = Path(__file__).resolve().parents[2]

_FOREACH = """foreach <- function(...) {
  args <- list(...); nm <- names(args)
  iter <- nm[nzchar(nm) & !startsWith(nm, ".")][1]
  structure(list(name = iter, values = args[[iter]],
                 combine = if (".combine" %in% nm) args[[".combine"]] else NULL),
            class = "foreachStub")
}
`%do%` <- function(obj, expr) {
  e <- substitute(expr); env <- parent.frame()
  out <- lapply(obj$values, function(v) { assign(obj$name, v, envir = env); eval(e, env) })
  if (is.null(obj$combine)) out else do.call(obj$combine, out)
}
`%dopar%` <- function(obj, expr) {
  e <- substitute(expr); env <- parent.frame()
  out <- lapply(obj$values, function(v) { assign(obj$name, v, envir = env); eval(e, env) })
  if (is.null(obj$combine)) out else do.call(obj$combine, out)
}
"""

_RSCRIPT = _FOREACH + """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
source("R/gamlss.prepdata/R/data_part.R")
source("R/gamlss.prepdata/R/data_str.R")
source("R/gamlss.prepdata/R/data_scale.R")
source("R/gamlss.prepdata/R/data_inter.R")
# data_inter() computes y_zscores(actY) and never uses it; the stub keeps the
# reference run from needing a gamlssML SHASHo fit.
y_zscores <- function(x, ...) x
a <- commandArgs(trailingOnly = TRUE); d <- fromJSON(a[1])
df <- data.frame(y = d$y, x1 = d$x1, x2 = d$x2, g = factor(d$g))
y <- df$y
s1 <- data_scale(df, response = y, scale.to = "0to1")
v1 <- data_vars2data(df, response = "y", type = "main.effect")
v2 <- data_vars2data(df, response = "y", type = "first.order")
f1 <- data_form2X(df, formula = y ~ x1 + g, response = "y")
f2 <- data_form2X(df, formula = y ~ (x1 + x2)^2, response = "y")
i1 <- data_inter(df[, c("y", "x1", "x2")], response = y, plot = FALSE, print.info = FALSE)
out <- list(
  scale_names = names(s1), scale_x1 = as.numeric(s1$x1), scale_x2 = as.numeric(s1$x2),
  scale_g = as.integer(s1$g), scale_y = as.numeric(s1$y),
  v1_names = names(v1), v1 = as.numeric(as.matrix(v1[, -1])),
  v2_names = names(v2), v2 = as.numeric(as.matrix(v2[, -1])),
  f1_names = colnames(f1), f1 = as.numeric(f1), f1_dim = dim(f1),
  f2_names = colnames(f2), f2 = as.numeric(f2), f2_dim = dim(f2),
  i1 = as.numeric(i1), i1_names = colnames(i1))
cat("JSONSTART"); cat(toJSON(out, auto_unbox = FALSE, digits = 17))
"""


def _data(seed: int = 3, n: int = 30) -> pd.DataFrame:
    rs = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "y": rs.normal(size=n),
            "x1": rs.normal(size=n),
            "x2": rs.normal(size=n),
            "g": pd.Categorical(rs.choice(["a", "b", "c"], n), categories=["a", "b", "c"]),
        }
    )


def test_data_scale_min_max_maps_the_covariates() -> None:
    frame = _data()
    scaled = data_scale(frame, response="y", scale_to="0to1")
    for name in ("x1", "x2"):
        values = scaled[name].to_numpy()
        assert values.min() == pytest.approx(0.0)
        assert values.max() == pytest.approx(1.0)
    np.testing.assert_allclose(scaled["y"].to_numpy(), frame["y"].to_numpy(), rtol=0)
    assert list(scaled["g"]) == list(frame["g"])


def test_data_scale_uses_the_response_position() -> None:
    frame = _data()
    shifted = frame.rename(columns={"y": "response"})
    scaled = data_scale(shifted, position_response=1, scale_to="0to1")
    assert scaled["x1"].min() == pytest.approx(0.0)
    np.testing.assert_allclose(scaled["response"].to_numpy(), frame["y"].to_numpy(), rtol=0)


def test_data_scale_validates_and_refuses_the_z_score_branch() -> None:
    frame = _data()
    with pytest.raises(ValueError, match=r"response \(or its position\) should be given"):
        data_scale(frame, scale_to="0to1")
    with pytest.raises(ValueError, match="should be one of"):
        data_scale(frame, response="y", scale_to="log")
    with pytest.raises(NotImplementedError, match="gamlssML"):
        data_scale(frame, response="y", scale_to="z-scores")
    with pytest.raises(ValueError, match="is not a column of the data"):
        data_scale(frame, response="zzz", scale_to="0to1")


def test_data_scale_respects_the_factor_heuristic() -> None:
    frame = _data()
    frame["small"] = np.arange(len(frame)) % 3  # fewer than 10 distinct values
    scaled = data_scale(frame, response="y", scale_to="0to1")
    np.testing.assert_allclose(scaled["small"].to_numpy(), frame["small"].to_numpy(), rtol=0)


def test_data_vars2data_main_effect_and_first_order() -> None:
    frame = _data()
    main = data_vars2data(frame, response="y", type="main.effect")
    assert list(main.columns) == ["y", "x1", "x2", "gb", "gc"]
    np.testing.assert_allclose(main["gb"].to_numpy(), (frame["g"].cat.codes == 1).astype(float), rtol=0)
    first = data_vars2data(frame, response="y", type="first.order")
    assert list(first.columns) == [
        "y", "x1", "x2", "gb", "gc", "x1.x2", "x1.gb", "x1.gc", "x2.gb", "x2.gc",
    ]
    np.testing.assert_allclose(
        first["x1.x2"].to_numpy(), frame["x1"].to_numpy() * frame["x2"].to_numpy(), rtol=1e-12
    )


def test_data_vars2data_validates() -> None:
    frame = _data()
    with pytest.raises(ValueError, match="response should be given"):
        data_vars2data(frame)
    with pytest.raises(ValueError, match="should be one of"):
        data_vars2data(frame, response="y", type="second.order")
    with pytest.raises(NotImplementedError, match="model.matrix"):
        data_vars2data(frame, response="y", nonlinear=True)
    excluded = data_vars2data(frame, response="y", exclude=["x2"])
    assert "x2" not in excluded.columns


def test_data_form2X_builds_the_design_matrix() -> None:
    frame = _data()
    design = data_form2X(frame, formula="y ~ x1 + g", response="y")
    assert list(design.columns) == ["x1", "gb", "gc"]
    interaction = data_form2X(frame, formula="y ~ (x1 + x2)^2", response="y")
    assert list(interaction.columns) == ["x1", "x2", "x1:x2"]
    np.testing.assert_allclose(
        interaction["x1:x2"].to_numpy(),
        frame["x1"].to_numpy() * frame["x2"].to_numpy(),
        rtol=1e-12,
    )


def test_data_form2X_without_a_formula_needs_the_response_first() -> None:
    frame = _data()
    design = data_form2X(frame, response="y")
    assert list(design.columns) == ["x1", "x2", "gb", "gc"]
    moved = frame[["x1", "y", "x2", "g"]]
    with pytest.raises(ValueError, match="should be first in data"):
        data_form2X(moved, response="y")


def test_data_form2X_can_scale_first() -> None:
    frame = _data()
    design = data_form2X(frame, formula="y ~ x1", response="y", scale_to="0to1")
    assert design["x1"].min() == pytest.approx(0.0)
    assert design["x1"].max() == pytest.approx(1.0)


def test_data_inter_returns_the_pairwise_p_values() -> None:
    frame = _data()
    matrix = data_inter(frame[["y", "x1", "x2"]], response="y", plot=False, print_info=False)
    assert list(matrix.columns) == ["x1", "x2"]
    assert np.isnan(matrix.loc["x1", "x1"])
    assert np.isnan(matrix.loc["x2", "x1"])  # lower triangle masked
    value = matrix.loc["x1", "x2"]
    assert 0.0 <= value <= 1.0
    full = data_inter(
        frame[["y", "x1", "x2"]], response="y", plot=False, lower_tri_off=False, print_info=False
    )
    assert full.loc["x1", "x2"] == full.loc["x2", "x1"]


def test_data_inter_rejects_factors_and_nonlinear() -> None:
    frame = _data()
    with pytest.raises(ValueError, match="only supports numeric covariates"):
        data_inter(frame, response="y", plot=False, print_info=False)
    with pytest.raises(NotImplementedError, match="mgcv"):
        data_inter(frame[["y", "x1", "x2"]], response="y", fit_method="nonlinear", print_info=False)
    with pytest.raises(ValueError, match="only numeric vectors"):
        data_inter(frame[["x1", "x2"]].assign(y=frame["g"].astype(str)), response="y", print_info=False)


@pytest.mark.slow
def test_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.prepdata" / "R" / "data_scale.R").exists():
        pytest.skip("vendored gamlss.prepdata not present")
    frame = _data()
    payload = {
        "y": frame["y"].tolist(),
        "x1": frame["x1"].tolist(),
        "x2": frame["x2"].tolist(),
        "g": frame["g"].astype(str).tolist(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "d.json"
        data_path.write_text(json.dumps(payload), encoding="utf-8")
        script_path = Path(tmp) / "scale.R"
        script_path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path), str(data_path)],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(_REPO),
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    ref = json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())

    def difference(mine, key, shape=None):
        expected = np.asarray([np.nan if value == "NA" else value for value in ref[key]], dtype=float)
        mine = np.asarray(mine, dtype=float)
        if shape is not None:
            expected = expected.reshape(shape, order="F")
        both = np.isnan(mine) & np.isnan(expected)
        return float(
            np.max(np.where(both, 0.0, np.abs(np.nan_to_num(mine) - np.nan_to_num(expected))))
        )

    scaled = data_scale(frame, response="y", scale_to="0to1")
    assert list(scaled.columns) == [str(value) for value in ref["scale_names"]]
    assert difference(scaled["x1"].to_numpy(), "scale_x1") == 0.0
    assert difference(scaled["x2"].to_numpy(), "scale_x2") == 0.0
    assert difference(scaled["y"].to_numpy(), "scale_y") == 0.0
    assert difference(scaled["g"].cat.codes + 1, "scale_g") == 0.0

    main = data_vars2data(frame, response="y", type="main.effect")
    assert list(main.columns) == [str(value) for value in ref["v1_names"]]
    assert difference(main.iloc[:, 1:].to_numpy(), "v1", main.iloc[:, 1:].shape) == 0.0

    first = data_vars2data(frame, response="y", type="first.order")
    assert list(first.columns) == [str(value) for value in ref["v2_names"]]
    assert difference(first.iloc[:, 1:].to_numpy(), "v2", first.iloc[:, 1:].shape) == 0.0

    design = data_form2X(frame, formula="y ~ x1 + g", response="y")
    assert list(design.columns) == [str(value) for value in ref["f1_names"]]
    assert difference(design.to_numpy(), "f1", design.shape) == 0.0

    pairwise = data_form2X(frame, formula="y ~ (x1 + x2)^2", response="y")
    assert list(pairwise.columns) == [str(value) for value in ref["f2_names"]]
    assert difference(pairwise.to_numpy(), "f2", pairwise.shape) == 0.0

    interactions = data_inter(
        frame[["y", "x1", "x2"]], response="y", plot=False, print_info=False
    )
    assert list(interactions.columns) == [str(value) for value in ref["i1_names"]]
    assert difference(interactions.to_numpy(), "i1", interactions.shape) == 0.0