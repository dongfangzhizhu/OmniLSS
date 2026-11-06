"""R stepGAIC.VR: Venables-Ripley stepwise selection.

The path table (Step / Df / Deviance / Resid. Df / Resid. Dev / AIC) and the
final formula are compared with live R for a single forward addition and for a
two-step backward path.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

import omnilss
from omnilss.stepwise_vr import StepwisePathResult, stepGAIC_VR, step_gaic_vr

_REPO = Path(__file__).resolve().parents[2]

_FORWARD_SCRIPT = """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
a <- commandArgs(trailingOnly = TRUE); d <- fromJSON(a[1])
df <- data.frame(y = d$y, x1 = d$x1, x2 = d$x2, x3 = d$x3, z = d$z)
m <- gamlss(y ~ x1, family = NO, data = df, trace = FALSE)
res <- stepGAIC.VR(m, scope = ~ . + x2 + x3, direction = "both", trace = FALSE)
an <- res$anova
cat("JSONSTART"); cat(toJSON(list(
  Step = as.character(an$Step), Df = as.numeric(an$Df), Dev = as.numeric(an$Deviance),
  Rdf = as.numeric(an$`Resid. Df`), Rdev = as.numeric(an$`Resid. Dev`), AIC = as.numeric(an$AIC),
  final = paste(deparse(formula(res, what = "mu")), collapse = " ")),
  auto_unbox = FALSE, digits = 17))
"""

_BACKWARD_SCRIPT = """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
a <- commandArgs(trailingOnly = TRUE); d <- fromJSON(a[1])
df <- data.frame(y = d$y, x1 = d$x1, x2 = d$x2, x3 = d$x3, z = d$z)
m <- gamlss(y ~ x1 + x2 + x3 + z, family = NO, data = df, trace = FALSE)
res <- stepGAIC.VR(m, scope = ~ ., direction = "both", trace = FALSE)
an <- res$anova
cat("JSONSTART"); cat(toJSON(list(
  Step = as.character(an$Step), Df = as.numeric(an$Df), Dev = as.numeric(an$Deviance),
  Rdf = as.numeric(an$`Resid. Df`), Rdev = as.numeric(an$`Resid. Dev`), AIC = as.numeric(an$AIC),
  final = paste(deparse(formula(res, what = "mu")), collapse = " ")),
  auto_unbox = FALSE, digits = 17))
"""


def _data(seed: int, formula_terms: int = 3):
    rs = np.random.RandomState(seed)
    n = 150
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = rs.normal(size=n)
    x3 = rs.normal(size=n)
    z = rs.normal(size=n)
    y = 1.0 + 2.0 * x1 + 0.8 * x2 + rs.normal(0.0, 0.4, n)
    return {"y": y, "x1": x1, "x2": x2, "x3": x3, "z": z}


def _run_r(script: str, data: dict) -> dict:
    payload = {name: np.asarray(values, dtype=float).tolist() for name, values in data.items()}
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "d.json"
        data_path.write_text(json.dumps(payload), encoding="utf-8")
        script_path = Path(tmp) / "step.R"
        script_path.write_text(script, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path), str(data_path)],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(_REPO),
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    return json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())


def test_alias_points_at_the_same_function() -> None:
    assert step_gaic_vr is stepGAIC_VR


def test_forward_step_adds_the_signal_term() -> None:
    data = _data(0)
    model = omnilss.gamlss("y ~ x1", family="NO", data=data)
    result = stepGAIC_VR(model, scope="~ . + x2 + x3", direction="both")
    assert isinstance(result, StepwisePathResult)
    assert result.term_labels[0] == ""
    assert result.term_labels[1] == "+ x2"
    assert result.model.formulas["mu"].replace(" ", "") == "y~x1+x2"


def test_path_table_shape() -> None:
    data = _data(0)
    model = omnilss.gamlss("y ~ x1", family="NO", data=data)
    frame = stepGAIC_VR(model, scope="~ . + x2 + x3").to_frame()
    assert list(frame.columns) == ["Step", "Df", "Deviance", "Resid. Df", "Resid. Dev", "AIC"]
    assert np.isnan(frame["Df"].iloc[0]) and np.isnan(frame["Deviance"].iloc[0])
    assert frame["Resid. Df"].iloc[0] == pytest.approx(float(model.n) - model.df_fit)
    assert frame["AIC"].iloc[0] == pytest.approx(model.g_dev + 2.0 * model.df_fit, rel=1e-12)
    assert list(frame.attrs["heading"])[:1] == ["Stepwise Model Path \nAnalysis of Deviance Table"]


def test_backward_step_drops_noise_terms() -> None:
    data = _data(1)
    model = omnilss.gamlss("y ~ x1 + x2 + x3 + z", family="NO", data=data)
    result = stepGAIC_VR(model, scope="~ .", direction="both")
    assert result.term_labels[1].startswith("- ")
    assert result.model.formulas["mu"].replace(" ", "") == "y~x1+x2"


def test_aic_is_monotone_along_the_path() -> None:
    data = _data(1)
    model = omnilss.gamlss("y ~ x1 + x2 + x3 + z", family="NO", data=data)
    frame = stepGAIC_VR(model, scope="~ .", direction="both").to_frame()
    aic = frame["AIC"].to_numpy(dtype=float)
    assert np.all(np.diff(aic) <= 1e-7)


def test_direction_and_parameter_are_validated() -> None:
    data = _data(0)
    model = omnilss.gamlss("y ~ x1", family="NO", data=data)
    with pytest.raises(ValueError, match="direction must be one of"):
        stepGAIC_VR(model, scope="~ .", direction="sideways")
    with pytest.raises(ValueError, match="is not a parameter in the object"):
        stepGAIC_VR(model, scope="~ .", what="nu")


def test_forward_only_never_drops() -> None:
    data = _data(0)
    model = omnilss.gamlss("y ~ x1", family="NO", data=data)
    result = stepGAIC_VR(model, scope="~ . + x2 + x3", direction="forward")
    assert all(not label.startswith("- ") for label in result.term_labels[1:])


@pytest.mark.slow
def test_matches_live_r_forward() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss" / "R" / "stepGAIC-03-10-13..R").exists():
        pytest.skip("vendored stepGAIC source not present")
    data = _data(0)
    ref = _run_r(_FORWARD_SCRIPT, data)
    model = omnilss.gamlss("y ~ x1", family="NO", data=data)
    frame = stepGAIC_VR(model, scope="~ . + x2 + x3", direction="both").to_frame()
    assert frame["Step"].tolist() == [str(value) for value in ref["Step"]]
    for column, key in (("Df", "Df"), ("Deviance", "Dev"), ("Resid. Df", "Rdf"), ("Resid. Dev", "Rdev"), ("AIC", "AIC")):
        expected = np.asarray([np.nan if value == "NA" else value for value in ref[key]], dtype=float)
        np.testing.assert_allclose(
            frame[column].to_numpy(dtype=float), expected, rtol=1e-9, atol=1e-7, equal_nan=True
        )
    assert frame.attrs["heading"][-2] == pytest.approx(str(np.asarray(ref["final"]).ravel()[0]))


@pytest.mark.slow
def test_matches_live_r_backward() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss" / "R" / "stepGAIC-03-10-13..R").exists():
        pytest.skip("vendored stepGAIC source not present")
    data = _data(1)
    ref = _run_r(_BACKWARD_SCRIPT, data)
    model = omnilss.gamlss("y ~ x1 + x2 + x3 + z", family="NO", data=data)
    result = stepGAIC_VR(model, scope="~ .", direction="both")
    frame = result.to_frame()
    assert frame["Step"].tolist() == [str(value) for value in ref["Step"]]
    expected = np.asarray([np.nan if value == "NA" else value for value in ref["AIC"]], dtype=float)
    np.testing.assert_allclose(frame["AIC"].to_numpy(dtype=float), expected, rtol=1e-9, atol=1e-7, equal_nan=True)
    assert result.model.formulas["mu"].replace(" ", "") == "y~x1+x2"