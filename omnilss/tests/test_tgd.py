"""R TGD(): the test-global-deviance accessor for gamlssTGD objects.

R stores TGD inside the object returned by getTGD(); the Python analogue is
TGDResult.  The accessor contract (single value, sorted comparison table with
the Pred.GD column, error text) is compared with live R on hand-built objects,
because R's getTGD() cannot run head-less (it expects the gamlss call frame).
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
from omnilss.gamlssVGD_23_12_21 import TGD, TGDComparisonResult, TGDResult

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
mk <- function(v) structure(list(TGD = v, predictError = v / 10, resid = c(1, 2)), class = "gamlssTGD")
a <- mk(12.5); b <- mk(3.25); cc <- mk(7.0)
cmp <- TGD(a, b, cc)
bad <- try(TGD(list(TGD = 1)), silent = TRUE)
cat("JSONSTART"); cat(toJSON(list(
  single = TGD(a), cmp = as.numeric(cmp$Pred.GD), cmpRows = rownames(cmp),
  cmpCols = colnames(cmp),
  badMsg = if (inherits(bad, "try-error")) as.character(bad) else ""),
  auto_unbox = FALSE, digits = 17))
"""


def _result(value: float, family: str = "NO") -> TGDResult:
    return TGDResult(
        family=family,
        test_global_deviance=float(value),
        prediction_error=float(value) / 10.0,
        deviance_increment=np.zeros(1),
        residuals=None,
        validation_size=2,
    )


def test_single_object_returns_the_deviance() -> None:
    assert TGD(_result(12.5)) == 12.5


def test_model_with_a_tgd_slot_is_accepted() -> None:
    class Carries:
        additional_slots = {"TGD": 4.5}

    assert TGD(Carries()) == 4.5


def test_comparison_sorts_and_labels_the_family() -> None:
    result = TGD(_result(12.5, "A"), _result(3.25, "B"), _result(7.0, "C"))
    assert isinstance(result, TGDComparisonResult)
    assert [round(row.pred_gd, 6) for row in result.rows] == [3.25, 7.0, 12.5]
    frame = result.to_frame()
    assert list(frame.columns) == ["Pred.GD"]
    assert list(frame.index) == ["B", "C", "A"]
    np.testing.assert_allclose(frame["Pred.GD"].to_numpy(), [3.25, 7.0, 12.5])


def test_non_tgd_objects_are_rejected() -> None:
    with pytest.raises(ValueError, match="this is not a TGD object"):
        TGD(object())
    with pytest.raises(ValueError, match="some of the objects are not TGD"):
        TGD(_result(1.0), object())
    with pytest.raises(ValueError, match="at least one object"):
        TGD()


def test_accessor_reads_get_tgd_output() -> None:
    rs = np.random.RandomState(0)
    n = 100
    x = np.linspace(0.0, 1.0, n)
    y = 1.0 + 2.0 * x + rs.normal(0.0, 0.3, n)
    data = {"y": y, "x": x}
    model = omnilss.gamlss("y ~ x", family="NO", data=data)
    result = omnilss.get_tgd(model, newdata=data)
    assert TGD(result) == pytest.approx(result.test_global_deviance, rel=1e-12)


@pytest.mark.slow
def test_accessor_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss" / "R" / "gamlssVGD_23_12_21.R").exists():
        pytest.skip("vendored gamlssVGD source not present")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tgd.R"
        path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(path)], capture_output=True, text=True, timeout=300, cwd=str(_REPO)
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    ref = json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())

    assert TGD(_result(12.5)) == pytest.approx(float(np.asarray(ref["single"]).ravel()[0]))

    comparison = TGD(_result(12.5, "A"), _result(3.25, "B"), _result(7.0, "C"))
    np.testing.assert_allclose(
        [row.pred_gd for row in comparison.rows],
        np.asarray(ref["cmp"], dtype=float).ravel(),
        rtol=0,
        atol=0,
    )
    assert list(comparison.to_frame().columns) == [str(value) for value in np.asarray(ref["cmpCols"]).ravel()]
    assert "this is not a TGD object" in str(np.asarray(ref["badMsg"]).ravel()[0])