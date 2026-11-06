"""R gamlss.spatial nb2nb() / nb2prec().

The named neighbourhood and the resulting GMRF precision matrix are compared
with the vendored gamlss.spatial sources for a three-region ring and for an
unsorted region.id vector (the reorder branch).
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

from omnilss.spatial import NeighbourNB, nb2nb, nb2prec

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
source("R/gamlss.spatial/R/extraFunctions.R")
library(jsonlite)
nb <- structure(list(c(2L, 3L), c(1L, 3L), c(1L, 2L)), region.id = c("A", "B", "C"), class = "nb")
nb2 <- nb2nb(nb)
G <- nb2prec(nb2, factor(c("A", "B", "C")))
nbU <- structure(list(c(2L), c(1L, 3L), c(2L)), region.id = c("C", "A", "B"), class = "nb")
GU <- nb2prec(nb2nb(nbU), factor(c("A", "B", "C")))
bad <- structure(list(c(2L, 3L), c(3L), c(1L, 2L)), region.id = c("A", "B", "C"), class = "nb")
BAD <- try(nb2prec(nb2nb(bad), factor(c("A", "B", "C"))), silent = TRUE)
out <- list(names = names(nb2$nb), values = lapply(nb2$nb, as.integer),
            G = as.numeric(G), Gdim = as.integer(dim(G)), Gnames = colnames(G),
            GU = as.numeric(GU), GUnames = colnames(GU),
            badMsg = if (inherits(BAD, "try-error")) as.character(BAD) else "")
cat("JSONSTART"); cat(toJSON(out, auto_unbox = FALSE, digits = 17))
"""


def _ring() -> NeighbourNB:
    return NeighbourNB(neighbours=((2, 3), (1, 3), (1, 2)), region_id=("A", "B", "C"))


def _factor() -> pd.Series:
    return pd.Series(pd.Categorical(["A", "B", "C"]))


def test_nb2nb_names_the_positions() -> None:
    named = nb2nb(_ring())
    assert list(named["nb"]) == ["A", "B", "C"]
    assert named["nb"]["A"] == (2, 3)
    assert named["nb"]["C"] == (1, 2)


def test_nb2nb_rejects_objects_without_region_ids() -> None:
    # R checks is(x, "nb"); a bare list carries no region.id, so it is rejected.
    with pytest.raises(ValueError, match="not a nb class"):
        nb2nb([[1], [0]])
    with pytest.raises(ValueError, match="different lengths"):
        NeighbourNB(neighbours=((2,),), region_id=("A", "B"))


def test_nb2prec_builds_the_ring_graph() -> None:
    matrix = nb2prec(nb2nb(_ring()), _factor())
    expected = np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]])
    np.testing.assert_allclose(matrix.to_numpy(), expected, rtol=0, atol=0)
    assert list(matrix.columns) == ["A", "B", "C"]
    assert list(matrix.index) == ["A", "B", "C"]
    np.testing.assert_allclose(matrix.to_numpy().sum(axis=1), 0.0, atol=1e-12)


def test_nb2prec_handles_unsorted_region_ids() -> None:
    unsorted = NeighbourNB(neighbours=((2,), (1, 3), (2,)), region_id=("C", "A", "B"))
    matrix = nb2prec(nb2nb(unsorted), _factor())
    expected = np.array([[2.0, -1.0, -1.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    np.testing.assert_allclose(matrix.to_numpy(), expected, rtol=0, atol=0)
    assert list(matrix.columns) == ["A", "B", "C"]


def test_nb2prec_accepts_an_area_superset() -> None:
    ring = NeighbourNB(
        neighbours=((2, 3, 4), (1, 3), (1, 2, 4), (1, 3)),
        region_id=("A", "B", "C", "D"),
    )
    data = pd.Series(pd.Categorical(["A", "B", "C", "D"]))
    matrix = nb2prec(nb2nb(ring), data, area=["A", "B", "C", "D"])
    assert matrix.shape == (4, 4)
    np.testing.assert_allclose(matrix.to_numpy().sum(axis=1), 0.0, atol=1e-12)


def test_nb2prec_validates_its_inputs() -> None:
    named = nb2nb(_ring())
    with pytest.raises(ValueError, match="x must be a factor"):
        nb2prec(named, pd.Series([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="MRF basis dimension set too high"):
        nb2prec(named, _factor(), area=["A", "B"])
    with pytest.raises(ValueError, match="not contained in the area specification"):
        nb2prec(named, _factor(), area=["A", "B", "Z"])
    with pytest.raises(ValueError, match="mismatch between neighbour"):
        nb2prec(named, pd.Series(pd.Categorical(["A", "B", "Z"]))),


def test_nb2prec_rejects_an_asymmetric_neighbourhood() -> None:
    broken = NeighbourNB(neighbours=((2, 3), (3,), (1, 2)), region_id=("A", "B", "C"))
    with pytest.raises(ValueError, match="Something wrong with auto- penalty construction"):
        nb2prec(nb2nb(broken), _factor())


@pytest.mark.slow
def test_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.spatial" / "R" / "extraFunctions.R").exists():
        pytest.skip("vendored gamlss.spatial not present")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nb.R"
        path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(path)], capture_output=True, text=True, timeout=300, cwd=str(_REPO)
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    ref = json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())

    named = nb2nb(_ring())
    assert list(named["nb"]) == [str(value) for value in ref["names"]]
    for name, values in named["nb"].items():
        assert list(values) == [int(value) for value in np.asarray(ref["values"][name]).ravel()]

    matrix = nb2prec(named, _factor())
    expected = np.asarray(ref["G"], dtype=float).reshape(
        tuple(int(value) for value in ref["Gdim"]), order="F"
    )
    np.testing.assert_allclose(matrix.to_numpy(), expected, rtol=0, atol=0)
    assert list(matrix.columns) == [str(value) for value in ref["Gnames"]]

    unsorted = NeighbourNB(neighbours=((2,), (1, 3), (2,)), region_id=("C", "A", "B"))
    unsorted_matrix = nb2prec(nb2nb(unsorted), _factor())
    expected_unsorted = np.asarray(ref["GU"], dtype=float).reshape((3, 3), order="F")
    np.testing.assert_allclose(unsorted_matrix.to_numpy(), expected_unsorted, rtol=0, atol=0)
    assert list(unsorted_matrix.columns) == [str(value) for value in ref["GUnames"]]

    message = str(np.asarray(ref["badMsg"]).ravel()[0])
    assert "Something wrong with auto- penalty construction" in message