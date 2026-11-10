"""R gamlss.spatial::polys2nb -- live-R parity and end-to-end neighbourhood.

R source: ``R/gamlss.spatial/R/extraFunctions.R``.  The oracle sources the real
vendored file (mgcv is installed, so ``uniquecombs`` is the genuine function)
and calls ``polys2nb`` on several polygon sets.  ``sp`` (S4 SpatialPolygons) is
NOT installed, so ``polys2polys`` / ``draw.polys`` are out of scope for this
batch -- ``polys2nb`` is the pure-structure core and is verified exactly.

R semantics (all numeric/lists verified exactly):
* neighbours are defined by **shared vertices** only (a vertex on another
  polygon's edge segment is NOT -- documented warning in the R source);
* NA rows are dropped (``!is.na(rowSums(poly))``), then vertices de-duplicated
  with ``uniquecombs``; a four-clause bbox overlap prefilter per axis cuts the
  O(n^2) vertex comparisons;
* shared-vertex test: ``nrow(rbind(a,b)) - nrow(uniquecombs(rbind(a,b))) > 0``;
* neighbour indices are **1-based** (R convention), ``unique()`` keeps order;
* returns ``xlim=(min(lo1),max(hi1))`` / ``ylim=(min(lo2),max(hi2))``.

End-to-end: ``polys2nb -> NeighbourNB -> nb2nb -> nb2prec`` reproduces R's
precision matrix G (diagonal = neighbour count, -1 per neighbour pair).
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

from omnilss.spatial import (
    NeighbourNB,
    PolyNB,
    nb2nb,
    nb2prec,
    polys2nb,
)

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = r"""# mgcv installed -> source REAL vendored extraFunctions.R (uniquecombs is real)
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(mgcv))
txt <- readLines('R/gamlss.spatial/R/extraFunctions.R')
eval(parse(text=paste(txt, collapse='\n')), envir=globalenv())
M <- function(...) matrix(c(...), ncol=2, byrow=TRUE)
sq <- function(x0,y0,x1,y1) M(x0,y0, x1,y0, x1,y1, x0,y1)
P1 <- list(A=sq(0,0,1,1), B=sq(1,0,2,1), C=sq(5,5,6,6))   # shared edge, isolated
P2 <- list(A=sq(0,0,1,1), B=sq(1,1,2,2), C=sq(2,2,3,3))   # shared single vertex
P3 <- list(A=M(0,0, 1,0, 1,1, 0,1, 1,0, 0,0), B=sq(1,0,2,1))  # duplicate vertices
P4 <- list(A=M(0,0, 1,0, NA,NA, 1,1, 0,1), B=sq(1,0,2,1))     # NA row
P5 <- list(A=sq(0,0,1,1), B=sq(2,0,3,1), C=sq(1,0,2,1))   # chain A-C-B
out <- list(p1=polys2nb(P1), p2=polys2nb(P2), p3=polys2nb(P3),
            p4=polys2nb(P4), p5=polys2nb(P5))
for (nm in names(out)) out[[nm]]$nb <- lapply(out[[nm]]$nb, as.numeric)
# 4x4 grid of unit squares, shuffled
cells <- list(); nm <- c()
for (i in 0:3) for (j in 0:3) {
  cells[[length(cells)+1]] <- M(i,j, i+1,j, i+1,j+1, i,j+1)
  nm <- c(nm, paste0('r',i,'_',j))
}
names(cells) <- nm
set.seed(5); ord <- sample(length(cells)); cells <- cells[ord]
r <- polys2nb(cells); r$nb <- lapply(r$nb, as.numeric)
out$grid <- list(nb=r$nb, xlim=r$xlim, ylim=r$ylim, names=nm[ord])
# end-to-end G for P1
p <- P1; rr <- polys2nb(p)
class(rr$nb) <- 'nb'; attr(rr$nb,'region.id') <- names(p)
G <- nb2prec(nb2nb(rr$nb), factor(c('A','B','C'), levels=c('A','B','C')))
out$G <- as.matrix(G)
cat("JSONSTART"); cat(toJSON(out, auto_unbox=FALSE, digits=17))
"""

_M = lambda *v: np.asarray(v, dtype=float).reshape(-1, 2)
_SQ = lambda a, b, c, d: _M(a, b, c, b, c, d, a, d)

_P1 = {"A": _SQ(0, 0, 1, 1), "B": _SQ(1, 0, 2, 1), "C": _SQ(5, 5, 6, 6)}
_P2 = {"A": _SQ(0, 0, 1, 1), "B": _SQ(1, 1, 2, 2), "C": _SQ(2, 2, 3, 3)}
_P3 = {"A": _M(0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0), "B": _SQ(1, 0, 2, 1)}
_P4 = {"A": _M(0, 0, 1, 0, np.nan, np.nan, 1, 1, 0, 1), "B": _SQ(1, 0, 2, 1)}
_P5 = {"A": _SQ(0, 0, 1, 1), "B": _SQ(2, 0, 3, 1), "C": _SQ(1, 0, 2, 1)}


@pytest.fixture(scope="module")
def r_reference():
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.spatial" / "R" / "extraFunctions.R").exists():
        pytest.skip("vendored gamlss.spatial source not present")
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "polys.R"
        script_path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path)], capture_output=True, text=True,
            timeout=900, cwd=str(_REPO),
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-500:]}")
    return json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())


def _assert_poly(ours: PolyNB, ref: dict, key: str) -> None:
    ref_nb = [list(v) for v in ref["nb"].values()]
    assert [list(v) for v in ours.neighbours] == ref_nb, key + ":nb"
    assert ours.xlim == tuple(ref["xlim"]), key + ":xlim"
    assert ours.ylim == tuple(ref["ylim"]), key + ":ylim"


def test_all_polygon_sets_match_live_r(r_reference):
    cases = {"p1": _P1, "p2": _P2, "p3": _P3, "p4": _P4, "p5": _P5}
    for key, polys in cases.items():
        _assert_poly(polys2nb(polys), r_reference[key], key)


def test_shared_vertex_only_and_bbox_prune(r_reference):
    # p2: A and C touch B at one vertex each but NOT each other -> A has [2],
    # B [1,3], C [2]; p5 chain A-C-B -> A and B are NOT neighbours.
    assert list(polys2nb(_P2).neighbours) == [(2,), (1, 3), (2,)]
    assert list(polys2nb(_P5).neighbours) == [(3,), (3,), (1, 2)]
    # p1: C is isolated (bbox far away)
    assert list(polys2nb(_P1).neighbours)[2] == ()


def test_grid_matches_live_r(r_reference):
    ref = r_reference["grid"]
    polys = {}
    for nm in ref["names"]:
        i, j = map(int, nm[1:].split("_"))
        polys[nm] = _M(i, j, i + 1, j, i + 1, j + 1, i, j + 1)
    ours = polys2nb(polys)
    assert [list(v) for v in ours.neighbours] == [list(v) for v in ref["nb"].values()]
    assert ours.xlim == tuple(ref["xlim"])
    assert ours.ylim == tuple(ref["ylim"])


def test_end_to_end_precision_matrix(r_reference):
    r = polys2nb(_P1)
    nb = NeighbourNB(neighbours=r.neighbours, region_id=r.region_id)
    named = nb2nb(nb)
    x = pd.Series(pd.Categorical(["A", "B", "C"], categories=["A", "B", "C"]))
    G = nb2prec(named, x)
    expected = np.asarray(r_reference["G"], float)
    np.testing.assert_allclose(G.to_numpy(), expected, rtol=0, atol=0)
    assert list(G.index) == ["A", "B", "C"]


def test_region_names_and_sequence_input():
    r = polys2nb(_P1)
    assert isinstance(r, PolyNB)
    assert r.region_id == ("A", "B", "C")
    # sequence input -> positional 1-based names
    seq = polys2nb([_P1["A"], _P1["B"]])
    assert seq.region_id == ("1", "2")
    assert list(seq.neighbours) == [(2,), (1,)]


def test_na_and_duplicate_vertices():
    assert list(polys2nb(_P4).neighbours) == [(2,), (1,)]
    assert list(polys2nb(_P3).neighbours) == [(2,), (1,)]


def test_bad_shape():
    with pytest.raises(ValueError, match="Nx2"):
        polys2nb({"A": np.arange(6.0)})
