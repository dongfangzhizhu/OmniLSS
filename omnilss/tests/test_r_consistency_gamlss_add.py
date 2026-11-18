"""Live-R consistency evidence for gamlss.add (D2.265).

Covers the two numerically comparable groups:

* tr()/gamlss_tr() against rpart's anova tree (same sample, cp/minsplit/minbucket);
* fitFixedKnots()/fitFreeKnots() against the vendored freeKnots source.

The delegation-based additive wrappers and the network term are deliberately not
compared to R: their R references are separate estimators (mgcv and nnet), so
their contract lives in test_gamlss_add.py instead.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss import RegressionTree, fitFixedKnots, fitFreeKnots, gamlss_tr, tr, tree_control

_REPO = Path(__file__).resolve().parents[2]
_R_SCRIPT = """
suppressPackageStartupMessages(library(rpart))
suppressPackageStartupMessages(library(gamlss.dist))
suppressPackageStartupMessages(library(jsonlite))
source("R/gamlss.add/R/freeKnots-18-06-13.R")
args <- commandArgs(trailingOnly = TRUE)
d <- fromJSON(args[1])
ft <- rpart(y ~ x, data = data.frame(x = d$tree_x, y = d$tree_y), method = "anova",
            control = rpart.control(cp = 0.01, minsplit = 20, minbucket = 7, xval = 0))
leaves <- ft$frame[ft$frame$var == "<leaf>", "yval"]
splits <- ft$splits
split <- splits[splits[, "count"] > 0, "index"]
fixed <- fitFixedKnots(y = d$fk_y, x = d$fk_x, knots = c(3, 7), degree = 1)
free1 <- fitFreeKnots(y = d$fk_y, x = d$fk_x, knots = c(5), degree = 1)
out <- list(
  tree = list(leaves = sort(as.numeric(leaves)), split = as.numeric(split)),
  fixed = list(rss = fixed$rss, coef = as.numeric(fixed$coef)),
  free1 = list(bp = as.numeric(free1$breakPoints), rss = free1$rss)
)
cat(toJSON(out, digits = 17, auto_unbox = TRUE))
"""


def _payload() -> dict:
    rng = np.random.default_rng(1)
    n = 200
    tree_x = rng.uniform(0.0, 10.0, size=n)
    tree_y = np.where(tree_x < 4.0, 1.0, 3.0) + rng.normal(scale=0.3, size=n)

    rng2 = np.random.default_rng(3)
    fk_x = np.sort(rng2.uniform(0.0, 10.0, size=100))
    fk_y = np.sin(fk_x) + rng2.normal(scale=0.2, size=100)
    return {
        "tree_x": tree_x.tolist(),
        "tree_y": tree_y.tolist(),
        "fk_x": fk_x.tolist(),
        "fk_y": fk_y.tolist(),
    }


def _live_r(payload: dict) -> dict:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.add" / "R" / "freeKnots-18-06-13.R").exists():
        pytest.skip("vendored gamlss.add source not present")
    with tempfile.TemporaryDirectory() as tmp:
        data_file = Path(tmp) / "add.json"
        data_file.write_text(json.dumps(payload), encoding="utf-8")
        script = Path(tmp) / "ref.R"
        script.write_text(_R_SCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script), str(data_file)],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(_REPO),
        )
    if proc.returncode != 0:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def reference() -> dict:
    return _live_r(_payload())


@pytest.mark.slow
def test_tree_matches_live_rpart(reference):
    payload = _payload()
    x = np.asarray(payload["tree_x"], dtype=float)
    y = np.asarray(payload["tree_y"], dtype=float)
    tree = RegressionTree(
        tree_control(cp=0.01, minsplit=20, minbucket=7, xval=0)
    ).fit({"x": x}, y)
    expected = reference["tree"]
    assert tree.leaf_count == len(expected["leaves"])
    split = expected["split"]
    split = float(split[0]) if isinstance(split, list) else float(split)
    assert tree.root.threshold == pytest.approx(split, abs=1e-9)
    leaves = []
    stack = [tree.root]
    while stack:
        node = stack.pop()
        if node.is_leaf:
            leaves.append(node.value)
        else:
            stack.extend([node.left, node.right])
    np.testing.assert_allclose(sorted(leaves), np.asarray(expected["leaves"], dtype=float), atol=1e-8)

    fitted = gamlss_tr(tr("~ x"), y, None, data={"x": x})
    np.testing.assert_allclose(fitted.fitted_values, tree.predict({"x": x}), rtol=0, atol=0)


@pytest.mark.slow
def test_fixed_and_one_free_knot_match_live_r(reference):
    payload = _payload()
    x = np.asarray(payload["fk_x"], dtype=float)
    y = np.asarray(payload["fk_y"], dtype=float)

    fixed = fitFixedKnots(y=y, x=x, knots=[3.0, 7.0], degree=1)
    assert fixed.rss == pytest.approx(float(reference["fixed"]["rss"]), rel=1e-12)
    np.testing.assert_allclose(
        fixed.coef, np.asarray(reference["fixed"]["coef"], dtype=float), rtol=1e-11
    )

    free = fitFreeKnots(y=y, x=x, knots=[5.0], degree=1)
    # auto_unbox turns the single break point into a scalar.
    reference_bp = reference["free1"]["bp"]
    reference_bp = float(reference_bp[0]) if isinstance(reference_bp, list) else float(reference_bp)
    assert float(free.break_points[0]) == pytest.approx(reference_bp, abs=1e-8)
    assert free.rss == pytest.approx(float(reference["free1"]["rss"]), rel=1e-11)
