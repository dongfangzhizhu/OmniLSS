# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.add regression tree: tr() / gamlss_tr (D2.249).

Live rpart 4.1.2 reference on the identical sample (seed 1, n=200,
x ~ U(0,10), y = 1 + 2 * I(x >= 4) + N(0, 0.3), cp=0.01, minsplit=20,
minbucket=7, xval=0):

    leaves 2, split 3.9894619305, leaf means 0.98361426 / 2.96683439

The port reproduces rpart's weighted-SSE criterion (improvement relative to the
root deviance, kept when >= cp).  Surrogates and cross-validation are not
ported and are rejected or ignored explicitly.
"""
from __future__ import annotations

import numpy as np
import pytest

from omnilss import gamlss, gamlss_tr, tr, tree_control
from omnilss.additive import RegressionTree, TreeControl

R_SPLIT = 3.9894619305
R_LEAF_MEANS = (0.98361426, 2.96683439)


def _sample():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.uniform(0, 10, size=n)
    y = np.where(x < 4, 1.0, 3.0) + rng.normal(scale=0.3, size=n)
    return x, y


def _leaf_means(node):
    values = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current.is_leaf:
            values.append(current.value)
        else:
            stack.extend([current.left, current.right])
    return sorted(values)


def test_tree_control_uses_rpart_defaults():
    control = tree_control()
    assert control.cp == 0.01
    assert control.minsplit == 20
    assert control.minbucket == 7  # round(20 / 3)
    assert control.maxdepth == 30
    assert control.xval == 10
    assert tree_control(minsplit=30).minbucket == 10
    with pytest.raises(TypeError, match="unknown"):
        tree_control(not_a_parameter=1)


def test_tree_matches_live_rpart_on_the_same_sample():
    x, y = _sample()
    tree = RegressionTree(
        TreeControl(cp=0.01, minsplit=20, minbucket=7, maxdepth=30, xval=0)
    ).fit({"x": x}, y)
    assert tree.leaf_count == 2
    assert tree.root.variable == "x"
    assert tree.root.threshold == pytest.approx(R_SPLIT, abs=1e-9)
    np.testing.assert_allclose(_leaf_means(tree.root), R_LEAF_MEANS, atol=1e-8)


def test_gamlss_tr_returns_the_fields_gamlss_consumes():
    x, y = _sample()
    fit = gamlss_tr(tr("~ x", cp=0.01, minsplit=20, minbucket=7, xval=0), y, None, data={"x": x})
    assert fit.fitted_values.shape == y.shape
    assert fit.residuals.shape == y.shape
    assert float(np.max(np.abs(fit.residuals.sum()))) < 1e-9
    assert fit.nl_df == 2  # 2 leaves -> 2 * 2 - 2
    assert fit.var is None
    np.testing.assert_allclose(fit.fitted_values, fit.coef_smo.predict({"x": x}), rtol=0, atol=0)


def test_tree_respects_weights():
    x, y = _sample()
    weights = np.ones_like(y)
    weights[x < 4] = 3.0
    fit = gamlss_tr(tr("~ x"), y, weights, data={"x": x})
    mask = x <= fit.coef_smo.root.threshold
    expected_left = float(np.sum(weights[mask] * y[mask]) / np.sum(weights[mask]))
    assert fit.coef_smo.root.left.value == pytest.approx(expected_left, rel=1e-12)


def test_maxdepth_zero_yields_a_single_leaf():
    x, y = _sample()
    tree = RegressionTree(TreeControl(maxdepth=0)).fit({"x": x}, y)
    assert tree.leaf_count == 1
    assert tree.root.value == pytest.approx(float(np.mean(y)))


def test_missing_or_non_numeric_predictors_are_rejected():
    x, y = _sample()
    bad = np.array(x)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        RegressionTree().fit({"x": bad}, y)
    labels = np.where(x < 4, "a", "b")
    with pytest.raises(NotImplementedError, match="not numeric"):
        RegressionTree().fit({"x": labels}, y)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"formula": "y ~ x"}, "starting with ~"),
        ({"formula": None}, "starting with ~"),
        ({"formula": "~ x", "method": "tree"}, "rpart"),
    ],
)
def test_tr_validates_its_arguments(kwargs, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        tr(**kwargs)


def test_tree_terms_are_refused_inside_the_joint_fitter():
    x, y = _sample()
    with pytest.raises(NotImplementedError, match="gamlss_tr"):
        gamlss("y ~ tr(~ x)", family="NO", data={"y": y, "x": x}, max_iter=5)


def test_gamlss_tr_requires_a_term_and_data():
    x, y = _sample()
    with pytest.raises(TypeError, match="tr\\(\\) term"):
        gamlss_tr("~ x", y, None, data={"x": x})
    with pytest.raises(ValueError, match="data"):
        gamlss_tr(tr("~ x"), y, None)
