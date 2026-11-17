# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.add additive terms: ga() / ba() and their fitting callbacks (D2.248).

R's ga()/ba() are *term constructors*: they run mgcv::gam()/bam() with fit=FALSE
to obtain a basis, then gamlss.ga()/gamlss.ba() refit that basis for the working
response z with weights w.  The port keeps that interface and the same object
shape, but delegates the basis and the penalty to OmniLSS's existing smooth
machinery:

    gamlss(y ~ ga(~ pb(x)))  is exactly  gamlss(y ~ pb(x))

That equality is the contract this module pins.  It is deliberately *not* a
claim of numerical parity with mgcv's REML smoothing-parameter selection: mgcv
is a separate estimator with its own penalty and is not reimplemented here.
Calling the callback directly (gamlss_ga / gamlss_ba) performs the weighted
least-squares refit of the stored basis, which is the part of R's callback that
does not depend on mgcv.

R source: R/gamlss.add/R/ga_new.R (ga) and R/gamlss.add/R/ba.R (ba, which uses
mgcv::bam -- "big additive models", not backfitting).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .add_controls import ga_control
from .formula_parser import ParsedFormula, build_design_matrix, parse_formula

_TERM_RESPONSE = "__additive_response__"


@dataclass(frozen=True)
class AdditiveTerm:
    """R's ga()/ba() term: an inner formula plus its control list."""

    kind: str
    formula: str
    parsed: ParsedFormula
    control: Mapping[str, Any]


@dataclass(frozen=True)
class AdditiveFit:
    """Result of a gamlss.ga / gamlss.ba callback on a working response."""

    kind: str
    formula: str
    fitted: np.ndarray
    coefficients: np.ndarray
    design: np.ndarray
    smooth_info: Mapping[str, Any]
    iterations: int = 1


def _term(kind: str, formula: Any, control: Any, kwargs: Mapping[str, Any]) -> AdditiveTerm:
    if not isinstance(formula, str) or not formula.strip().startswith("~"):
        raise ValueError(
            f"formula argument in {kind}() needs a formula starting with ~"
        )
    if control is None:
        settings = ga_control(**dict(kwargs))
    elif isinstance(control, Mapping):
        settings = dict(control)
    else:
        settings = control
    parsed = parse_formula(f"{_TERM_RESPONSE} {formula.strip()}")
    return AdditiveTerm(kind=kind, formula=formula.strip(), parsed=parsed, control=settings)


def ga(formula: Any = None, control: Any = None, **kwargs: Any) -> AdditiveTerm:
    """R ga(): declare an additive term fitted with gam-style splines."""

    return _term("ga", formula, control, kwargs)


def ba(formula: Any = None, control: Any = None, **kwargs: Any) -> AdditiveTerm:
    """R ba(): declare an additive term fitted with bam-style splines.

    mgcv's bam is a large-data variant of gam, not a backfitting loop; the port
    therefore shares ga's design and fit path and only records the different
    kind.
    """

    return _term("ba", formula, control, kwargs)


def _fit_additive(
    kind: str,
    data: Mapping[str, Any],
    z: Any,
    w: Any,
    formula: Any,
) -> AdditiveFit:
    if data is None:
        raise ValueError("the data argument is needed")
    if not isinstance(formula, str) or not formula.strip().startswith("~"):
        raise ValueError(
            f"formula argument in {kind}() needs a formula starting with ~"
        )
    working = np.asarray(z, dtype=np.float64).ravel()
    weights = np.ones(working.size) if w is None else np.asarray(w, dtype=np.float64).ravel()
    if weights.size != working.size:
        raise ValueError("z and w must have the same length")
    parsed = parse_formula(f"{_TERM_RESPONSE} {formula.strip()}")
    augmented = dict(data)
    augmented[_TERM_RESPONSE] = working
    design, smooth_info = build_design_matrix(parsed, augmented, fit_smooths=True)
    design = np.asarray(design, dtype=np.float64)
    root_w = np.sqrt(weights)
    coefficients, *_ = np.linalg.lstsq(design * root_w[:, None], working * root_w, rcond=None)
    fitted = design @ coefficients
    return AdditiveFit(
        kind=kind,
        formula=formula.strip(),
        fitted=fitted,
        coefficients=np.asarray(coefficients, dtype=np.float64),
        design=design,
        smooth_info=smooth_info,
        iterations=1,
    )


def gamlss_ga(
    data: Mapping[str, Any] = None,
    z: Any = None,
    w: Any = None,
    formula: Any = None,
    control: Any = None,
    **kwargs: Any,
) -> AdditiveFit:
    """R gamlss.ga: weighted refit of a ga() basis for the working response z."""

    return _fit_additive("ga", data, z, w, formula)


def gamlss_ba(
    data: Mapping[str, Any] = None,
    z: Any = None,
    w: Any = None,
    formula: Any = None,
    control: Any = None,
    **kwargs: Any,
) -> AdditiveFit:
    """R gamlss.ba: weighted refit of a ba() basis for the working response z."""

    return _fit_additive("ba", data, z, w, formula)




# ---------------------------------------------------------------------------
# tr / gamlss.tr: rpart-style regression tree term (D2.249)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreeControl:
    """rpart.control defaults (rpart 4.1.2; live R checked)."""

    cp: float = 0.01
    minsplit: int = 20
    minbucket: int | None = None
    maxcompete: int = 4
    maxsurrogate: int = 5
    usesurrogate: int = 2
    surrogatestyle: int = 0
    maxdepth: int = 30
    xval: int = 10


def tree_control(**kwargs: Any) -> TreeControl:
    """Build a TreeControl, applying rpart's minbucket = round(minsplit/3)."""

    values = dict(kwargs)
    minsplit = int(values.get("minsplit", 20))
    minbucket = values.get("minbucket")
    if minbucket is None:
        minbucket = int(round(minsplit / 3.0))
    values["minsplit"] = minsplit
    values["minbucket"] = int(minbucket)
    allowed = set(TreeControl.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise TypeError(f"unknown tree control parameter(s): {unknown}")
    return TreeControl(**values)


@dataclass(frozen=True)
class TreeNode:
    value: float
    n: int
    deviance: float
    variable: str | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.variable is None


class RegressionTree:
    """Weighted CART for squared error, following rpart's split criterion.

    rpart's improvement is the reduction in weighted SSE divided by the root
    deviance, and a split is kept when it is at least cp.  Surrogate splits and
    cross-validation are not ported: missing values are rejected and xval is
    accepted only for signature compatibility.
    """

    def __init__(self, control: TreeControl | None = None) -> None:
        self.control = control or tree_control()
        self.root: TreeNode | None = None
        self.root_deviance: float = 0.0
        self.variables: tuple[str, ...] = ()

    def fit(self, data: Mapping[str, Any], y: Any, w: Any = None) -> "RegressionTree":
        response = np.asarray(y, dtype=np.float64).ravel()
        weights = (
            np.ones(response.size) if w is None else np.asarray(w, dtype=np.float64).ravel()
        )
        if weights.size != response.size:
            raise ValueError("y and w must have the same length")
        if not np.all(np.isfinite(response)) or not np.all(np.isfinite(weights)):
            raise ValueError("tree terms reject missing or non-finite y / weights")
        columns: dict[str, np.ndarray] = {}
        for name, values in data.items():
            array = np.asarray(values)
            if array.ndim != 1 or array.size != response.size:
                continue
            if array.dtype.kind not in "iuf":
                raise NotImplementedError(
                    f"tree term variable {name!r} is not numeric; rpart's factor "
                    "splitting is not ported"
                )
            numeric = np.asarray(array, dtype=np.float64)
            if not np.all(np.isfinite(numeric)):
                raise ValueError("tree terms reject missing predictors")
            columns[name] = numeric
        self.variables = tuple(columns)
        indices = np.arange(response.size)
        mean = float(np.sum(weights * response) / np.sum(weights))
        self.root_deviance = float(np.sum(weights * (response - mean) ** 2))
        self.root = self._build(columns, response, weights, indices, 0)
        return self

    def _build(
        self,
        columns: Mapping[str, np.ndarray],
        y: np.ndarray,
        w: np.ndarray,
        indices: np.ndarray,
        depth: int,
    ) -> TreeNode:
        yv = y[indices]
        wv = w[indices]
        total_w = float(np.sum(wv))
        mean = float(np.sum(wv * yv) / total_w)
        deviance = float(np.sum(wv * (yv - mean) ** 2))
        node = TreeNode(value=mean, n=int(indices.size), deviance=deviance)
        control = self.control
        if (
            depth >= control.maxdepth
            or indices.size < control.minsplit
            or deviance <= 0.0
            or self.root_deviance <= 0.0
        ):
            return node
        best: tuple[float, str, float, np.ndarray, np.ndarray] | None = None
        for name in self.variables:
            values = columns[name][indices]
            order = np.argsort(values, kind="mergesort")
            sorted_x = values[order]
            sorted_y = yv[order]
            sorted_w = wv[order]
            n = sorted_x.size
            boundary = np.nonzero(sorted_x[:-1] < sorted_x[1:])[0]
            if boundary.size == 0:
                continue
            valid = boundary[
                (boundary + 1 >= control.minbucket) & (n - boundary - 1 >= control.minbucket)
            ]
            if valid.size == 0:
                continue
            cum_w = np.cumsum(sorted_w)
            cum_wy = np.cumsum(sorted_w * sorted_y)
            cum_wy2 = np.cumsum(sorted_w * sorted_y * sorted_y)
            total_wy = cum_wy[-1]
            total_wy2 = cum_wy2[-1]
            left_w = cum_w[valid]
            left_wy = cum_wy[valid]
            left_wy2 = cum_wy2[valid]
            right_w = total_w - left_w
            right_wy = total_wy - left_wy
            right_wy2 = total_wy2 - left_wy2
            sse_left = left_wy2 - (left_wy**2) / left_w
            sse_right = right_wy2 - (right_wy**2) / right_w
            improvement = (deviance - sse_left - sse_right) / self.root_deviance
            position = int(np.argmax(improvement))
            score = float(improvement[position])
            if best is None or score > best[0]:
                cut = int(valid[position])
                threshold = float(0.5 * (sorted_x[cut] + sorted_x[cut + 1]))
                mask = values <= threshold
                best = (score, name, threshold, indices[mask], indices[~mask])
        if best is None or best[0] < control.cp:
            return node
        _, name, threshold, left_idx, right_idx = best
        return TreeNode(
            value=mean,
            n=int(indices.size),
            deviance=deviance,
            variable=name,
            threshold=threshold,
            left=self._build(columns, y, w, left_idx, depth + 1),
            right=self._build(columns, y, w, right_idx, depth + 1),
        )

    def predict(self, data: Mapping[str, Any]) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("the tree has not been fitted")
        n = len(next(iter(data.values())))
        output = np.empty(n, dtype=np.float64)
        for index in range(n):
            node = self.root
            while not node.is_leaf:
                value = float(np.asarray(data[node.variable])[index])
                node = node.left if value <= node.threshold else node.right
            output[index] = node.value
        return output

    @property
    def leaf_count(self) -> int:
        count = 0
        stack = [self.root] if self.root is not None else []
        while stack:
            node = stack.pop()
            if node.is_leaf:
                count += 1
            else:
                stack.extend([node.left, node.right])
        return count


@dataclass(frozen=True)
class TreeTerm:
    """R's tr() term: an inner formula plus rpart-style control."""

    formula: str
    method: str
    control: TreeControl


def tr(formula: Any = None, method: str = "rpart", control: Any = None, **kwargs: Any):
    """R tr(): declare a regression-tree term (rpart, squared error)."""

    if not isinstance(formula, str) or not formula.strip().startswith("~"):
        raise ValueError("formula argument in tr() needs a formula starting with ~")
    if str(method) != "rpart":
        raise NotImplementedError('only method="rpart" is ported')
    if control is None:
        settings = tree_control(**dict(kwargs))
    elif isinstance(control, TreeControl):
        settings = control
    elif isinstance(control, Mapping):
        settings = tree_control(**dict(control))
    else:
        raise TypeError("control must be a TreeControl or a mapping")
    return TreeTerm(formula=formula.strip(), method="rpart", control=settings)


@dataclass(frozen=True)
class TreeFit:
    """R gamlss.tr's return list, reduced to the fields gamlss consumes."""

    fitted_values: np.ndarray
    residuals: np.ndarray
    nl_df: int
    coef_smo: RegressionTree
    var: Any = None


def gamlss_tr(
    x: TreeTerm = None,
    y: Any = None,
    w: Any = None,
    xeval: Any = None,
    data: Mapping[str, Any] = None,
    **kwargs: Any,
) -> TreeFit:
    """R gamlss.tr: fit the tree of a tr() term to the working response y.

    xeval (the prediction-frame branch of R) is not ported: prediction in this
    package goes through the model's design schema, and a tree term is not part
    of that schema yet.
    """

    if not isinstance(x, TreeTerm):
        raise TypeError("x must be a tr() term")
    if data is None:
        raise ValueError("the data argument is needed")
    if xeval is not None:
        raise NotImplementedError("the xeval (prediction) branch of gamlss.tr is not ported")
    inner = parse_formula(f"{_TERM_RESPONSE} {x.formula}")
    columns = {name: np.asarray(data[name]) for name in _formula_variables(inner)}
    tree = RegressionTree(x.control).fit(columns, y, w)
    fitted = tree.predict(columns)
    residuals = np.asarray(y, dtype=np.float64).ravel() - fitted
    # R: df = 2 * leaves - 1, nl.df = df - 1.
    return TreeFit(
        fitted_values=fitted,
        residuals=residuals,
        nl_df=2 * tree.leaf_count - 2,
        coef_smo=tree,
    )


def _formula_variables(parsed: ParsedFormula) -> list[str]:
    names: list[str] = []
    for term in parsed.linear_terms:
        names.append(term.variable)
    for term in parsed.smooth_terms:
        names.append(term.variable)
    for term in parsed.tensor_terms:
        names.extend(term.variables)
    return names


__all__ = [
    "AdditiveFit",
    "AdditiveTerm",
    "RegressionTree",
    "TreeControl",
    "TreeFit",
    "TreeTerm",
    "TreeNode",
    "ba",
    "ga",
    "gamlss_ba",
    "gamlss_ga",
    "gamlss_tr",
    "tr",
    "tree_control",
]
