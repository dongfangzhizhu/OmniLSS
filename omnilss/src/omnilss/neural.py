# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.add neural-network term: nn() / gamlss_nn (D2.251).

R source: R/gamlss.add/R/nnet_gamlss.R.  R delegates the fit to nnet(), a
single-hidden-layer network trained by BFGS from random weights, and reuses the
previous weight vector on later backfitting passes.

The port implements the same term interface and the same control surface
(nn_control), with an equivalent one-hidden-layer network:

    hidden  logistic units, size = control["size"]
    output  linear when linout (the default), otherwise logistic
    skip    optional direct input-output connection
    loss    sum(w * (y - f)^2) / n + decay * sum(weights^2)

It is fitted with SciPy L-BFGS-B, seeded by the seed argument for
reproducibility.  This is NOT a bit-for-bit port of nnet's BFGS: the optimiser,
its random start and the internal scaling differ.  The verification is
functional (approximation quality, determinism, warm start, decay, the skip-only
linear case), not a numerical comparison with nnet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .add_controls import nn_control
from .formula_parser import ParsedFormula, parse_formula

_TERM_RESPONSE = "__nn_response__"


@dataclass(frozen=True)
class NnTerm:
    """R nn(): an inner formula plus its nn.control settings."""

    formula: str
    control: Mapping[str, Any]


def nn(formula: Any = None, control: Any = None, **kwargs: Any) -> NnTerm:
    """R nn(): declare a single-hidden-layer neural-network term."""

    if not isinstance(formula, str) or not formula.strip().startswith("~"):
        raise ValueError("formula argument in nn() needs a formula starting with ~")
    if control is None:
        settings = nn_control(**dict(kwargs))
    elif isinstance(control, Mapping):
        settings = dict(control)
    else:
        settings = control
    parsed = parse_formula(f"{_TERM_RESPONSE} {formula.strip()}")
    unsupported = [t.smoother for t in parsed.smooth_terms] + [
        t.smoother for t in parsed.tensor_terms
    ]
    if unsupported:
        raise NotImplementedError(
            "nn() takes a plain additive formula (for example ~ x + z); nested "
            f"smoothers are not supported, got {unsupported}"
        )
    return NnTerm(formula=formula.strip(), control=settings)


def _variables(parsed: ParsedFormula) -> list[str]:
    return [term.variable for term in parsed.linear_terms]


class NeuralNet:
    """One-hidden-layer network with an optional skip connection."""

    def __init__(
        self,
        size: int = 3,
        linout: bool = True,
        skip: bool = False,
        decay: float = 0.0,
        rang: float = 0.7,
        seed: int = 0,
    ) -> None:
        self.size = int(size)
        self.linout = bool(linout)
        self.skip = bool(skip)
        self.decay = float(decay)
        self.rang = float(rang)
        self.seed = int(seed)
        self.variables: tuple[str, ...] = ()
        self.weights: np.ndarray | None = None

    def _init_weights(self, n_inputs: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        parts = []
        if self.size > 0:
            parts.append(rng.uniform(-self.rang, self.rang, (n_inputs, self.size)))
            parts.append(rng.uniform(-self.rang, self.rang, self.size))
            parts.append(rng.uniform(-self.rang, self.rang, self.size))
        if self.skip:
            parts.append(rng.uniform(-self.rang, self.rang, n_inputs))
        parts.append(np.array([0.0]))
        return np.concatenate([np.ravel(part) for part in parts])

    def _unpack(self, weights: np.ndarray, n_inputs: int):
        index = 0
        matrices = {}
        if self.size > 0:
            matrices["W1"] = weights[index : index + n_inputs * self.size].reshape(
                n_inputs, self.size
            )
            index += n_inputs * self.size
            matrices["b1"] = weights[index : index + self.size]
            index += self.size
            matrices["W2"] = weights[index : index + self.size]
            index += self.size
        if self.skip:
            matrices["Ws"] = weights[index : index + n_inputs]
            index += n_inputs
        matrices["b2"] = weights[index : index + 1]
        return matrices

    def _forward(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        matrices = self._unpack(weights, X.shape[1])
        output = np.zeros(X.shape[0])
        if self.size > 0:
            hidden = 1.0 / (1.0 + np.exp(-(X @ matrices["W1"] + matrices["b1"])))
            output = output + hidden @ matrices["W2"]
        if self.skip:
            output = output + X @ matrices["Ws"]
        output = output + matrices["b2"][0]
        if not self.linout:
            output = 1.0 / (1.0 + np.exp(-output))
        return output

    def fit(
        self,
        data: Mapping[str, Any],
        y: Any,
        w: Any = None,
        init_weights: Any = None,
        maxit: int = 100,
        abstol: float = 1e-4,
        reltol: float = 1e-8,
    ) -> "NeuralNet":
        response = np.asarray(y, dtype=np.float64).ravel()
        weights = (
            np.ones(response.size)
            if w is None
            else np.asarray(w, dtype=np.float64).ravel()
        )
        columns = []
        for name in self.variables:
            values = np.asarray(data[name])
            if values.dtype.kind not in "iuf":
                raise NotImplementedError(
                    f"nn() predictor {name!r} is not numeric; only numeric inputs are ported"
                )
            columns.append(np.asarray(values, dtype=np.float64).ravel())
        if not columns:
            raise ValueError("nn() needs at least one predictor")
        X = np.column_stack(columns)
        if init_weights is None:
            start = self._init_weights(X.shape[1])
        else:
            start = np.asarray(init_weights, dtype=np.float64).ravel()
            expected = self._init_weights(X.shape[1]).size
            if start.size != expected:
                raise ValueError(
                    f"init_weights has {start.size} values, expected {expected}"
                )

        def loss(candidate: np.ndarray) -> float:
            fitted = self._forward(X, candidate)
            residual = response - fitted
            value = float(np.mean(weights * residual * residual))
            if self.decay:
                value += self.decay * float(np.sum(candidate * candidate))
            return value

        from scipy.optimize import minimize

        result = minimize(
            loss,
            start,
            method="L-BFGS-B",
            options={"maxiter": int(maxit), "ftol": reltol, "gtol": abstol},
        )
        self.weights = np.asarray(result.x, dtype=np.float64)
        self._n_inputs = X.shape[1]
        return self

    def predict(self, data: Mapping[str, Any]) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("the network has not been fitted")
        columns = [
            np.asarray(data[name], dtype=np.float64).ravel() for name in self.variables
        ]
        if not columns:
            raise ValueError("nn() needs at least one predictor")
        X = np.column_stack(columns)
        return self._forward(X, self.weights)

    @property
    def n_weights(self) -> int:
        if self.weights is None:
            raise RuntimeError("the network has not been fitted")
        return int(self.weights.size)


@dataclass(frozen=True)
class NnFit:
    """R gamlss.nn's return list, reduced to the fields gamlss consumes."""

    fitted_values: np.ndarray
    residuals: np.ndarray
    nl_df: int
    coef_smo: NeuralNet
    var: Any = None


def gamlss_nn(
    x: NnTerm = None,
    y: Any = None,
    w: Any = None,
    data: Mapping[str, Any] = None,
    xeval: Any = None,
    warm_start: Any = None,
    seed: int = 0,
    **kwargs: Any,
) -> NnFit:
    """R gamlss.nn: fit the nn() term for the working response y.

    R reuses the previous weight vector on later backfitting passes through its
    gamlss frame; here the caller passes warm_start (the coef_smo.weights of the
    previous call).  The xeval prediction branch is not ported.
    """

    if not isinstance(x, NnTerm):
        raise TypeError("x must be an nn() term")
    if data is None:
        raise ValueError("the data argument is needed")
    if xeval is not None:
        raise NotImplementedError(
            "the xeval (prediction) branch of gamlss.nn is not ported"
        )
    parsed = parse_formula(f"{_TERM_RESPONSE} {x.formula}")
    control = x.control
    network = NeuralNet(
        size=int(control.get("size", 3)),
        linout=bool(control.get("linout", True)),
        skip=bool(control.get("skip", False)),
        decay=float(control.get("decay", 0.0)),
        rang=float(control.get("rang", 0.7)),
        seed=seed,
    )
    network.variables = tuple(_variables(parsed))
    network.fit(
        data,
        y,
        w,
        init_weights=warm_start,
        maxit=int(control.get("maxit", 100)),
        abstol=float(control.get("abstol", 1e-4)),
        reltol=float(control.get("reltol", 1e-8)),
    )
    fitted = network.predict(data)
    response = np.asarray(y, dtype=np.float64).ravel()
    return NnFit(
        fitted_values=fitted,
        residuals=response - fitted,
        nl_df=network.n_weights,
        coef_smo=network,
    )


__all__ = [
    "NeuralNet",
    "NnFit",
    "NnTerm",
    "gamlss_nn",
    "nn",
]
