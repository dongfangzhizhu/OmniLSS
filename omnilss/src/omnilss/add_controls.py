"""R gamlss.add control constructors: nn.control() and ga.control().

R sources: R/gamlss.add/R/nnet_gamlss.R (nn.control) and
R/gamlss.add/R/ga_new.R (ga.control).

ga.control() wraps mgcv::gam.control(), so the port carries mgcv's documented
defaults for the inner "control" list (scalar options plus the nlm / optim /
newton sub-lists); overrides are merged on top, exactly as gam.control does.

R bug kept on purpose: nn.control's size guard assigns to "order" instead of
"size", so a negative size warns but the value is returned unchanged.
"""
from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np


def _warn(message: str) -> None:
    warnings.warn(message, UserWarning, stacklevel=3)


def nn_control(
    size: int = 3,
    linout: bool = True,
    entropy: bool = False,
    softmax: bool = False,
    censored: bool = False,
    skip: bool = False,
    rang: float = 0.7,
    decay: float = 0.0,
    maxit: int = 100,
    Hess: bool = False,
    trace: bool = False,
    MaxNWts: int = 1000,
    abstol: float = 1.0e-4,
    reltol: float = 1.0e-8,
    **kwargs: Any,
) -> dict[str, Any]:
    """R nn.control: settings for gamlss.nn() (nnet based)."""

    if size < 0:
        # R assigns to "order" here, so the value is left alone.
        _warn(
            "the value of size supplied is negative the default value of 3 was used instead"
        )
    if rang <= 0:
        _warn(
            "the value of rang supplied is less than zero the default value of 0.7 was used instead"
        )
        rang = 0.7
    if decay < 0:
        _warn(
            "the value of decay supplied is negative the default value of 0 was used instead"
        )
        decay = 0.0
    if maxit < 0:
        _warn(
            "the value of maxit supplied is negative the default value of 100 was used instead"
        )
        maxit = 100
    if MaxNWts < 0:
        _warn(
            "the value of MaxNWts supplied is negative the default value of 1000 was used instead"
        )
        MaxNWts = 1000
    if abstol < 0:
        _warn(
            "the value of abstol supplied is negative the default value of  1.0e-4 was used instead"
        )
        abstol = 1.0e-4
    if reltol < 0:
        _warn(
            "the value of reltol supplied is negative the default value of  1.0e-8 was used instead"
        )
        reltol = 1.0e-8

    return {
        "size": size,
        "linout": bool(linout),
        "entropy": bool(entropy),
        "softmax": bool(softmax),
        "censored": bool(censored),
        "skip": bool(skip),
        "rang": rang,
        "decay": decay,
        "maxit": maxit,
        "Hess": bool(Hess),
        "trace": bool(trace),
        "MaxNWts": MaxNWts,
        "abstol": abstol,
        "reltol": reltol,
    }


def gam_control(
    epsilon: float = 1.0e-7,
    maxit: int = 200,
    trace: bool = False,
    nthreads: int = 1,
    irls_reg: float = 0.0,
    mgcv_tol: float = 1.0e-7,
    mgcv_half: int = 15,
    rank_tol: float | None = None,
    nlm: dict[str, Any] | None = None,
    optim: dict[str, Any] | None = None,
    newton: dict[str, Any] | None = None,
    keep_data: bool = False,
    scale_est: str = "fletcher",
    edge_correct: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """mgcv::gam.control defaults (the list ga.control() nests under "control")."""

    if rank_tol is None:
        rank_tol = float(np.sqrt(np.finfo(np.float64).eps))

    inner = {
        "nthreads": nthreads,
        "ncv.threads": 1,
        "irls.reg": irls_reg,
        "epsilon": epsilon,
        "maxit": maxit,
        "trace": bool(trace),
        "mgcv.tol": mgcv_tol,
        "mgcv.half": mgcv_half,
        "rank.tol": rank_tol,
        "nlm": {
            "ndigit": 7,
            "gradtol": 1.0e-6,
            "stepmax": 2.0,
            "steptol": 1.0e-4,
            "iterlim": 200,
            "check.analyticals": False,
            **(nlm or {}),
        },
        "optim": {"factr": 1.0e7, **(optim or {})},
        "newton": {
            "conv.tol": 1.0e-6,
            "maxNstep": 5.0,
            "maxSstep": 2.0,
            "maxHalf": 30,
            "use.svd": False,
            **(newton or {}),
        },
        "idLinksBases": True,
        "scalePenalty": True,
        "efs.lspmax": 15,
        "efs.tol": 0.1,
        "keepData": bool(keep_data),
        "scale.est": scale_est,
        "edge.correct": bool(edge_correct),
    }
    inner.update(kwargs)
    return inner


def ga_control(
    offset: Any = None,
    method: str = "REML",
    optimizer: Sequence[str] = ("outer", "newton"),
    control: dict[str, Any] | None = None,
    scale: float = 0.0,
    select: bool = False,
    knots: Any = None,
    sp: Any = None,
    min_sp: Any = None,
    H: Any = None,
    gamma: float = 1.0,
    paraPen: Any = None,
    in_out: Any = None,
    drop_unused_levels: bool = True,
    drop_intercept: Any = None,
    discrete: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """R ga.control: settings for gamlss.ga() (mgcv::gam wrapper)."""

    # R overwrites its own "control" argument with gam.control(...) inside the
    # body, so passing control=... has no effect; the port reproduces that.
    inner = gam_control(**kwargs)
    result = {
        "offset": offset,
        "method": method,
        "optimizer": list(optimizer),
        "control": inner,
        "scale": scale,
        "select": bool(select),
        "knots": knots,
        "sp": sp,
        "min.sp": min_sp,
        "H": H,
        "gamma": gamma,
        "paraPen": paraPen,
        "in.out": in_out,
        "drop.unused.levels": bool(drop_unused_levels),
        "drop.intercept": drop_intercept,
        "discrete": bool(discrete),
    }
    result.update(kwargs)
    return result


__all__ = ["ga_control", "gam_control", "nn_control"]