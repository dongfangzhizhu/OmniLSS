"""Control structures mirroring R `gamlss.control` and `glim.control`.

R source references:
- file: `gamlss/R/gamlss-5.R`
- functions: `gamlss.control`, `glim.control`
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class GAMLSSControl:
    """Python representation of `gamlss.control`."""

    c_crit: float = 0.001
    n_cyc: int = 20
    mu_step: float = 1.0
    sigma_step: float = 1.0
    nu_step: float = 1.0
    tau_step: float = 1.0
    gd_tol: float = math.inf
    iter: int = 0
    trace: bool = True
    autostep: bool = True
    save: bool = True

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class GLIMControl:
    """Python representation of `glim.control`."""

    cc: float = 0.001
    cyc: int = 50
    glm_trace: bool = False
    bf_cyc: int = 30
    bf_tol: float = 0.001
    bf_trace: bool = False

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def gamlss_control(
    c_crit: float = 0.001,
    n_cyc: int = 20,
    mu_step: float = 1.0,
    sigma_step: float = 1.0,
    nu_step: float = 1.0,
    tau_step: float = 1.0,
    gd_tol: float = math.inf,
    iter: int = 0,
    trace: bool = True,
    autostep: bool = True,
    save: bool = True,
) -> GAMLSSControl:
    """R reference: `gamlss/R/gamlss-5.R::gamlss.control`."""

    c_crit = 0.001 if c_crit <= 0 else c_crit
    n_cyc = 20 if n_cyc < 1 else n_cyc
    iter = 0 if iter < 0 else iter
    mu_step = 1.0 if not 0 <= mu_step <= 1 else mu_step
    sigma_step = 1.0 if not 0 <= sigma_step <= 1 else sigma_step
    nu_step = 1.0 if not 0 <= nu_step <= 1 else nu_step
    tau_step = 1.0 if not 0 <= tau_step <= 1 else tau_step
    gd_tol = math.inf if gd_tol < 0 else gd_tol
    return GAMLSSControl(
        c_crit=c_crit,
        n_cyc=n_cyc,
        mu_step=mu_step,
        sigma_step=sigma_step,
        nu_step=nu_step,
        tau_step=tau_step,
        gd_tol=gd_tol,
        iter=iter,
        trace=bool(trace),
        autostep=bool(autostep),
        save=bool(save),
    )


def glim_control(
    cc: float = 0.001,
    cyc: int = 50,
    glm_trace: bool = False,
    bf_cyc: int = 30,
    bf_tol: float = 0.001,
    bf_trace: bool = False,
) -> GLIMControl:
    """R reference: `gamlss/R/gamlss-5.R::glim.control`."""

    cc = 0.001 if cc <= 0 else cc
    bf_tol = 0.001 if bf_tol <= 0 else bf_tol
    cyc = 20 if cyc < 1 else cyc
    bf_cyc = 30 if bf_cyc < 1 else bf_cyc
    return GLIMControl(
        cc=cc,
        cyc=cyc,
        glm_trace=bool(glm_trace),
        bf_cyc=bf_cyc,
        bf_tol=bf_tol,
        bf_trace=bool(bf_trace),
    )
