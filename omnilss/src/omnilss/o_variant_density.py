"""Public density functions for the R 'o' (original) parameterisation families.

R gamlss.dist exports dBCPEo / dBCTo / dBCCGo / dBEo / dGEOMo / dJSUo /
dPARETO2o / dSHASHo alongside the family constructors.  The density kernels
already existed privately; these thin wrappers expose them with R's signatures
and log switch.  Verified against live R (gamlss.dist 6.1.1) to <= 7.1e-15 on
the audit grid.
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .distributions_b12 import _beo_log_pdf, _geomo_log_pdf, _pareto2o_log_pdf
from .distributions_b14 import _bcpeo_log_pdf, _bcto_log_pdf, _jsuo_log_pdf
from .distributions_b16 import _bccgo_log_pdf
from .distributions_b3 import _shasho_log_pdf


def _emit(log_value: Any, log: bool) -> Any:
    return log_value if log else jnp.exp(log_value)


def dBCCGo(x, mu=1.0, sigma=0.1, nu=1.0, log: bool = False):
    """R dBCCGo: Box-Cox-Cole-Green (original parameterisation)."""
    return _emit(_bccgo_log_pdf(x, mu, sigma, nu), log)


def dBCPEo(x, mu=5.0, sigma=0.1, nu=1.0, tau=2.0, log: bool = False):
    return _emit(_bcpeo_log_pdf(x, mu, sigma, nu, tau), log)


def dBCTo(x, mu=5.0, sigma=0.1, nu=1.0, tau=2.0, log: bool = False):
    return _emit(_bcto_log_pdf(x, mu, sigma, nu, tau), log)


def dBEo(x, mu=0.5, sigma=0.2, log: bool = False):
    return _emit(_beo_log_pdf(x, mu, sigma), log)


def dGEOMo(x, mu=0.5, log: bool = False):
    return _emit(_geomo_log_pdf(x, mu), log)


def dJSUo(x, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, log: bool = False):
    return _emit(_jsuo_log_pdf(x, mu, sigma, nu, tau), log)


def dPARETO2o(x, mu=1.0, sigma=0.5, log: bool = False):
    return _emit(_pareto2o_log_pdf(x, mu, sigma), log)


def dSHASHo(x, mu=0.0, sigma=1.0, nu=0.0, tau=1.0, log: bool = False):
    return _emit(_shasho_log_pdf(x, mu, sigma, nu, tau), log)


__all__ = [
    "dBCCGo",
    "dBCPEo",
    "dBCTo",
    "dBEo",
    "dGEOMo",
    "dJSUo",
    "dPARETO2o",
    "dSHASHo",
]
