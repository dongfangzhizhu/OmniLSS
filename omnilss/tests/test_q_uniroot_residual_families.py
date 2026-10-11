"""dpqr residuals that belong to R's own solver tolerances (D2.32).

Nine gate failures where our value tracks a high-precision re-solve of R's
own objective to ~1e-9 while R's returned value carries the error of its
default-tolerance solver:

- qIG / qSEP1 / qSEP2 / qSHASH / qSN1 / qST1 / qST2 are uniroot solves at
  the DEFAULT tol (.Machine$double.eps^0.25 ~ 1.2e-4) -- live-R source;
  qZAIG composes qIG and inherits the error;
- pSEP1 evaluates adaptive integrate() at its default settings per point
  (measured self-error up to 1.13e-06 on the probe grid).

These are registered in tools/rs_family_parity_scan.py _DPQR_TOLERANCE with
the live-R evidence.  The tests here pin the two sides at the registered
slack: ours must stay inside R's solver noise (so a future regression on
our side cannot hide behind the registration), and p/q must stay mutually
inverse at our own precision.
"""

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

import rs_family_parity_scan  # noqa: E402  (registers the tolerances below)
from rs_family_parity_scan import _family_grids  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss.distributions import resolve_family  # noqa: E402
from tests.rbus.core import RTestBus  # noqa: E402

# (family, params tuple, registered q tolerance, p grid for the round-trip
# test) -- params in family order.  ZAIG is zero-altered: for p <= nu the
# quantile is the atom 0 and F(0) = nu, so the round-trip only makes sense
# on p strictly above the largest nu in the probe grid.
P_GRID = np.linspace(0.1, 0.9, 8)
Q_FAMILIES = [
    ("IG", ("mu", "sigma"), 1e-4, P_GRID),
    ("SEP1", ("mu", "sigma", "nu", "tau"), 1e-4, P_GRID),
    ("SEP2", ("mu", "sigma", "nu", "tau"), 1e-4, P_GRID),
    ("SHASH", ("mu", "sigma", "nu", "tau"), 1e-4, P_GRID),
    ("SN1", ("mu", "sigma", "nu"), 1e-4, P_GRID),
    ("ST1", ("mu", "sigma", "nu", "tau"), 1e-4, P_GRID),
    ("ST2", ("mu", "sigma", "nu", "tau"), 1e-4, P_GRID),
    ("ZAIG", ("mu", "sigma", "nu"), 1e-4, np.linspace(0.78, 0.94, 8)),
]


def _param_arrays(family: str, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """The scanner's own probe grids (link-domain derived, always legal).

    Hand-rolled nu vectors walked outside the parameter's own legal domain
    (SHASH wants nu > 0, ZAIG wants 0 < nu < 1) -- the D2.28 lesson that a
    probe's parameters, not just its implementation, can fake a failure.
    """
    _, parrays, _bd = _family_grids(family, 8)
    return {n: np.asarray(a, dtype=np.float64) for n, a in zip(names, parrays)}


@pytest.mark.parametrize("family,param_names,tol", [(f, p, t) for f, p, t, _ in Q_FAMILIES],
                         ids=[f[0] for f in Q_FAMILIES])
def test_q_stays_within_r_s_own_uniroot_noise(family, param_names, tol) -> None:
    """Our q vs live R's default-tolerance uniroot, at the registered slack."""
    params = _param_arrays(family, param_names)
    r_vals = RTestBus().eval_family(
        family, "q", {"p": P_GRID.tolist(), **{k: v.tolist() for k, v in params.items()}}
    )
    ours = np.asarray(resolve_family(family).q(P_GRID, *[params[n] for n in param_names]))
    np.testing.assert_allclose(ours, r_vals, atol=tol)


@pytest.mark.parametrize("family,param_names,p_grid",
                         [(f, p, g) for f, p, _, g in Q_FAMILIES],
                         ids=[f[0] for f in Q_FAMILIES])
def test_p_and_q_stay_mutually_inverse(family, param_names, p_grid) -> None:
    """Whatever the registered slack vs R, our own p/q must round-trip."""
    params = _param_arrays(family, param_names)
    fam = resolve_family(family)
    q = np.asarray(fam.q(p_grid, *[params[n] for n in param_names]))
    back = np.asarray(fam.p(q, *[params[n] for n in param_names]))
    np.testing.assert_allclose(back, p_grid, atol=1e-6)


def test_sep1_p_stays_within_r_s_own_quadrature_noise() -> None:
    """SEP1 p: R integrates dSEP1 adaptively per point; ours is the exact CDF."""
    params = _param_arrays("SEP1", ("mu", "sigma", "nu", "tau"))
    q_grid, parrays, _bd = _family_grids("SEP1", 8)
    q_grid = np.asarray(q_grid, dtype=np.float64)
    r_vals = RTestBus().eval_family(
        "SEP1", "p", {"q": q_grid.tolist(), **{k: v.tolist() for k, v in params.items()}}
    )
    ours = np.asarray(
        resolve_family("SEP1").p(q_grid, *[params[n] for n in ("mu", "sigma", "nu", "tau")])
    )
    np.testing.assert_allclose(ours, r_vals, atol=1e-5)
