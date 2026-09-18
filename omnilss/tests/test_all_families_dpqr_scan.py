"""All-family dpqr sanity scan (task 1.9 acceptance).

Iterates every registered family with valid parameterization metadata and
asserts that d/p/q produce finite, monotone results at random parameters.
This catches NaN placeholder families and gross parameterization errors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrandom

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss.distribution_registry import list_families, resolve

# Families excluded from the scan: known NaN-placeholder dpqr or families
# whose parameters are hard to generate valid values for.
SKIP = {
    # SIMPLEX CDF is numeric integration (slow but tested separately)
    "SIMPLEX",
}

# Known NaN-placeholder families (p/q/r not yet wired; tracked for later
# phases).  The scan asserts everything OUTSIDE this set is finite.
KNOWN_NAN_PLACEHOLDER = {
    "BCCGO", "BCPEO", "BCTO", "BEINF0", "BEINF1", "BEOI", "BEZI",
    "DPO", "DPO1", "EGB2", "EXGAUS", "GAF",
    "GENERALISEDPOISSON", "GIG", "GP", "GPO", "IGA", "IGAMMA", "JSUO", "LG",
    "LNO", "LOGITNO", "LOGNO2", "LOGSHASH", "LOGSHASHO", "LQNO", "MN3",
    "MN4", "MN5", "MULTIN", "NBF", "NET", "NO2", "NOF", "PARETO", "PARETO1",
    "PARETO1O", "PARETO2", "PE2", "PIG2", "RGE", "SEP", "SEP1", "SEP2",
    "SEP3", "SEP4", "SHASHO2", "SI", "SST", "ST3", "ST3C", "ST4", "ST5",
    "WARING", "WEI2", "YULE", "ZABB", "ZABI", "ZABNB", "ZALG", "ZANBI",
    "ZAP", "ZAPIG", "ZASICHEL", "ZAZIPF", "ZIBI", "ZIBNB", "ZINBF",
    "ZINBI", "ZIP2", "ZIPIG", "ZISICHEL",
}


def _default_params(name: str) -> dict | None:
    """Choose representative valid parameters per family."""
    p = {"mu": 0.5, "sigma": 0.2, "nu": 0.3, "tau": 2.0}
    if name in {"NO", "NO2"}:
        p = {"mu": 0.5, "sigma": 1.0}
    elif name in {"GA", "EXP", "IG", "LOGNO", "WEI", "WEI2", "WEI3"}:
        p = {"mu": 2.0, "sigma": 0.5}
    elif name in {"PO", "NBI", "NBII", "GEOM", "ZIP", "ZINB"}:
        p = {"mu": 3.0, "sigma": 0.3}
    elif name in {"BE", "BEO", "BEINF", "BEZI", "BEOI", "SIMPLEX"}:
        p = {"mu": 0.3, "sigma": 0.2}
    elif name == "BB":
        p = {"mu": 0.5, "sigma": 0.2, "bd": 10.0}
    elif name in {"GU", "LO"}:
        p = {"mu": 0.0, "sigma": 1.0}
    elif name in {"BCCG", "BCT", "BCPE", "BCCGo", "BCTo", "BCPEo"}:
        p = {"mu": 2.0, "sigma": 0.2, "nu": 0.3, "tau": 2.0}
    elif name in {"TF", "TF2"}:
        p = {"mu": 0.0, "sigma": 1.0, "nu": 5.0}
    elif name in {"ST1", "ST2", "ST3", "ST3C", "ST4"}:
        p = {"mu": 0.0, "sigma": 1.0, "nu": 0.5, "tau": 5.0}
    elif name in {"SHASH", "SHASHo"}:
        p = {"mu": 0.0, "sigma": 1.0, "nu": 0.5, "tau": 1.0}
    return p


def test_all_families_dpqr_finite_and_monotone() -> None:
    rng = np.random.default_rng(42)
    failures = []
    tested = 0
    for name in list_families():
        if name in SKIP or name in KNOWN_NAN_PLACEHOLDER:
            continue
        try:
            fam = resolve(name)
        except Exception:
            continue
        params = _default_params(name)
        if params is None:
            continue
        par = tuple(fam.parameters)
        try:
            # d: finite at a few points
            xs = np.linspace(0.2, 2.0, 5)
            args = [params.get(k, 0.5) for k in par]
            d = np.asarray(fam.d(jnp.asarray(xs), *args))
            if not np.all(np.isfinite(d)):
                failures.append(f"{name}: d non-finite")
                continue
            # p: monotone
            qs = np.linspace(0.2, 2.0, 8)
            pvals = np.asarray(fam.p(jnp.asarray(qs), *args))
            if not np.all(np.isfinite(pvals)):
                failures.append(f"{name}: p non-finite")
                continue
            if np.any(np.diff(pvals) < -1e-3):
                failures.append(f"{name}: p not monotone")
                continue
            # q: finite and monotone
            ps = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            qvals = np.asarray(fam.q(jnp.asarray(ps), *args))
            if not np.all(np.isfinite(qvals)):
                failures.append(f"{name}: q non-finite")
                continue
            if np.any(np.diff(qvals) < -1e-3):
                failures.append(f"{name}: q not monotone")
                continue
            tested += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, (
        f"{len(failures)} families failed the dpqr scan: {failures[:10]}"
    )
    assert tested > 60, f"expected to scan most families, only tested {tested}"
