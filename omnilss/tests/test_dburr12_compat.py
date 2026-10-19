from __future__ import annotations

import warnings

import jax

jax.config.update("jax_enable_x64", True)  # parity prerequisite
import numpy as np  # noqa: E402

from omnilss import DBURR12  # noqa: E402
from omnilss.distribution_registry import resolve  # noqa: E402
from omnilss.family_capabilities import (  # noqa: E402
    ImplementationStatus,
    get_family_capability,
)


def test_dburr12_constructor_and_registry() -> None:
    fam = DBURR12()
    assert fam.name == "DBURR12"
    assert fam.parameters == ("mu", "sigma", "nu")
    assert resolve("DBURR12").name == "DBURR12"


def test_dburr12_left_the_compat_alias_list() -> None:
    """Fail-closed admission only lets go once the numerics are there.

    D2.45 implemented the discretised Burr XII and produced the three pieces
    of evidence the set demands (density parity <= 8.3e-14, slot parity
    <= 2.2e-13, end-to-end deviance rel 7.9e-7 against live R); D2.46 is the
    separate admission commit.  The status moves to EXPERIMENTAL, not
    IMPLEMENTED: DBURR12 is not in ``_PRODUCTION_SAFE``, and claiming
    otherwise would be the kind of unverified assertion this project keeps
    removing.
    """
    cap = get_family_capability("DBURR12")
    assert cap.implementation_status is not ImplementationStatus.COMPAT_ALIAS
    assert cap.implementation_status is ImplementationStatus.EXPERIMENTAL


def test_dburr12_is_fittable_through_the_public_api() -> None:
    """``gamlss()`` used to raise FamilyCapabilityError for DBURR12."""
    from omnilss.controls import GAMLSSControl
    from omnilss.fitting import gamlss

    y = np.array([5.0, 2.0, 2.0, 6.0, 9.0, 2.0, 2.0, 1.0, 2.0, 2.0,
                  1.0, 0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0, 2.0, 2.0,
                  1.0, 2.0, 0.0, 5.0, 2.0, 2.0, 1.0, 1.0, 9.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = gamlss(
            formula="y ~ 1",
            family="DBURR12",
            data={"y": y},
            control=GAMLSSControl(n_cyc=200, c_crit=0.001, trace=False),
        )
    assert fit.convergence_status() is True, fit.termination_reason
    assert np.isfinite(float(fit.deviance))
