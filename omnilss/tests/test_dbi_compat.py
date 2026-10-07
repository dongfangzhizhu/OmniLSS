"""``DBI`` constructor and registry wiring.

The assertions here were written when ``DBI()`` was a shell forwarding every
slot to ``BI()``.  D2.23 replaced it with a real two-parameter likelihood, so
the parameter list below is the point of the file rather than an incidental
detail: it is the cheapest possible check that the shell did not come back.

R's ``DBI`` (Borg 1988) estimates ``(mu, sigma)`` with links ``logit`` and
``log``.  A regression to ``("mu",)`` would still fit, still converge and
still report a plausible deviance -- just ``BI``'s, which is exactly how the
shell stayed invisible until D2.22's deviance comparison caught it.
"""

from __future__ import annotations

from omnilss import DBI
from omnilss.distribution_registry import resolve


def test_dbi_constructor_and_registry() -> None:
    fam = DBI()
    assert fam.name == "DBI"
    assert fam.parameters == ("mu", "sigma"), (
        "DBI regressed to a single-parameter BI forwarder"
    )
    assert dict(fam.links) == {"mu": "logit", "sigma": "log"}
    assert resolve("DBI").name == "DBI"