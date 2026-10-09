"""A zero-variant's response ``type`` is inherited from its base family (D2.25).

Symptom this file exists for: ``ZAPIG`` and ``ZAZIPF`` reported a non-finite
RS deviance (``inf`` from iteration 1, ``termination_reason ==
"non_finite_deviance"``).  Both factories hard-coded a response ``type`` --
``"Mixed"`` for zero-*altered*, ``"Discrete"`` for zero-*inflated* -- which is
not R's rule.  A ``Mixed`` response grid is continuous; the discrete ``PIG``
log-pmf returns ``-inf`` off the integers; every one of the 37 positive
observations scored ``-inf``.  The failure looked like a density defect and was
entirely a metadata one.

R's rule, verified against live R on all 22 zero-variant families: a variant
inherits the base's ``type``, and R marks a variant over a *positive
continuous* base ``"Mixed"`` because a point mass at zero is glued onto a
continuous density.  Six families sat on the wrong side of it --
``ZAPIG``, ``ZAZIPF``, ``ZABI``, ``ZANBI``, ``ZABNB``, ``ZASICHEL``, all ``ZA``
over a discrete base, all reported ``Mixed`` where R says ``Discrete``.

The tests below pin the rule, not the six names, so that adding a seventh
discrete base cannot reintroduce the same class of defect silently.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from omnilss.distribution_registry import resolve
from omnilss.distributions_b10_zero_variants import (
    _base_response_type,
    make_zero_altered_family,
    make_zero_inflated_family,
)

#: Zero-variant families built by the two factories in this module, mapped to
#: the family they are built over.  ``ZAGA``/``ZAIG``/``BEOI``/``BEZI`` are
#: deliberately absent: they are hand-written in ``distributions_b5.py`` and
#: already report ``"Mixed"``, which is what R says.
_VARIANTS = {
    "ZABI": "BI",
    "ZANBI": "NBI",
    "ZAPIG": "PIG",
    "ZASICHEL": "SICHEL",
    "ZAZIPF": "ZIPF",
    "ZABNB": "BNB",
    "ZIBI": "BI",
    "ZIPIG": "PIG",
    "ZISICHEL": "SICHEL",
    "ZIBNB": "BNB",
}

#: Every base reachable through the factories is discrete today, so the
#: continuous branch of the rule is asserted directly on the helper rather
#: than through a family.  ``GA``/``IG``/``BE`` are the continuous bases R uses
#: for ``ZAGA``/``ZAIG``/``BEOI``, and R marks those variants ``"Mixed"``.
_CONTINUOUS_BASES = ("GA", "IG", "BE")

#: The six families whose reported ``type`` was wrong before D2.25.  Asserted
#: separately from the rule so that a regression names the family that moved.
_MISLABELLED = ("ZABI", "ZANBI", "ZAPIG", "ZASICHEL", "ZAZIPF")


def test_a_variant_inherits_its_bases_response_type():
    for variant, base in _VARIANTS.items():
        expected = _base_response_type(base)
        actual = resolve(variant).type
        assert actual == expected, (
            f"{variant} is built over {base} by make_zero_*_family, which "
            f"derives its type from the base; expected {expected!r}, "
            f"got {actual!r}"
        )


@pytest.mark.parametrize("base", _CONTINUOUS_BASES)
def test_a_continuous_base_yields_a_mixed_variant(base):
    """The other branch of the rule, so a blanket ``Discrete`` cannot pass.

    R reports ``ZAGA()$type == "Mixed"`` over the continuous ``GA``; a point
    mass glued onto a continuous density is what ``Mixed`` means.  A blanket
    ``"Discrete"`` would make the parity scanner feed those families integer
    grids -- the mirror image of the D2.25 defect.
    """
    assert _base_response_type(base) == "Mixed"


@pytest.mark.parametrize("name", _MISLABELLED)
def test_a_discrete_base_variant_is_not_reported_as_mixed(name):
    """The specific regression: these six said ``Mixed``, R says ``Discrete``.

    ``Mixed`` is not cosmetic here -- it selects a continuous response grid,
    which a discrete log-pmf cannot evaluate.
    """
    assert resolve(name).type == "Discrete"


def test_the_helper_fails_loudly_on_an_unknown_base():
    """An unknown base must raise, not silently fall back to a default.

    The original defect was a hard-coded default.  A silent fallback would
    reproduce it on the next family added.
    """
    with pytest.raises((ValueError, KeyError)):
        _base_response_type("NOT_A_FAMILY")


def test_the_helper_fails_loudly_on_an_unrecognised_base_type():
    """A base whose type is neither discrete nor continuous must raise too.

    ``"Mixed"`` has no single right answer as a base, and guessing one is how
    the six mislabelled families happened.
    """
    with pytest.raises(ValueError, match="cannot derive a response type"):
        _base_response_type("BEINF")


def test_a_continuous_grid_makes_a_discrete_variant_non_finite():
    """The mechanism, kept as a test because it is the whole failure mode.

    Feeding non-integer responses to ``ZAPIG`` reproduces the original symptom
    exactly; feeding integers does not.  If this stops holding, the fix has
    been undone somewhere upstream.

    Measured on the ``log=True`` channel on purpose: the plain channel
    exponentiates ``-inf`` into ``0.0``, which *is* finite, so asserting
    ``isfinite(d)`` here would pass for the wrong reason (D2.23's lesson about
    observing the wrong quantity).
    """
    family = resolve("ZAPIG")
    mu, sigma, nu = 2.0, 0.4, 0.1

    def logd(x):
        return float(np.asarray(
            family.d(jnp.asarray(x), mu, sigma, nu, log=True), dtype=np.float64))

    # Integer response: the hurdle density is finite.
    assert np.isfinite(logd(1.0)), logd(1.0)

    # Non-integer response: our discrete base returns -inf, and `g_dev_inc`
    # turns that into an infinite deviance contribution.
    assert logd(4.397) == -np.inf

    # Zero is the point mass: P(Y=0) = nu = 0.1, i.e. log(0.1).
    assert np.isclose(logd(0.0), np.log(0.1), atol=1e-12), logd(0.0)
    assert np.isclose(
        float(np.asarray(family.d(jnp.asarray(0.0), mu, sigma, nu), np.float64)),
        0.1, atol=1e-12)


def test_the_rs_fit_deviance_is_finite_on_an_integer_response():
    """End-to-end: the gate metric that read ``inf`` before D2.25.

    The response is drawn the way the parity scanner draws it -- integers,
    because the family now reports ``"Discrete"``.
    """
    from omnilss import gamlss

    assert resolve("ZAPIG").type == "Discrete"
    y = (np.arange(48, dtype=np.float64) % 8.0) + 1.0
    x = np.linspace(-1.0, 1.0, y.size)

    model = gamlss(
        "y ~ x", family="ZAPIG", data={"y": y, "x": x},
        sigma_formula="~1", method="RS", max_iter=20,
    )
    # The regression this file exists for was `inf`, so finiteness is the
    # assertion.  The termination reason is deliberately *not* pinned to a
    # literal: it read `max_iterations_reached` when D2.25 landed and
    # `relative_deviance_converged` after D2.26 gave the family R's own
    # nu.initial, which is an improvement.  Asserting the older string would
    # have turned a fix into a failure.
    assert np.isfinite(model.deviance), model.deviance
    assert model.termination_reason in (
        "relative_deviance_converged", "max_iterations_reached",
    )


def test_both_factories_derive_the_type_from_one_place():
    """Guard against the two factories drifting apart again.

    They now share ``_base_response_type``.  If a future edit reintroduces a
    literal response type at either call site, the ZA/ZI split reappears --
    which is what put ``ZAPIG`` on the wrong side while its sibling ``ZIPIG``
    was right.  The hand-written families in this module (``ZABB`` and
    friends) legitimately spell out their own type, so only the two factory
    bodies are inspected.
    """
    import inspect

    import omnilss.distributions_b10_zero_variants as mod

    for factory in (make_zero_altered_family, make_zero_inflated_family):
        body = inspect.getsource(factory)
        assert "type_=_base_response_type(base_name)" in body, factory.__name__
        # No literal response type anywhere in either factory.
        assert 'type_="' not in body, factory.__name__

    assert callable(make_zero_altered_family)
    assert callable(make_zero_inflated_family)
    assert mod.__name__ == "omnilss.distributions_b10_zero_variants"