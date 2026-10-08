"""The eta-scale score of a fitted RS point, published as a diagnostic (D2.24).

What D2.24 set out to fix
------------------------
D2.22 recorded the trap -- "``converged=True`` does not mean the iterate reached
a minimum; it reports the step collapsing" -- and left it as a warning in the
memory file.  ``BCPE`` supplied the concrete instance that made it impossible to
keep ignoring.  On the parity scanner's own 48-point sample:

    ours   n_cyc=20  dev 94.1535753952  max_iterations_reached
    ours   n_cyc=40  dev 94.1150248919  relative_deviance_converged
    R      n.cyc=20  dev 122.825553739168   converged = TRUE
    R      n.cyc=1000 dev 122.825553739168  converged = TRUE

and at **R's own reported optimum** the eta-scale score is

    dldm = -2.1556e+01   sum(dldm * x) = -4.19   sigma*dldd = -4.8e+01
    dldv = -8.2223       tau*dldt      = -3.8e-09

R stops because the *absolute* deviance change fell under ``c_crit = 0.001``
(0.0006 on the last step: 122.8262 -> 122.8256), not because it arrived
anywhere.  We stop for the mirror-image reason -- ``tau`` is still climbing.
Neither flag was reporting optimality, so this module publishes the quantity
that does: ``max_p ||X_p' W dlogL/deta_p||_inf`` at the point being returned.

Why the chain rule needs the link derivative
--------------------------------------------
The natural-scale ``dld`` is not comparable across parameters.  ``BCPE`` has
``sigma.link = "log"``, ``tau.link = "log"`` and ``nu.link = "identity"``, so
``sum(tau * dldt) = 1e-09`` can coexist with ``sum(dldm) = -21.6``.  Multiplying
by ``dmu/deta`` at ``eta`` is what puts all four on one scale, and it is the
scale RS's own WLS step descends along.

Why this does **not** gate ``converged``
----------------------------------------
The first version of this feature cleared ``converged`` above a ``1e-3``
gradient and emitted a new termination reason.  The threshold was chosen from a
clean-looking separation -- converged families at 8.3e-14 .. 9.4e-05, stuck ones
at 1.0 and 1.6e+09 -- and that separation is an artifact of comparing
magnitudes.  Asking the only question that matters, "does one Fisher step off
this point actually lower the deviance?", refuted it:

    family   eta_gradient   deviance gain
    PO           1.6e-03          1.9e-08   <- no descent
    WEI2         1.2e-02          2.9e-06   <- descends
    GG           2.6e-01          8.1e-05   <- descends
    LOGITNO      2.5e+00          9.1e-13   <- no descent
    BCPE         1.2e+00          9.0e-03   <- descends
    LNO          8.8e+12          4.3e+12   <- descends

PO and WEI2 overlap by a factor of seven, so no cut separates them; gating on
the gradient would have relabelled a converged Poisson fit as stalled.  The
number is therefore reported and the caller decides -- the same shape as the
D2.19 parameter-floor judgement, where the only admissible test for "is this
really the boundary" was to ask what the objective does on the other side.

Note the measurement trap this file also pins: ``family.g_dev_inc`` already
returns the *full* deviance contribution.  Multiplying its sum by 2 -- the
natural reflex when a deviance is defined as ``-2 * logL`` -- doubles every
number and makes a 7.6e-16 Newton step look like it moved the deviance by 168.
That error produced a confident, entirely wrong verdict before it was caught.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss import gamlss
from omnilss.algorithms.rs_algorithm import (
    ETA_GRADIENT_TOL,
    eta_gradient_norms,
)


def _positive_data(n: int = 60, seed: int = 7):
    """Strictly positive response; the Box-Cox families stop R outright otherwise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.abs(rng.normal(3.0, 1.0, n)) + 0.05
    return y, x


def _fit(family: str, y, x, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gamlss(
            "y ~ x",
            family=family,
            data={"y": y, "x": x},
            sigma_formula="~1",
            method="RS",
            **kwargs,
        )


def _slot_norms(model):
    return model.get_slot("eta_gradient_norm_by_param")


# ---------------------------------------------------------------------------
# The measurement itself
# ---------------------------------------------------------------------------


def test_well_converged_fit_reports_a_vanishing_eta_gradient():
    y, x = _positive_data()
    model = _fit("NO", y, x)

    assert model.termination_reason == "relative_deviance_converged"
    assert model.get_slot("eta_gradient_norm") < ETA_GRADIENT_TOL
    # mu is the best-conditioned direction here; asserting on it keeps the test
    # from silently passing because sigma happened to converge harder.
    assert _slot_norms(model)["mu"] < 1e-10


def test_published_slot_matches_a_direct_call_of_the_helper():
    """The slot is the same number, not a re-derivation that can drift.

    Two implementations of one measurement is exactly how D2.20's "the density
    has two implementations" defect happened; the whole point of publishing it
    is that a caller can recompute it.
    """
    y, x = _positive_data()
    model = _fit("BCPE", y, x)

    direct = eta_gradient_norms(
        family=model.family,
        y=y,
        fitted_values={k: np.asarray(v, np.float64) for k, v in model.fitted_values.items()},
        linear_predictors={
            k: np.asarray(v, np.float64) for k, v in model.linear_predictors.items()
        },
        design_matrices={
            k: np.asarray(v, np.float64) for k, v in model.design_matrices.items()
        },
        weights=np.asarray(model.weights, np.float64),
    )

    assert direct == _slot_norms(model)
    assert model.get_slot("eta_gradient_norm") == pytest.approx(
        max(direct.values()), rel=1e-12
    )


def test_every_estimated_parameter_is_measured():
    y, x = _positive_data()
    model = _fit("BCPE", y, x)

    assert set(_slot_norms(model)) == set(model.family.estimable_parameters)


def test_measurement_uses_the_link_derivative_not_the_raw_score():
    """A log link and an identity link must not be compared on raw ``dld``.

    ``BCPE`` has ``sigma.link = "log"``, ``tau.link = "log"`` and
    ``nu.link = "identity"`` -- all three fitted with ``~1``.  If the helper
    forgot ``dmu/deta`` the reported number would be the natural-scale score and
    the ``tau`` entry would come back as ``dldt / tau`` instead of
    ``dldt * tau``; at the point this test builds, the two differ by more than
    six orders of magnitude.
    """
    y, x = _positive_data()
    model = _fit("BCPE", y, x)
    fam = model.family
    fitted = {k: np.asarray(v, np.float64) for k, v in model.fitted_values.items()}
    etas = {k: np.asarray(v, np.float64) for k, v in model.linear_predictors.items()}
    X = np.asarray(model.design_matrices["tau"], np.float64)
    w = np.asarray(model.weights, np.float64)

    context = {k: v for k, v in fitted.items() if k != "tau"}
    raw = np.asarray(
        fam.score_functions["tau"](y=y, tau=fitted["tau"], **context), np.float64
    )
    linked = np.asarray(fam.link_derivatives["tau"](etas["tau"]), np.float64)
    expected = np.abs(X.T @ (w * raw * linked)).max()

    assert _slot_norms(model)["tau"] == pytest.approx(float(expected), rel=1e-12)
    # The natural-scale-only variant is a different number by a wide margin, so
    # this test fails if the chain rule is ever dropped.
    assert not np.isclose(
        _slot_norms(model)["tau"], float(np.abs(X.T @ (w * raw)).max()), rtol=1e-6
    )


def test_an_unmeasurable_score_is_reported_as_nan_not_as_zero():
    """Absence of evidence must not read as evidence of stationarity.

    ``NaN`` is filtered out of the headline ``eta_gradient_norm``, so a family
    whose score refuses to re-evaluate cannot drag the maximum to zero and make
    a broken fit look perfect.
    """
    y, x = _positive_data()
    model = _fit("NO", y, x)

    class _Exploding:
        parameters = ("mu",)
        estimable_parameters = ("mu",)

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        @property
        def score_functions(self):
            def _boom(**_kwargs):
                raise RuntimeError("score unavailable")

            return {"mu": _boom}

    broken = _Exploding(model.family)
    norms = eta_gradient_norms(
        family=broken,
        y=y,
        fitted_values={"mu": np.asarray(model.fitted_values["mu"], np.float64)},
        linear_predictors={"mu": np.asarray(model.linear_predictors["mu"], np.float64)},
        design_matrices={"mu": np.asarray(model.design_matrices["mu"], np.float64)},
        weights=np.ones_like(y),
    )
    assert set(norms) == {"mu"}
    # NaN != NaN, so compare with isnan rather than with == (D2.21).
    assert np.isnan(norms["mu"])
    assert np.isfinite(np.asarray([v for v in norms.values() if v == v], float)).size == 0


# ---------------------------------------------------------------------------
# What it must NOT do
# ---------------------------------------------------------------------------


def test_the_diagnostic_does_not_clear_the_converged_flag():
    """The D2.24 verdict: report, do not gate.

    ``BCPE`` on this data stops with a large eta gradient and a perfectly
    ordinary ``max_iterations_reached``.  What must not happen is the flag being
    silently rewritten by the new measurement -- the first draft did exactly
    that and would have mislabelled converged fits.
    """
    y, x = _positive_data()
    model = _fit("BCPE", y, x)

    assert model.get_slot("eta_gradient_norm") > ETA_GRADIENT_TOL
    assert model.termination_reason == "max_iterations_reached"
    assert model.converged is False


def test_a_good_fit_is_not_relabelled_by_the_diagnostic():
    """The regression that killed the gated version.

    With the ``1e-3`` gate in place this converged Poisson fit was reported as
    ``stalled_at_nonzero_gradient`` even though a Fisher step off its optimum
    lowers the deviance by 1.9e-08 -- i.e. there is nothing left to find.
    """
    y, x = _positive_data()
    model = _fit("PO", y, x)

    assert model.termination_reason == "relative_deviance_converged"
    assert model.converged is True


def test_no_new_termination_reason_is_introduced():
    """The vocabulary is closed: D2.2 owns it.

    A new reason would need a documented meaning and an owner, and D2.24
    concluded it had neither the evidence nor the need for one.
    """
    from omnilss.serialization import load_model_json, save_model_json

    y, x = _positive_data()
    model = _fit("NO", y, x)
    known = {
        "relative_deviance_converged",
        "max_iterations_reached",
        "non_finite_deviance",
        "step_rejected",
        "no_progress_step_rejected",
        "cg_stage_not_converged",
    }
    assert model.termination_reason in known

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.json"
        save_model_json(model, path)
        reloaded = load_model_json(path)
    assert reloaded.termination_reason == model.termination_reason
    assert reloaded.convergence_status() == model.convergence_status()