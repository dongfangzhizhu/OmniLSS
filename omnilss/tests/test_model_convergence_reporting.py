"""Convergence reporting on the fitted-model container (D2.2).

Background: ``GAMLSSModel`` had no first-class convergence field.  The
information existed only inside ``additional_slots["rs_converged"]`` /
``["cg_converged"]``, so a diverged fit (``g_dev = inf``, ``mu`` clipped at
1e8) was indistinguishable from a converged one at the model level — the
"divergence silently reported as success" defect that D2.2 targets.  R's
``gamlss()`` object exposes ``$converged`` directly.

These tests pin:
- converged fits are reported as converged with a classifiable reason;
- a non-finite deviance is fail-closed (``converged=False`` +
  ``"non_finite_deviance"``), never a success;
- exhausting the iteration budget is reported as ``"max_iterations_reached"``;
- every fitting route (NumPy RS, CG, JAX RS) populates the fields;
- the pre-existing ``additional_slots`` keys stay in sync (backward compat);
- JSON artifact round-trips preserve the outcome.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss import gamlss
from omnilss.serialization import load_model_json, save_model_json

CONVERGED_REASON = "relative_deviance_converged"
MAXED_REASON = "max_iterations_reached"
NONFINITE_REASON = "non_finite_deviance"


def _normal_data(n: int = 120, seed: int = 5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    y = 1.0 + 0.8 * x + rng.normal(scale=0.4, size=n)
    return {"y": y, "x": x}


def _diverging_data(n: int = 120, seed: int = 0):
    """RGE on constant-scale data: the support constraint (x > mu - sigma/nu)
    cannot hold through the RS iterations, so the deviance diverges to inf.

    The two discarded draws reproduce the historical batch-one probe byte for
    byte, which is the dataset that pinned the divergence."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    mu = np.exp(0.5 + 0.8 * x)
    rng.pareto(3.0, n)        # PARETO1 draw (probe order)
    rng.weibull(1.5, n)      # WEI2 draw
    y = mu + rng.gumbel(0.0, 0.3, n)
    return {"y": y, "x": x}


def test_rs_reports_converged_fit():
    model = gamlss("y ~ x", family="NO", data=_normal_data(), method="RS")
    assert model.converged is True
    assert model.termination_reason == CONVERGED_REASON


def test_rs_iteration_budget_exhaustion_is_classified():
    model = gamlss("y ~ x", family="NO", data=_normal_data(), method="RS", max_iter=1)
    assert model.converged is False
    assert model.termination_reason == MAXED_REASON


def test_non_finite_deviance_is_never_reported_as_converged():
    """A non-finite deviance must never be classified as converged.

    D2.154: this used to rely on RGE diverging on _diverging_data().  That fit
    now finishes at a finite 136.4307 (the initial-value / clipping work since
    then stopped the blow-up), so the test was asserting a property of one
    particular trajectory rather than of the classifier.  The classification is
    what matters, so the density is made to return an infinite deviance
    directly -- any trajectory reaching it must be reported as non-finite.
    """
    import dataclasses

    import jax.numpy as jnp

    from omnilss.distributions import NO

    broken = dataclasses.replace(
        NO(), g_dev_inc=lambda **kwargs: jnp.full(jnp.shape(kwargs["y"]), jnp.inf)
    )
    model = gamlss("y ~ x", family=broken, data=_normal_data(), method="RS", max_iter=20)
    assert not np.isfinite(model.g_dev), model.g_dev
    assert model.converged is False
    assert model.termination_reason == NONFINITE_REASON


def test_rs_additional_slots_stay_in_sync():
    model = gamlss("y ~ x", family="NO", data=_normal_data(), method="RS")
    assert model.additional_slots["rs_converged"] is model.converged


def test_cg_reports_convergence_outcome():
    model = gamlss("y ~ x", family="NO", data=_normal_data(), method="CG")
    assert isinstance(model.converged, bool)
    assert model.termination_reason in {
        CONVERGED_REASON, MAXED_REASON, NONFINITE_REASON, "step_rejected",
    }
    assert model.additional_slots["cg_converged"] is model.converged


def test_rs_jax_reports_convergence_outcome():
    model = gamlss("y ~ x", family="NO", data=_normal_data(), method="RS_JAX")
    assert isinstance(model.converged, bool)
    assert model.termination_reason in {
        CONVERGED_REASON, MAXED_REASON, NONFINITE_REASON,
    }


def test_json_artifact_roundtrip_preserves_outcome(tmp_path):
    model = gamlss("y ~ x", family="NO", data=_normal_data(), method="RS")
    path = tmp_path / "model.json"
    save_model_json(model, path)
    loaded = load_model_json(path)
    assert loaded.converged == model.converged
    assert loaded.termination_reason == model.termination_reason


def test_legacy_artifact_without_convergence_fields_loads_as_unknown(tmp_path):
    """Artifacts written before the first-class fields must stay loadable and
    must not claim success."""
    import json
    import zipfile

    model = gamlss("y ~ x", family="NO", data=_normal_data(), method="RS")
    path = tmp_path / "model.json"
    save_model_json(model, path)

    with zipfile.ZipFile(path) as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}
    meta = json.loads(entries["meta.json"].decode("utf-8"))
    meta.pop("converged", None)
    meta.pop("termination_reason", None)
    entries["meta.json"] = json.dumps(meta).encode("utf-8")

    legacy_path = tmp_path / "legacy.json"
    with zipfile.ZipFile(legacy_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)

    loaded = load_model_json(legacy_path)
    assert loaded.converged is None
    assert loaded.termination_reason is None


def test_convergence_status_falls_back_to_route_slots():
    """A model that only carries the per-route slot keys must still answer."""
    from omnilss.model import GAMLSSModel

    model = gamlss("y ~ x", family="NO", data=_normal_data(), method="RS")
    bare = GAMLSSModel(
        par=model.par,
        family=model.family,
        df_fit=model.df_fit,
        g_dev=model.g_dev,
        n=model.n,
        y=model.y,
        additional_slots={"rs_converged": True},
    )
    assert bare.converged is None
    assert bare.convergence_status() is True

    unknown = GAMLSSModel(
        par=model.par,
        family=model.family,
        df_fit=model.df_fit,
        g_dev=model.g_dev,
        n=model.n,
        y=model.y,
    )
    assert unknown.convergence_status() is None


def test_model_selection_reports_real_convergence_flag():
    """compare_distributions used to read additional_slots["converged"], which
    the RS route never writes, so every fit was reported as not converged."""
    from omnilss.model_selection import compare_distributions

    np.random.seed(42)
    n = 50
    x = np.random.randn(n)
    y = 2 + 3 * x + np.random.randn(n)
    results = compare_distributions(
        "y ~ x", families=["NO", "GA"], data={"y": y, "x": x}, criterion="AIC"
    )
    assert len(results) >= 1
    # NO converges; GA does not reach the 20-iteration budget on this dataset
    # (R rejects it outright — "response variable out of range" — because some
    # responses are negative).  The point of the check is that the flag is now
    # read from the fit itself instead of defaulting to False.
    assert bool(results["converged"].any()) is True
    assert set(results["converged"].map(bool)) <= {True, False}


def test_mixed_route_reports_convergence():
    np.random.seed(7)
    n = 80
    x = np.random.randn(n)
    y = 1.0 + 0.5 * x + np.random.normal(scale=0.5, size=n)
    model = gamlss(
        "y ~ x",
        family="NO",
        data={"y": y, "x": x},
        method="MIXED",
        mixed_n1=1,
        mixed_n2=5,
    )
    assert isinstance(model.converged, bool)
    assert model.termination_reason is not None
    assert model.convergence_status() is model.converged
