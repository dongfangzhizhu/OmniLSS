import numpy as np

from omnilss.algorithms.backend_protocol import IWLSState
from omnilss.algorithms.rs_algorithm import JaxSolverBackend, NumpySolverBackend, rs_fit
from omnilss.algorithms.warm_start_policy import FamilyHeuristicInit, NoWarmStart
from omnilss.controls import glim_control
from omnilss.distributions import resolve_family


def test_no_warm_start_initializes_constant_vectors():
    policy = NoWarmStart()
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    result = policy.initialize(y=y, parameter_names=("mu", "sigma"))

    assert result.metadata["policy"] == "none"
    assert np.allclose(result.initial_eta["mu"], 0.0)
    assert np.allclose(result.initial_fitted["mu"], 1.0)
    assert np.allclose(result.initial_fitted["sigma"], 1.0)


def test_family_heuristic_initializes_mu_sigma_with_data_statistics():
    policy = FamilyHeuristicInit()
    y = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)

    result = policy.initialize(y=y, parameter_names=("mu", "sigma", "nu"))

    assert result.metadata["policy"] == "family_heuristic"
    assert np.allclose(result.initial_fitted["mu"], np.mean(y))
    assert np.all(result.initial_fitted["sigma"] > 0)
    assert np.allclose(result.initial_eta["nu"], 0.0)


def test_glim_control_defaults_to_r_compatible_cycles():
    # Live R (gamlss 5.5-0) reports glim.control()$cyc == 50 (D2.36); commit
    # 3eaae3d had restored the then-believed R default of 20, which D2.36's
    # live-R check corrected.  The previous local default before that was 1.
    control = glim_control()
    assert control.cyc == 50

    fallback = glim_control(cyc=0)
    assert fallback.cyc == 50


def test_numpy_solver_backend_exposes_expected_operations():
    family = resolve_family("NO")
    backend = NumpySolverBackend(family)
    n = 8
    x = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
    y = np.linspace(0.5, 2.5, n)
    mu = np.full(n, np.mean(y))
    sigma = np.full(n, np.std(y) + 1e-3)
    eta = np.zeros(n)
    link_derivative = np.ones(n)

    score, hessian = backend.compute_score_hessian(
        parameter="mu",
        y=y,
        fitted=mu,
        context={"sigma": sigma},
    )
    z, w = backend.iwls_step(
        state=IWLSState(design=x, eta=eta, fitted=mu, observation_weights=np.ones(n)),
        score=score,
        hessian=hessian,
        link_derivative=link_derivative,
        offset=np.zeros(n),
    )
    beta = backend.wls_solve(design=x, working_response=z, weights=w)

    assert score.shape == (n,)
    assert hessian.shape == (n,)
    assert z.shape == (n,)
    assert w.shape == (n,)
    assert beta.shape == (x.shape[1],)


def test_jax_solver_backend_exposes_expected_operations():
    family = resolve_family("NO")
    backend = JaxSolverBackend(family)
    n = 8
    x = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
    y = np.linspace(0.5, 2.5, n)
    mu = np.full(n, np.mean(y))
    sigma = np.full(n, np.std(y) + 1e-3)
    eta = np.zeros(n)
    link_derivative = np.ones(n)

    score, hessian = backend.compute_score_hessian(
        parameter="mu",
        y=y,
        fitted=mu,
        context={"sigma": sigma},
    )
    z, w = backend.iwls_step(
        state=IWLSState(design=x, eta=eta, fitted=mu, observation_weights=np.ones(n)),
        score=score,
        hessian=hessian,
        link_derivative=link_derivative,
        offset=np.zeros(n),
    )
    beta = backend.wls_solve(design=x, working_response=z, weights=w)

    assert score.shape == (n,)
    assert hessian.shape == (n,)
    assert z.shape == (n,)
    assert w.shape == (n,)
    assert beta.shape == (x.shape[1],)


def test_rs_fit_records_selected_solver_backend():
    x = np.linspace(0.0, 1.0, 24)
    y = 2.0 + 0.7 * x
    model = rs_fit(
        formula="y ~ x",
        family="NO",
        data={"x": x, "y": y},
        routing_decision={"solver_backend": "jax"},
    )
    assert model.additional_slots["solver_backend"] == "jax"


def test_rs_fit_emits_timing_split_keys():
    x = np.linspace(0.0, 1.0, 32)
    y = 1.0 + 0.5 * x
    model = rs_fit(
        formula="y ~ x",
        family="NO",
        data={"x": x, "y": y},
        routing_decision={"solver_backend": "numpy"},
    )
    split = model.additional_slots["timing_split"]
    assert set(split.keys()) == {"warm_start", "core_rs", "host_device_sync"}
    assert split["warm_start"] >= 0.0
    assert split["core_rs"] >= 0.0


def test_rs_fit_records_warm_start_metadata():
    x = np.linspace(0.0, 1.0, 20)
    y = 3.0 + x
    model = rs_fit(
        formula="y ~ x",
        family="NO",
        data={"x": x, "y": y},
    )
    assert model.additional_slots["warm_start_policy"] == "family_heuristic"
    assert "y_mean" in model.additional_slots["warm_start_metadata"]


def test_rs_fit_accepts_explicit_no_warm_start_policy():
    x = np.linspace(0.0, 1.0, 16)
    y = 2.0 + x
    model = rs_fit(
        formula="y ~ x",
        family="NO",
        data={"x": x, "y": y},
        warm_start_policy=NoWarmStart(),
    )
    assert model.additional_slots["warm_start_policy"] == "none"
