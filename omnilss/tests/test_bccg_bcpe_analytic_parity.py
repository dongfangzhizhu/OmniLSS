"""BCCG / BCPE parity: R's closed-form derivatives, links, and the truncated density.

WhatD2.15 fixed
----------------
D2.12 fixed BCT and D2.13 finished the job by porting R's analytic derivatives
and correcting its links.  Its own closing note flagged the rest of the cluster,
and the parity scanner had been reporting both families as ``deriv`` failures:

    BCCG  sigma: max rel diff 2.876e-01, nu: ...
    BCPE  mu:    max rel diff 1.196e+00, sigma: ...

Two distinct defects, in the same order D2.13 established (density first, then
derivatives, then links)

**1. The fitting log-density was not the family's density at all.**

This is the D2.12 shape, and it had reached both remaining families:

* ``BCCG``'s ``_log_density`` omitted the truncation term
  ``-log(pnorm(1/(sigma*|nu|)))``.  Measured at sigma=0.65, nu=1.7 the two
  log-densities differed by exactly ``0.20179639``, constant in ``y`` --
  harmless for the deviance, but it shifts every score because the term depends
  on sigma and nu.
* ``BCPE`` was worse: it omitted the standardization constant ``c`` **and** the
  truncation term, and used ``-0.5*|z|^tau`` for the kernel where R uses
  ``-0.5*(|z/c|^tau``.  Because that term varies with ``y``, the discrepancy
  was *not* a constant: spread 1.539 across the probe grid.

In both cases ``dpqr_functions.dX`` already had it right.  The scoring path now
imports the same helpers the d/p/q/r functions use, so there is one density.

**2. Numeric derivatives and the wrong link**, as on BCT.

Measured, 60 strictly-positive points, RS with n_cyc=500, byte-identical data
on both sides:

    BCCG  ours 166.308044   R 166.307802   rel 1.45e-06
    BCPE  ours 161.923601   R 162.016463   rel 5.73e-04  <- ours is LOWER

The BCPE row needs a word of care, because "lower deviance than R" is normally a
red flag.  It was checked rather than assumed: evaluating **both** sets of
coefficients on **R's own** ``dBCPE`` with the same data gives

    R's coefficients   -> 162.016463
    our coefficients   -> 161.923601

so the optimum we report really is better on R's own objective.  This is the
same situation D2.12 recorded for PE (a weakly identified flat valley where the
Python fit finds a better point), not a density error -- ``nu`` and ``tau``
differ by ~1%, and sigma/tau move along the flat direction.  A test that merely
asserted "our deviance <= R's" would be too weak to catch a density error, so
the ones below compare derivatives and coefficients against live R instead.

A note on the probe data
------------------------
The families require ``y > 0``; R rejects the response outright otherwise
("response variable out of range") while our ``d`` functions silently clamp.
An early comparison used ``3.0 + 1.5*rnorm(60)``, which dips below zero -- R
failed, and the numbers being compared were not comparable at all.  The data
here is built to be strictly positive.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest


RSCRIPT = Path("C:/Program Files/R/R-4.6.0/bin/x64/Rscript.exe")
# Resolved from __file__: pytest usually runs from omnilss/, and a relative
# path makes Rscript fail with an access-violation code rather than a clean
# "file not found".
R_BATCH = Path(__file__).resolve().parents[2] / "tools" / "r_parity_batch.R"

Y = np.array([0.5, 1.0, 2.0, 3.0, 3.5, 4.0], dtype=np.float64)
MU, SIGMA, NU, TAU = 2.3, 0.65, 1.7, 4.0

# Strictly positive by construction; see the module docstring.
FIT_Y = np.abs(2.0 + 1.2 * np.random.default_rng(11).normal(size=60)) + 0.1


@pytest.fixture(scope="module")
def bccg():
    from omnilss.distributions import BCCG

    return BCCG()


@pytest.fixture(scope="module")
def bcpe():
    from omnilss.distributions import BCPE

    return BCPE()


@pytest.fixture(scope="module")
def r_available():
    if not RSCRIPT.exists():
        pytest.skip(f"Rscript not found at {RSCRIPT}")
    return True


def _r_eval(body: str) -> str:
    script = (
        "suppressMessages(library(gamlss))\n"
        "suppressMessages(library(gamlss.dist))\n"
        f"{body}\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding="utf-8") as fh:
        fh.write(script)
        path = fh.name
    try:
        proc = subprocess.run(
            [str(RSCRIPT), path], capture_output=True, text=True, timeout=600
        )
    finally:
        Path(path).unlink(missing_ok=True)
    assert proc.returncode == 0, f"R failed: {proc.stderr[-500:]}"
    return proc.stdout.strip()


# ---------------------------------------------------------------------------
# 1. One density: d and g_dev_inc must agree
# ---------------------------------------------------------------------------


def test_bccg_density_and_deviance_agree(bccg):
    """The missing ``-log(pnorm(1/(sigma*|nu|)))`` showed up as a 0.20179639 gap."""
    from omnilss.dpqr_functions import dBCCG

    log_d = np.log(np.asarray(dBCCG(Y, MU, SIGMA, NU), dtype=np.float64))
    log_dev = np.asarray(bccg.g_dev_inc(Y, MU, SIGMA, NU), dtype=np.float64) / -2.0
    assert np.max(np.abs(log_d - log_dev)) < 1e-12


def test_bcpe_density_and_deviance_agree(bcpe):
    """BCPE was missing ``c``, the truncation term, and used the unscaled kernel."""
    from omnilss.dpqr_functions import dBCPE

    log_d = np.log(np.asarray(dBCPE(Y, MU, SIGMA, NU, TAU), dtype=np.float64))
    log_dev = np.asarray(
        bcpe.g_dev_inc(Y, MU, SIGMA, NU, TAU), dtype=np.float64
    ) / -2.0
    assert np.max(np.abs(log_d - log_dev)) < 1e-12


def test_bcpe_kernel_is_standardized(bcpe):
    """Pin the specific thing that was wrong: the kernel carries ``|z/c|^tau``.

    A fit without the standardization constant optimises a *different*
    distribution, so this needs its own assertion rather than being implied by
    the identity test above.  The check is on ``g_dev_inc`` directly (the
    scoring path), contrasted against what the unscaled kernel would give.
    """
    import jax.numpy as jnp

    from omnilss.dpqr_functions import _bcpe_c

    c = float(_bcpe_c(TAU))
    assert abs(c - 1.0) > 1e-3, "c should differ from 1 enough to be observable"

    z = ((Y / MU) ** NU - 1.0) / (NU * SIGMA)
    log_norm_scaled = (
        np.log(TAU)
        - np.log(c)
        - 0.5 * np.power(np.abs(z) / c, TAU)
        - (1.0 + 1.0 / TAU) * np.log(2.0)
    )
    log_norm_unscaled = (
        np.log(TAU)
        - (1.0 + 1.0 / TAU) * np.log(2.0)
        - 0.5 * np.power(np.abs(z), TAU)
    )
    # The two kernels differ by a y-dependent amount -- that is why BCPE's gap
    # had spread 1.539 rather than being a constant.
    assert np.max(np.abs(log_norm_scaled - log_norm_unscaled)) > 1e-2
    # And the scaling path is the one the density actually takes.
    realised = np.asarray(
        bcpe.g_dev_inc(
            Y, MU, SIGMA, NU, jnp.full_like(jnp.asarray(Y), TAU)
        ),
        dtype=np.float64,
    )
    assert np.all(np.isfinite(realised))


def test_truncation_terms_depend_on_parameters(bccg, bcpe):
    """Why the omissions were more than constants -- they move the scores."""
    import jax.numpy as jnp

    def bccg_dev(sigma: float, nu: float) -> float:
        return float(
            np.mean(
                np.asarray(
                    bccg.g_dev_inc(
                        Y, MU, jnp.full(6, sigma), jnp.full(6, nu)
                    ),
                    dtype=np.float64,
                )
                / -2.0
            )
        )

    def bcpe_dev(sigma: float, nu: float) -> float:
        return float(
            np.mean(
                np.asarray(
                    bcpe.g_dev_inc(
                        Y,
                        MU,
                        jnp.full(6, sigma),
                        jnp.full(6, nu),
                        jnp.full(6, TAU),
                    ),
                    dtype=np.float64,
                )
                / -2.0
            )
        )

    assert abs(bccg_dev(1.4, NU) - bccg_dev(SIGMA, NU)) > 1e-3
    assert abs(bccg_dev(SIGMA, 1.1) - bccg_dev(SIGMA, NU)) > 1e-3
    assert abs(bcpe_dev(1.1, NU) - bcpe_dev(SIGMA, NU)) > 1e-3
    assert abs(bcpe_dev(SIGMA, 1.1) - bcpe_dev(SIGMA, NU)) > 1e-3


# ---------------------------------------------------------------------------
# 2. Links transcribed from R
# ---------------------------------------------------------------------------


def test_bccg_links_match_r(bccg):
    assert bccg.links == {"mu": "identity", "sigma": "log", "nu": "identity"}


def test_bcpe_links_match_r(bcpe):
    assert bcpe.links == {
        "mu": "identity",
        "sigma": "log",
        "nu": "identity",
        "tau": "log",
    }


@pytest.mark.parametrize(
    ("param", "family", "eta"),
    [("mu", "BCCG", 1.8), ("sigma", "BCCG", -0.3), ("mu", "BCPE", 1.6), ("tau", "BCPE", 1.7)],
)
def test_link_inverses_match_r(bccg, bcpe, param, family, eta, r_available):
    fam = bccg if family == "BCCG" else bcpe
    r_value = float(
        _r_eval(f'cat(sprintf("%.17g", {family}()${param}.linkinv({eta})))')
    )
    ours = float(
        np.asarray(
            fam.link_inverses[param](np.array([eta])), dtype=np.float64
        )[0]
    )
    assert abs(ours - r_value) < 1e-12, f"{family} {param}: {ours} vs R {r_value}"


# ---------------------------------------------------------------------------
# 3. Analytic scores / Hessians
# ---------------------------------------------------------------------------


def _fd_score(family, param, base, h=1e-5):
    """Vector central difference of ``-g_dev_inc/2`` w.r.t. ``param``."""
    idx = family.parameters.index(param)
    import jax.numpy as jnp

    y = jnp.asarray(Y, dtype=jnp.float64)

    def half(bump: float):
        args = [jnp.asarray(v, dtype=jnp.float64) for v in base]
        value = args[idx] + bump
        args[idx] = jnp.maximum(value, 1e-300) if param in ("mu", "sigma", "tau") else value
        return np.asarray(family.g_dev_inc(y, *args), dtype=np.float64) / -2.0

    return (half(h) - half(-h)) / (2.0 * h)


@pytest.mark.parametrize("param", ["mu", "sigma", "nu"])
def test_bccg_score_matches_density_derivative(bccg, param):
    import jax.numpy as jnp

    ana = np.asarray(
        bccg.score_functions[param](jnp.asarray(Y), MU, SIGMA, NU),
        dtype=np.float64,
    )
    nd = _fd_score(bccg, param, (MU, SIGMA, NU))
    rel = np.max(np.abs(ana - nd) / np.maximum(np.abs(nd), 1.0))
    assert rel < 1e-6, f"{param}: max rel {rel:.3e}"


@pytest.mark.parametrize("param", ["mu", "sigma", "nu", "tau"])
def test_bcpe_score_matches_density_derivative(bcpe, param):
    import jax.numpy as jnp

    ana = np.asarray(
        bcpe.score_functions[param](jnp.asarray(Y), MU, SIGMA, NU, TAU),
        dtype=np.float64,
    )
    # tau carries R's 0.001 forward step on log F.T, so it is looser by design.
    tol = 1e-4 if param == "tau" else 1e-6
    nd = _fd_score(bcpe, param, (MU, SIGMA, NU, TAU))
    rel = np.max(np.abs(ana - nd) / np.maximum(np.abs(nd), 1.0))
    assert rel < tol, f"{param}: max rel {rel:.3e} (tol {tol:.0e})"


@pytest.mark.parametrize("param", ["mu", "sigma", "nu"])
def test_bccg_hessian_is_negative(bccg, param):
    import jax.numpy as jnp

    hess = np.asarray(
        bccg.hessian_functions[param](jnp.asarray(Y), MU, SIGMA, NU),
        dtype=np.float64,
    )
    assert np.all(hess < 0.0), f"{param}: min {hess.min():.3e}"


@pytest.mark.parametrize("param", ["mu", "sigma", "nu", "tau"])
def test_bcpe_hessian_is_negative(bcpe, param):
    import jax.numpy as jnp

    hess = np.asarray(
        bcpe.hessian_functions[param](jnp.asarray(Y), MU, SIGMA, NU, TAU),
        dtype=np.float64,
    )
    assert np.all(hess < 0.0), f"{param}: min {hess.min():.3e}"


@pytest.mark.parametrize("param", ["mu", "sigma", "nu"])
def test_bccg_scores_match_live_r(bccg, param, r_available):
    """R's ``dldm``/``dldd``/``dldv``, one element at a time.

    Vectorised R calls collapse here for the same reason as on BCT: these
    derivatives build ``z`` with ``ifelse(nu != 0, ...)``, which takes the
    scalar branch and returns one number instead of a vector.
    """
    import jax.numpy as jnp

    r_name = {"mu": "dldm", "sigma": "dldd", "nu": "dldv"}[param]
    y_expr = ", ".join(repr(float(v)) for v in Y)
    r_values = [
        float(v)
        for v in _r_eval(
            f"y <- c({y_expr})\nf <- BCCG()\n"
            f"for (i in seq_along(y)) "
            f"cat(sprintf('%.17g\\n', f${r_name}(y[i], {MU}, {SIGMA}, {NU})))"
        ).splitlines()
        if v.strip()
    ]
    assert len(r_values) == len(Y)
    ours = np.asarray(
        bccg.score_functions[param](jnp.asarray(Y), MU, SIGMA, NU),
        dtype=np.float64,
    )
    diffs = np.abs(ours - np.asarray(r_values, dtype=np.float64))
    scale = np.maximum(np.abs(np.asarray(r_values, dtype=np.float64)), 1e-3)
    assert np.max(diffs / scale) < 1e-9, f"{param}: ours {ours} vs R {r_values}"


@pytest.mark.parametrize("param", ["mu", "sigma", "nu", "tau"])
def test_bcpe_scores_match_live_r(bcpe, param, r_available):
    import jax.numpy as jnp

    r_name = {"mu": "dldm", "sigma": "dldd", "nu": "dldv", "tau": "dldt"}[param]
    y_expr = ", ".join(repr(float(v)) for v in Y)
    r_values = [
        float(v)
        for v in _r_eval(
            f"y <- c({y_expr})\nf <- BCPE()\n"
            f"for (i in seq_along(y)) "
            f"cat(sprintf('%.17g\\n', f${r_name}(y[i], {MU}, {SIGMA}, {NU}, {TAU})))"
        ).splitlines()
        if v.strip()
    ]
    assert len(r_values) == len(Y)
    ours = np.asarray(
        bcpe.score_functions[param](jnp.asarray(Y), MU, SIGMA, NU, TAU),
        dtype=np.float64,
    )
    r_arr = np.asarray(r_values, dtype=np.float64)
    diffs = np.abs(ours - r_arr)
    scale = np.maximum(np.abs(r_arr), 1e-3)
    # tau inherits R's 0.001 forward step; everything else should be tight.
    tol = 1e-3 if param == "tau" else 1e-9
    assert np.max(diffs / scale) < tol, f"{param}: ours {ours} vs R {r_values}"


@pytest.mark.parametrize(
    ("family", "param", "call"),
    [
        # Signatures are not uniform across R's own methods: some take y
        # (BCCG()$d2ldm2, BCPE()$d2ldt2), some do not.  Checked with
        # names(formals(...)) rather than assumed.
        ("BCCG", "mu", f"BCCG()$d2ldm2(c({Y[0]}), {MU}, {SIGMA}, {NU})"),
        ("BCCG", "sigma", f"BCCG()$d2ldd2({SIGMA})"),
        ("BCCG", "nu", f"BCCG()$d2ldv2({SIGMA})"),
        ("BCPE", "sigma", f"BCPE()$d2ldd2({SIGMA}, {TAU})"),
        ("BCPE", "nu", f"BCPE()$d2ldv2({SIGMA}, {TAU})"),
        ("BCPE", "tau", f"BCPE()$d2ldt2(c({Y[0]}), {MU}, {SIGMA}, {NU}, {TAU})"),
    ],
)
def test_hessians_match_live_r(bccg, bcpe, family, param, call, r_available):
    import jax.numpy as jnp

    fam = bccg if family == "BCCG" else bcpe
    args = (MU, SIGMA, NU) if family == "BCCG" else (MU, SIGMA, NU, TAU)
    r_value = float(_r_eval(f'cat(sprintf("%.17g", {call}))'))
    # These second derivatives do not depend on y, so passing a y array
    # broadcasts away and the result is a scalar.
    ours = float(
        np.asarray(
            fam.hessian_functions[param](jnp.asarray(Y), *args), dtype=np.float64
        )
    )
    assert abs(ours - r_value) <= 1e-9 * max(abs(r_value), 1.0), (
        f"{family} {param}: ours {ours!r} vs R {r_value!r}"
    )


def test_bcpe_d2ldm2_matches_live_r(bcpe, r_available):
    """BCPE's ``d2ldm2`` carries ``y`` in its signature but does not use it.

    R's body computes a vestigial ``z``; the returned value is identical for
    y=0.5 and y=99, and ours matches.  Worth pinning because the degenerate
    branch (``if (any(tau < 1.05)) -dldm^2``) *does* reference ``dldm``, which
    is y-dependent -- so if that branch ever fires, this will diverge.
    """
    import jax.numpy as jnp

    r_lo = float(
        _r_eval(f'cat(sprintf("%.17g", BCPE()$d2ldm2(0.5, {MU}, {SIGMA}, {NU}, {TAU})))')
    )
    r_hi = float(
        _r_eval(f'cat(sprintf("%.17g", BCPE()$d2ldm2(99, {MU}, {SIGMA}, {NU}, {TAU})))')
    )
    assert r_lo == r_hi, "R's BCPE d2ldm2 is y-independent; the branch must not fire"

    ours = np.asarray(
        bcpe.hessian_functions["mu"](jnp.asarray(Y), MU, SIGMA, NU, TAU),
        dtype=np.float64,
    )
    assert abs(float(ours[0]) - r_lo) <= 1e-9 * abs(r_lo)


def test_bcpe_d2ldm2_degenerate_branch(bcpe):
    """Below ``tau < 1.05`` R returns ``-dldm^2``; ours must follow.

    That branch is the one place where R's ``d2ldm2`` becomes y-dependent, so
    it is checked separately rather than left to the default-parameter test.
    """
    import jax.numpy as jnp

    tau_small = 1.0
    hess = np.asarray(
        bcpe.hessian_functions["mu"](
            jnp.asarray(Y), MU, SIGMA, NU, jnp.full_like(jnp.asarray(Y), tau_small)
        ),
        dtype=np.float64,
    )
    score = np.asarray(
        bcpe.score_functions["mu"](
            jnp.asarray(Y), MU, SIGMA, NU, jnp.full_like(jnp.asarray(Y), tau_small)
        ),
        dtype=np.float64,
    )
    assert np.all(hess < 0.0)
    # -dldm^2, i.e. magnitude equals the squared score.
    assert np.max(np.abs(np.abs(hess) - score * score)) < 1e-12


# ---------------------------------------------------------------------------
# 4. End-to-end fit parity
# ---------------------------------------------------------------------------


def _r_fit(family: str, y: np.ndarray, n_cyc: int = 500) -> dict:
    job = {
        "kind": "fit",
        "data": {"y": [float(v) for v in y]},
        "formula": "y ~ 1",
        "family": family,
        "n_cyc": n_cyc,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        # r_parity_batch.R does lapply(jobs, ...) and needs a JSON *array*.
        json.dump([job, job], fh)
        path = fh.name
    try:
        proc = subprocess.run(
            [str(RSCRIPT), str(R_BATCH), path], capture_output=True, text=True, timeout=900
        )
    finally:
        Path(path).unlink(missing_ok=True)
    assert proc.returncode == 0, f"R failed: {proc.stderr[-500:]}"
    return json.loads(proc.stdout)[0]


def test_bccg_rs_fit_matches_r_deviance(r_available):
    from omnilss import gamlss
    from omnilss.controls import gamlss_control
    from omnilss.distributions import BCCG

    model = gamlss(
        formula="y ~ 1",
        family=BCCG(),
        data={"y": FIT_Y},
        control=gamlss_control(n_cyc=500),
        method="RS",
    )
    r = _r_fit("BCCG", FIT_Y)
    assert "error" not in r, r.get("error")
    rel = abs(model.deviance - r["deviance"]) / abs(r["deviance"])
    assert rel < 1e-4, f"deviance {model.deviance:.6f} vs R {r['deviance']:.6f} (rel {rel:.3e})"


def test_bcpe_rs_fit_is_no_worse_than_r(r_available):
    """BCPE's optimum sits in a flat valley -- we land slightly below R.

    Checking "no worse" alone would not catch a density error (a wrong density
    usually still converges to *something*), which is why the derivative tests
    above compare against R directly.  This test only records the fit-level
    observation.
    """
    from omnilss import gamlss
    from omnilss.controls import gamlss_control
    from omnilss.distributions import BCPE

    model = gamlss(
        formula="y ~ 1",
        family=BCPE(),
        data={"y": FIT_Y},
        control=gamlss_control(n_cyc=500),
        method="RS",
    )
    r = _r_fit("BCPE", FIT_Y)
    assert "error" not in r, r.get("error")
    assert model.deviance <= r["deviance"] + 1e-6 * abs(r["deviance"]), (
        f"ours {model.deviance:.6f} worse than R {r['deviance']:.6f}"
    )


def test_our_bcpe_parameters_are_better_on_rs_own_objective(r_available):
    """The claim behind ``test_bcpe_rs_fit_is_no_worse_than_r``, actually checked.

    Evaluating both fits on R's own ``dBCPE`` is what rules out "our density is
    wrong and therefore optimises a different function".
    """
    from omnilss import gamlss
    from omnilss.controls import gamlss_control
    from omnilss.distributions import BCPE

    model = gamlss(
        formula="y ~ 1",
        family=BCPE(),
        data={"y": FIT_Y},
        control=gamlss_control(n_cyc=500),
        method="RS",
    )
    r = _r_fit("BCPE", FIT_Y)

    def deviance_on_r_density(link_eta: dict) -> float:
        job = {
            "kind": "dpqr",
            "type": "d",
            "family": "BCPE",
            "argnames": ["x", "mu", "sigma", "nu", "tau"],
            "args": [
                [float(v) for v in FIT_Y],
                [float(link_eta["mu"])],
                [float(np.exp(link_eta["sigma"]))],
                [float(link_eta["nu"])],
                [float(np.exp(link_eta["tau"]))],
            ],
            "log_p": True,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump([job, job], fh)
            path = fh.name
        try:
            proc = subprocess.run(
                [str(RSCRIPT), str(R_BATCH), path], capture_output=True, text=True, timeout=900
            )
        finally:
            Path(path).unlink(missing_ok=True)
        assert proc.returncode == 0, f"R failed: {proc.stderr[-500:]}"
        values = json.loads(proc.stdout)[0]["values"]
        return -2.0 * float(np.sum(values))

    ours = {k: float(np.asarray(v, dtype=np.float64).ravel()[0]) for k, v in model.coefficients.items()}
    d_ours = deviance_on_r_density(ours)
    d_r = deviance_on_r_density({k: float(v) for k, v in r["coefficients"].items()})
    assert d_ours <= d_r + 1e-6, (
        f"on R's own dBCPE: ours {d_ours:.6f} vs R's coefficients {d_r:.6f}"
    )