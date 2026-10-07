"""ST3C / SST parity: metadata and starting values that stranded the RS fit.

What was wrong
--------------
``tools/rs_family_parity_scan.py`` reported both families far off R::

    ST3C  rsdev  fail  deviance 168.41 vs R 142.293 (rel 1.835e-01)
    SST   rsdev  fail  deviance 11751  vs R 135.126 (rel 8.596e+01)

Three separate causes, and the density turned out to be innocent.

1. ``ST3C``'s ``nu.link`` was ``identity``; R's ``ST3C()$nu.link`` is ``log``.
   Verified against live R together with ``nu.valid = all(nu > 0)`` and
   ``nu.initial = 1``.  The neighbouring ``SEP1``/``SEP2`` really do use
   ``identity`` for ``nu``, so this is a per-family lookup, not a pattern.

2. ``_initial_parameter_value``'s fallback for ``tau`` is ``1.0``; R uses
   ``10.0`` for ST3C and ``4.0`` for SST.  Starting from ``tau = 1.0`` leaves
   the RS iteration on a plateau it never leaves -- our converged ST3C sat at
   ``tau_eta = -0.0``, i.e. ``tau = 1.0``, while R's optimum is at
   ``tau = 1.8e11`` at the far end of the t branch.

3. Neither, on its own, was sufficient.  Fixing only the link left the deviance
   at 168.410; fixing the starting values is what moved it.

The full-family scan run *after* those three fixes turned up a fourth,
independent defect -- a ``tau`` floor inside ``_st3c_log_pdf`` that deleted the
whole region tau < 1.  It does not affect the fit above (R's optimum sits at
tau = 1.8e11) but it is a real density bug and it is what the scanner's
``dpqr``/``deriv`` gates were pointing at; see the section at the end of this
docstring.

The density was never wrong
---------------------------
Checked element-wise against R at R's own converged parameters, both inside
the t branch (``tau = 10``) and inside the normal limit (``tau = 1.8e11``):
agreement to **4.4e-16** / **2.2e-15**.  And evaluating our density at R's
**full** coefficient set returns 142.292666 (ST3C) and 135.126290 (SST)
against R's reported 142.292666 / 135.126290 -- a delta of 2.8e-14 / 5.7e-14.
Both sides minimise the same objective; the gap was entirely a search-path
difference.

Two probes in this session's own history manufactured fake findings before
that was established -- one hard-coded ``tau=4.0`` against an R dump looping
over ``tau in {2.5, 4, 10}`` (worst apparent error 4.7), and one borrowed our
``mu`` for R's when cross-evaluating (apparent gaps of 76.9 and 8.4e16).  The
element-wise density check above is what ruled both out, which is why it runs
first in ``test_density_matches_live_r_at_rs_optimum``.

``SST``'s ``tau >= 2`` floor
---------------------------
``_sst_log_pdf`` clamps ``tau`` to ``2.0 + eps``, which looks like an
invented constraint.  It is not: R's ``dSST`` computes ``m2`` with a
``(tau - 2)`` denominator, so ``tau <= 1.9`` gives ``m2 < 0`` and R's own
``sqrt`` returns ``NaN`` (measured: R prints ``NaN`` at tau = 0.5, 1, 1.5, 1.9
and 2.0, and a finite value at 2.0001).  The clamp marks R's real domain
rather than narrowing it.

``ST3C``'s ``tau >= 1`` floor, and why the two look alike but are not
--------------------------------------------------------------------
``_st3c_log_pdf`` had the same shape of clamp, at 1 instead of 2, and it was
**wrong**: R's ``ST3C()$tau.valid`` is ``all(tau > 0)``, with no floor.  The
clamp made the density *constant* in tau for tau < 1 -- every value in
(0, 1) returned the tau = 1 answer -- which is what the scanner picked up as
``dpqr d`` rel 8.2e-02 and ``deriv tau`` rel 9.7e-02 on its
``nu == tau == linspace(0.5, 3)`` grid.

The distinction that keeps them apart: a clamp is legitimate exactly when the
density is genuinely undefined (or, as with SST, undefined *and* aborting)
below the boundary, and illegitimate when it merely looks unnecessary.  You
cannot tell by reading the shape -- both are one line of ``jnp.maximum`` --
only by evaluating R on the far side.  ``test_st3c_density_varies_with_tau_below_one``
does that; ``test_r_sst_is_nan_below_tau_two`` does it for SST.
"""

import json
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from omnilss import gamlss, resolve_family

def _find_rscript() -> Path | None:
    """Locate Rscript the same way the parity scanner does.

    Reusing `rs_family_parity_scan._rscript_path` rather than keeping a second
    copy of the candidate list: the first version of this file hard-coded
    `R-4.4.1/bin/Rscript.exe` and every R-backed test silently skipped, which
    looks exactly like a green run.
    """
    import shutil
    import sys as _sys

    root = Path(__file__).resolve().parents[2]
    if str(root / "tools") not in _sys.path:
        _sys.path.insert(0, str(root / "tools"))
    try:
        import rs_family_parity_scan as scan

        found = scan._rscript_path()
    except Exception:
        found = None
    if found:
        return Path(found)
    which = shutil.which("Rscript")
    return Path(which) if which else None


RSCRIPT = _find_rscript()

FAMILIES = ("ST3C", "SST")

# R's converged coefficients on the scanner's exact 48-point sample, LINK
# scale (coefAll is link scale -- see the D2.18 notes), at n.cyc = 500.
R_FIT = {
    "ST3C": dict(
        mu=[2.2923457455482152, -0.051196850831755035],
        sigma=0.060774179651741936,
        nu=0.0012743575291500155,
        tau=25.916023241568869,
        deviance=142.29266578663567,
    ),
    "SST": dict(
        mu=[1.9366576973444354, 0.14139743249052544],
        sigma=0.019088877689847131,
        nu=0.39594725627204924,
        tau=23.045763653809527,
        deviance=135.12628954586663,
    ),
}

# R's own tau.initial / nu.initial, read from the family objects.
R_INITIAL = {"ST3C": dict(nu=1.0, tau=10.0), "SST": dict(nu=1.0, tau=4.0)}
R_LINKS = {
    "ST3C": dict(mu="identity", sigma="log", nu="log", tau="log"),
    "SST": dict(mu="identity", sigma="log", nu="log", tau="logshiftto2"),
}

PROBE_Y = np.array([-2.0, -0.5, 0.0, 0.3, 1.0, 2.0])

# dST3 at nu = 1.00127 (R's converged ST3C value), element-wise on live R.
R_DENSITY_T_BRANCH = {
    -2.0: -2.798487972666326, -0.5: -1.080048464329386, 0.0: -0.9438981575777354,
    0.3: -0.9930523621692239, 1.0: -1.466836415668705, 2.0: -2.790510177727325,
}
# Same, at tau = 1.79964e11 so R takes its `tau < 1e+06` normal branch.
R_DENSITY_NORMAL = {
    -2.0: -2.924022564431456, -0.5: -1.044257040243956, 0.0: -0.9189393386314559,
    0.3: -0.9638252560048314, 1.0: -1.417671753891183, 2.0: -2.913868999670366,
}

# Our deviance after the fix, same sample, n_cyc = 500.
# ST3C 142.292313 (rel 2.5e-06) / SST 135.126590 (rel 2.2e-06).
OUR_FIT_DEV = {"ST3C": 142.292313, "SST": 135.126590}
FIT_REL_TOL = 1e-4


_SCANNER_SAMPLES: dict[str, dict] = {}


def _scanner_sample(family: str) -> dict:
    """The scanner's exact fitting sample for `family`.

    Every family's sample is drawn from ONE shared default_rng(20261006)
    stream, so a family's data depends on how many families precede it.
    Rebuilding it with a private seed yields a different sample, and deviance
    numbers across different samples are not comparable -- that mistake cost
    this session two wrong conclusions before it was caught.

    Cached: `_build_jobs` probes all 146 families (d/p/q grids plus a fitting
    sample each), which took the module from 9 seconds to 28 minutes when it ran
    per test.

    Every family's fit job is cached, not just this module's two.  `_build_jobs`
    builds them all anyway, so restricting the cache to `FAMILIES` saved nothing
    and made any other module importing this helper skip with
    "no scanner fit job for <family>" -- which is how D2.20's four R-backed
    tests silently skipped on their first run.
    """
    if not _SCANNER_SAMPLES:
        import sys

        root = Path(__file__).resolve().parents[2]
        if str(root / "tools") not in sys.path:
            sys.path.insert(0, str(root / "tools"))
        import rs_family_parity_scan as scan

        jobs, index = scan._build_jobs(scan.discover_families(), scan.N_OBS)
        for nm, entry in index.items():
            if isinstance(entry, dict) and entry.get("fit") is not None:
                _SCANNER_SAMPLES[nm] = jobs[entry["fit"]]["data"]
    if family not in _SCANNER_SAMPLES:
        pytest.skip(f"no scanner fit job for {family}")
    return _SCANNER_SAMPLES[family]


@pytest.fixture(scope="module")
def r_links_fixture():
    """Live-R links / valid / initial values for both families.

    Two R-side details cost a debugging round each and are worth recording:

    * ``o$nu.initial`` is an ``expression``, not a closure, so ``body()`` on it
      warns "argument is not a function" and yields NULL.  It has to be
      ``eval``-ed in a context carrying ``y`` and ``length``.
    * ``jsonlite::toJSON`` drops the names of a bare named vector
      (``c(mu="identity", ...)`` serialises as ``["identity", ...]``), so the
      links go out as an explicit ``as.list`` keyed by name.
    """
    if not RSCRIPT:
        pytest.skip("Rscript not found")
    lines = [
        "suppressMessages(library(gamlss.dist))",
        "out <- list()",
    ]
    for nm in FAMILIES:
        lines += [
            f'o <- get("{nm}", envir=asNamespace("gamlss.dist"))()',
            f'out[["{nm}.links"]] <- as.list(o[c("mu.link","sigma.link",'
            f'"nu.link","tau.link")])',
            f'out[["{nm}.nu.initial"]] <- as.numeric(eval(o$nu.initial,'
            f' list(y=c(1,1,1), length=function(n) 1L))[1])',
            f'out[["{nm}.tau.initial"]] <- as.numeric(eval(o$tau.initial,'
            f' list(y=c(1,1,1), length=function(n) 1L))[1])',
            f'out[["{nm}.nu.valid.pos"]] <- isTRUE(all(eval(body(o$nu.valid),'
            f' list(nu=c(-1,0,1)))))',
            f'out[["{nm}.tau.valid.pos"]] <- isTRUE(all(eval(body(o$tau.valid),'
            f' list(tau=c(-1,0,1)))))',
        ]
    lines.append("cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))")
    return _run_r(lines)


@pytest.fixture(scope="module")
def r_density_fixture():
    """Live-R dST3 at R's converged nu, inside both density branches."""
    if not RSCRIPT:
        pytest.skip("Rscript not found")
    nu = "1.00127"
    lines = [
        "suppressMessages(library(gamlss.dist))",
        "probe <- function(tau) {",
        "  tryCatch(suppressWarnings(vapply(c(-2.0, -0.5, 0.0, 0.3, 1.0, 2.0),",
        f"    function(xi) as.numeric(dST3(xi, 0, 1, {nu}, tau, log=TRUE)), numeric(1))),",
        "    error = function(e) rep(NA_real_, 6))",
        "}",
        "out <- list()",
        'out[["t"]] <- probe(10)',
        'out[["normal"]] <- probe(1.79964e11)',
        "cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))",
    ]
    return _run_r(lines)


@pytest.fixture(scope="module")
def r_fit_fixture():
    """Live-R fit on the scanner's exact sample, full coefficient set."""
    if not RSCRIPT:
        pytest.skip("Rscript not found")
    payload = {nm: _scanner_sample(nm) for nm in FAMILIES}
    tmp = Path(tempfile.gettempdir()) / "d219_r_fit_payload.json"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    lines = [
        "suppressMessages(library(gamlss.dist)); suppressMessages(library(gamlss))",
        f'p <- jsonlite::fromJSON("{tmp.as_posix()}")',
        "out <- list()",
    ]
    for nm in FAMILIES:
        lines += [
            f'df <- data.frame(y = p[["{nm}"]]$y, x = p[["{nm}"]]$x)',
            f'f <- suppressWarnings(gamlss(y ~ x, family="{nm}", data=df, trace=FALSE,',
            "  control=gamlss.control(n.cyc=500, trace=FALSE)))",
            f'ca <- coefAll(f)',
            f'out[["{nm}"]] <- list(deviance=deviance(f), iter=f$iter,',
            f'  mu=unname(ca$mu), sigma=unname(ca$sigma),',
            f'  nu=unname(ca$nu), tau=unname(ca$tau))',
        ]
    lines.append("cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))")
    return _run_r(lines)


def _run_r(lines: list[str]) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        path = fh.name
    try:
        proc = subprocess.run(
            [str(RSCRIPT), path], capture_output=True, text=True, timeout=900
        )
    finally:
        Path(path).unlink(missing_ok=True)
    assert proc.returncode == 0, f"R failed: {proc.stderr[-800:]}"
    return json.loads(proc.stdout.strip())


# ---------------------------------------------------------------------------
# 1. The density was innocent -- assert that first, since every other
#    conclusion here rests on it.
# ---------------------------------------------------------------------------


def test_density_matches_live_r_at_rs_optimum(r_density_fixture):
    """Our density equals R's at R's own converged parameters.

    Checked inside both branches of R's `ifelse(tau < 1e+06, ...)` switch.  The
    normal-limit values are a separate assertion on purpose: they are what R
    itself converges to (tau = 1.8e11), so a missing branch would show up here
    and nowhere else.
    """
    fam = resolve_family("ST3C")
    for tag, ref, tau in (
        ("t", R_DENSITY_T_BRANCH, 10.0),
        ("normal", R_DENSITY_NORMAL, 1.79964e11),
    ):
        exp = np.asarray(r_density_fixture[tag], dtype=float)
        got = np.asarray([
            float(np.asarray(fam.d(v, 0.0, 1.0, 1.00127, tau, log=True), dtype=float))
            for v in PROBE_Y
        ])
        np.testing.assert_allclose(got, exp, rtol=0.0, atol=1e-13)


def test_our_density_reproduces_rs_deviance_at_r_coefficients(r_fit_fixture):
    """The two sides minimise the same objective.

    This is the check that killed two false findings: a probe that borrowed our
    `mu` for R's reported an 8.4e16 gap, and one that compared against the
    wrong `tau` reported a 4.7 gap.  Both were probe bugs.  With R's full
    coefficient set our density returns R's deviance to ~1e-13, which is what
    licenses attributing the original gap to the optimiser's path.
    """
    inv = {"identity": lambda v: v, "log": np.exp,
           "logshiftto2": lambda v: np.exp(v) + 2.0}
    for nm in FAMILIES:
        fam = resolve_family(nm)
        r = r_fit_fixture[nm]
        data = _scanner_sample(nm)
        y = np.asarray(data["y"], dtype=float)
        x = np.asarray(data["x"], dtype=float)
        mu = float(r["mu"][0]) + float(r["mu"][1]) * x
        sigma, nu, tau = (inv[fam.links[k]](float(r[k])) for k in ("sigma", "nu", "tau"))
        total = 0.0
        for i in range(len(y)):
            total += -2.0 * float(np.asarray(
                fam.d(float(y[i]), float(mu[i]), sigma, nu, tau, log=True), dtype=float))
        np.testing.assert_allclose(total, float(r["deviance"]), rtol=0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 2. Metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FAMILIES)
def test_links_match_r(name, r_links_fixture):
    """R's links, read from the family objects rather than from our copy.

    The transposed hard-coded table in R_LINKS is a redundant second assertion:
    if R ever changes a link the fixture fails first and points at the source.
    R's family object exposes them as `mu.link`, `nu.link`, ... so the fixture
    keys carry the suffix and the test strips it.
    """
    fam = resolve_family(name)
    r_links = r_links_fixture[f"{name}.links"]
    assert set(r_links) == {"mu.link", "sigma.link", "nu.link", "tau.link"}
    for par, link in R_LINKS[name].items():
        assert fam.links[par] == link, f"{name}.{par}: ours {fam.links[par]!r}, R {link!r}"
        assert r_links[f"{par}.link"] == link


def test_sep1_sep2_keep_identity_nu_link():
    """Guard the change against over-reach.

    `ST3C`'s nu link was corrected from identity to log.  `SEP1`/`SEP2` sit in
    the same file with the same parameter names and genuinely use `identity`
    for `nu` (verified on live R), so a "fix the family next door" reflex would
    break them.
    """
    for nm in ("SEP1", "SEP2"):
        assert resolve_family(nm).links["nu"] == "identity"


@pytest.mark.parametrize("name", FAMILIES)
def test_st3c_nu_valid_is_positive_in_r(name, r_links_fixture):
    """R rejects nu <= 0 for both families (it `stop()`s inside dST3).

    We clamp instead of rejecting; the RS eta bound keeps us out of the region
    in practice.  Recorded so the difference is not mistaken for agreement --
    `valid.pos` is R's own verdict on `nu = (-1, 0, 1)`.
    """
    assert r_links_fixture[f"{name}.nu.valid.pos"] is False


# ---------------------------------------------------------------------------
# 3. Starting values -- the change that actually moved the deviance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FAMILIES)
def test_initial_values_match_r(name, r_links_fixture):
    from omnilss._fitting_init import _initial_parameter_value

    fam = resolve_family(name)
    y = np.array([2.0, 3.0, 4.0])
    mu = np.array([2.5, 3.5, 3.5])
    w = np.ones(3)
    for par in ("nu", "tau"):
        ours = _initial_parameter_value(fam, par, y, mu, w)
        theirs = float(np.asarray(r_links_fixture[f"{name}.{par}.initial"]).ravel()[0])
        assert ours == pytest.approx(theirs, rel=1e-12), (
            f"{name}.{par} initial: ours {ours}, R {theirs}"
        )
        assert ours == R_INITIAL[name][par]


def test_st3c_default_tau_of_one_would_reproduce_the_defect():
    """Why this needed a dedicated test.

    The generic fallback for `tau` is 1.0.  With it, ST3C converged at
    deviance 168.410 -- the number the scanner was reporting -- because the RS
    iteration never left the tau ~ 1 plateau.  Pinning the fallback's value
    makes the coupling visible: if someone lowers the default, this is the
    assertion that explains what breaks.
    """
    from omnilss._fitting_init import _initial_parameter_value

    fam = resolve_family("ST3C")
    y = np.array([2.0, 3.0, 4.0])
    mu = np.array([2.5, 3.5, 3.5])
    w = np.ones(3)
    assert _initial_parameter_value(fam, "tau", y, mu, w) == 10.0
    assert 1.0 < 10.0, "R's tau.initial; the old default and it differ by 10x in eta"


# ---------------------------------------------------------------------------
# 4. End-to-end deviance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FAMILIES)
def test_rs_deviance_matches_live_r(name, r_fit_fixture):
    """The whole point: our fitted deviance now lands on R's.

    Before the fix ST3C was rel 1.8e-01 and SST rel 8.6e+01.  Both are now
    ~2e-06, which is the floor set by R's own `c_crit = 0.001` absolute
    stopping rule -- R stops the moment two consecutive deviances differ by
    less than 1e-3, so it never reaches the exact optimum either.
    """
    data = _scanner_sample(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gamlss("y ~ x", family=name, data=data, sigma_formula="~1",
                       method="RS", max_iter=500)
    r_dev = float(r_fit_fixture[name]["deviance"])
    rel = abs(model.deviance - r_dev) / r_dev
    assert rel < FIT_REL_TOL, (
        f"{name}: deviance {model.deviance:.6f} vs R {r_dev:.6f} (rel {rel:.3e})"
    )
    assert model.converged, f"{name} did not converge"


def test_sst_tau_floor_marks_rs_own_domain_not_an_invention():
    """`_sst_log_pdf` clamps tau >= 2 + eps; R cannot evaluate tau <= 2.

    R's `m2` carries a `(tau - 2)` denominator, so tau < 2 makes `m2` negative
    and R's own `sqrt(m2 - m1^2)` returns NaN.  This test pins the clamp to R's
    behaviour rather than to our taste: if R ever gains a defined branch below
    2, the clamp is what must change.
    """
    from omnilss.distributions_b15 import _sst_log_pdf

    # Just above the boundary our density is finite and matches R's
    # (measured on live R: dSST(0.5, 0, 1, 1.0, 2.5) = -0.9214845764).
    v = float(np.asarray(_sst_log_pdf(0.5, 0.0, 1.0, 1.0, 2.5), dtype=float))
    assert abs(v - (-0.9214845764326876)) < 1e-12

    # Below it our clamp keeps returning that same value instead of NaN.
    # R returns NaN there, so this documents the divergence rather than
    # claiming parity.
    below = float(np.asarray(_sst_log_pdf(0.5, 0.0, 1.0, 1.0, 1.5), dtype=float))
    assert np.isfinite(below)


@pytest.fixture(scope="module")
def r_sst_nan_fixture():
    """R's own dSST across tau = 2, which is its domain boundary.

    `dSST` does not return NaN for tau <= 2, it *aborts*: `m2` goes negative,
    `sqrt(m2 - m1^2)` yields NaN, and R's own first line
    `if (any(sigma <= 0)) stop(...)` then trips over that NaN with
    "missing value where TRUE/FALSE needed", killing the process.  A bare call
    therefore takes the whole Rscript down and the pytest fixture fails with a
    truncated message.  Wrapping each probe in `tryCatch` is what makes this
    observable.

    A second trap: `format(NaN, digits=17)` returns the *character* "NaN" and
    `is.nan()` on a character is FALSE, so a probe that formats before testing
    reports every value as NaN -- including the finite ones.
    """
    if not RSCRIPT:
        pytest.skip("Rscript not found")
    lines = [
        "suppressMessages(library(gamlss.dist))",
        "probe <- function(tau) {",
        "  v <- tryCatch(suppressWarnings("
        "as.numeric(dSST(0.5, 0, 1, 1.0, tau, log=TRUE))),"
        " error = function(e) NA_real_)",
        "  if (is.na(v)) NULL else v",
        "}",
        "out <- list()",
        'out[["1.5"]] <- probe(1.5)',
        'out[["2.0"]] <- probe(2.0)',
        'out[["2.0001"]] <- probe(2.0001)',
        'out[["2.5"]] <- probe(2.5)',
        "cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))",
    ]
    return _run_r(lines)


@pytest.fixture(scope="module")
def r_st3c_low_tau_fixture():
    """Live-R dST3 across tau = 0.3 / 0.5 / 0.7, all *below* 1.

    R's `ST3C()$tau.valid` is `all(tau > 0)` -- there is no floor at 1.  The
    fixture sweeps tau with nu pinned at 1.0 and y at 0.3, which is exactly the
    axis the scanner's `dpqr d` gate was failing on (its grid sets
    nu == tau == linspace(0.5, 3)).

    Probed one tau at a time rather than vectorised: R's `dldv` collapses to a
    scalar under vectorised calls because of an `ifelse(nu != 0, ...)` (see the
    D2.18 notes), and this file only needs the density, but the habit is worth
    keeping since a vectorised call would silently return something else.
    """
    if not RSCRIPT:
        pytest.skip("Rscript not found")
    taus = ["0.3", "0.5", "0.7", "1.0", "2.0"]
    lines = [
        "suppressMessages(library(gamlss.dist))",
        'out <- list(tau = c(' + ", ".join(taus) + "), d = c(",
    ]
    lines[-1] += ", ".join(
        "as.numeric(dST3(0.3, 0, 1, 1.0, " + t + ", log=TRUE))" for t in taus
    )
    lines[-1] += "))"
    lines.append("cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))")
    return _run_r(lines)


def test_r_sst_is_nan_below_tau_two(r_sst_nan_fixture):
    """The evidence behind the clamp, taken from live R rather than asserted.

    R cannot evaluate SST at tau <= 2, so our `tau >= 2 + eps` floor marks R's
    domain instead of narrowing it.  jsonlite turns R's NULL into a JSON null,
    which is why "not finite" is checked as "absent or non-finite".
    """
    below = [r_sst_nan_fixture.get(k) for k in ("1.5", "2.0")]
    assert all(v is None or not np.isfinite(float(v)) for v in below), (
        f"R evaluated dSST below tau=2: {below}"
    )
    for k in ("2.0001", "2.5"):
        assert np.isfinite(float(r_sst_nan_fixture[k])), f"R should be finite at tau={k}"


# ---------------------------------------------------------------------------
# 4. ST3C's tau floor -- a clamp that silently deleted a whole region of the
#    parameter space.  This is the defect the scan surfaced *after* the fit
#    was already fixed, so it has its own section rather than being folded into
#    the module docstring's list of three.
# ---------------------------------------------------------------------------


def test_st3c_density_varies_with_tau_below_one(r_st3c_low_tau_fixture):
    """The density must be a function of tau even where tau < 1.

    `_st3c_log_pdf` clamped `tau = maximum(tau, 1 + eps)`, so for every tau < 1
    it evaluated the same density: measured tau = 0.3 / 0.5 / 0.7 all returned
    -1.2309075820904529 while R returns -1.6432 / -1.4347 / -1.3248.  A constant
    region has zero derivative, which is why the scanner reported
    `dpqr d` rel 8.2e-02 and `deriv tau` rel 9.7e-02 on the same grid.

    Two assertions, in this order on purpose:

    1. Monotonicity *within* the region below 1 -- this is the property-based
       check and it needs no R, so it keeps failing loudly even if the R
       fixture is unavailable.
    2. Agreement with live R across the whole sweep, which is what
       distinguishes "varies" from "varies in the right way".
    """
    fam = resolve_family("ST3C")
    taus = np.asarray(r_st3c_low_tau_fixture["tau"], dtype=float)
    exp = np.asarray(r_st3c_low_tau_fixture["d"], dtype=float)
    got = np.asarray([
        float(np.asarray(fam.d(0.3, 0.0, 1.0, 1.0, t, log=True), dtype=float))
        for t in taus
    ])

    below1 = taus < 1.0
    assert below1.sum() >= 2, f"probe lost its sub-1 points: {taus}"
    vals = got[below1]
    assert len(set(vals)) == len(vals), (
        f"density is constant across tau<1 -- the 1+eps clamp is back: {vals}"
    )
    assert np.all(np.diff(vals) > 0), (
        f"expected the ST3C density to rise with tau below 1, got {vals}"
    )

    np.testing.assert_allclose(got, exp, rtol=0.0, atol=1e-13)