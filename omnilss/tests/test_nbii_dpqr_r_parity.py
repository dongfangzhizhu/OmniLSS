"""Live-R parity tests for the NBII dpqr module functions (D2.49).

Why this file exists
--------------------
Three defects met in the NBII dpqr surface, all found while repairing the
empty ``rNBII`` body (which made ``family.r`` return ``None`` and reddened
``test_dpqr_comprehensive.py``):

1. the module-level ``dNBII`` used ``size = 1/(sigma*mu)`` instead of R's
   ``size = mu/sigma`` -- a different distribution whenever ``mu != 1``
   (the family's own ``d`` slot was correct, so the dpqr gate -- which
   probes the family slot -- never saw it);
2. ``pNBII`` applied its ``q < 0 -> 0`` mask *before* the
   ``lower.tail``/``log.p`` flip, while R applies it afterwards -- so the
   upper tail at ``q < 0`` answered 1 instead of 0 and ``log.p`` answered
   ``-inf`` instead of 0 (the gate grid is all-positive, so this was
   invisible too);
3. R's ``sigma < 1e-04 -> Poisson`` switch (present in d/p/q/r and hence in
   ``G.dev.incr``, which is ``-2*dNBII(..., log=TRUE)``) was missing.

The probe below dumps R's answers over both sides of the branch threshold
(``sigma`` in {2, 0.5, 1e-04, 9.999e-05, 1e-06}) plus negative quantiles,
and the sampler's distributional reference.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys  # noqa: E402

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import jax  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

from omnilss.dpqr_functions import dNBII, pNBII, qNBII, rNBII  # noqa: E402

RSCRIPT = shutil.which("Rscript")

#: (mu, sigma) probe grid.  The sigma values straddle R's Poisson-switch
#: threshold: 1e-04 itself still takes the nbinom path (R's test is a
#: strict ``<``), 9.999e-05 and 1e-06 take the Poisson path.
MUS = (0.5, 1.5, 2.0)
SIGMAS = (2.0, 0.5, 1e-04, 9.999e-05, 1e-06)
X_GRID = (-1.0, 0.0, 1.0, 2.0, 5.0, 10.0)
P_GRID = (0.01, 0.1, 0.5, 0.9, 0.99)

#: Measured agreement floors against the digits=17 probe.  d: 3.7e-11
#: absolute, all of it at sigma=1e-04 where R's own dnbinom runs at
#: size = mu/1e-04 ~ 1e4 and carries that error itself; elsewhere 1e-15.
#: p family: 9e-14 (same boundary).  q: exact (both sides are integer
#: quantiles of the same nbinom).  r: two-sample Kolmogorov-Smirnov
#: D <= 0.021 at n=2000 with both seeds fixed.
RTOL_D = 1e-9
ATOL_D = 1e-10
ATOL_P = 1e-11
#: Two-sample Kolmogorov-Smirnov critical value, two-sided alpha=0.05 at
#: n=m=2000 (1.36/sqrt(n_eff)).
KS_CRITICAL = 0.031


def _r_vector(grid):
    """Render a numeric grid as R ``c(...)`` code with plain float literals.

    numpy 2.x renders ``np.float64(0.01)`` inside f-strings, which R cannot
    parse; ``repr(float(v))`` round-trips the double as a plain literal.
    """
    return "c(" + ", ".join(repr(float(v)) for v in grid) + ")"


def _r_probe_script():
    """R program dumping the NBII dpqr/r contract as one JSON blob."""
    mus = ", ".join(repr(float(v)) for v in MUS)
    sigmas = ", ".join(repr(float(v)) for v in SIGMAS)
    return f"""
suppressPackageStartupMessages({{library(gamlss); library(gamlss.dist); library(jsonlite)}})
out <- list()
mus <- c({mus})
sigmas <- c({sigmas})
xs <- {_r_vector(X_GRID)}
ps <- {_r_vector(P_GRID)}
rows <- list()
for (mu in mus) for (sg in sigmas) {{
  rows[[length(rows)+1]] <- list(
    mu = mu, sigma = sg, x = xs,
    d = dNBII(xs, mu = mu, sigma = sg),
    dlog = dNBII(xs, mu = mu, sigma = sg, log = TRUE),
    p = pNBII(xs, mu = mu, sigma = sg),
    pupper = pNBII(xs, mu = mu, sigma = sg, lower.tail = FALSE),
    plog = pNBII(xs, mu = mu, sigma = sg, log.p = TRUE),
    q = qNBII(ps, mu = mu, sigma = sg),
    qupper = qNBII(ps, mu = mu, sigma = sg, lower.tail = FALSE),
    # R validates p in [0,1] BEFORE converting log.p, so qNBII's log.p
    # mode errors out for every sensible probability; record the message.
    qlog = tryCatch(qNBII(log(ps), mu = mu, sigma = sg, log.p = TRUE),
                    error = function(e) paste("ERR:", conditionMessage(e)))
  )
}}
out$dpqr <- rows
set.seed(20261020)
rs <- list()
for (mu in c(2.0, 5.0)) for (sg in c(0.5, 2.0)) {{
  s <- rNBII(50000, mu = mu, sigma = sg)
  rs[[length(rs)+1]] <- list(mu = mu, sigma = sg,
                             mean = mean(s), var = var(s),
                             samples = s[1:2000])
}}
out$r <- rs
cat(toJSON(out, auto_unbox = TRUE, digits = 17, na = "null"))
"""


def _run_r(lines, timeout=600):
    """Run an R script, return stdout; skip the test when R is absent."""
    if RSCRIPT is None:  # pragma: no cover - CI without R
        pytest.skip("Rscript not on PATH")
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False,
                                     encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        path = fh.name
    try:
        proc = subprocess.run([RSCRIPT, path], capture_output=True,
                              text=True, timeout=timeout)
    finally:
        Path(path).unlink(missing_ok=True)
    assert proc.returncode == 0, f"R failed: {proc.stderr[-800:]}"
    return proc.stdout


@pytest.fixture(scope="module")
def r_probe():
    """Live-R contract dump; the whole module compares against it."""
    stdout = _run_r(_r_probe_script().splitlines())
    return json.loads(stdout)


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestDAgainstLiveR:
    def test_d_matches_live_r(self, r_probe):
        for row in r_probe["dpqr"]:
            x = np.asarray(row["x"], dtype=float)
            ours = np.asarray(dNBII(x, row["mu"], row["sigma"]), dtype=float)
            r = np.asarray(row["d"], dtype=float)
            np.testing.assert_allclose(
                ours, r, rtol=RTOL_D, atol=ATOL_D,
                err_msg=f"dNBII at mu={row['mu']}, sigma={row['sigma']}",
            )

    def test_d_log_matches_live_r(self, r_probe):
        for row in r_probe["dpqr"]:
            x = np.asarray(row["x"], dtype=float)
            ours = np.asarray(
                dNBII(x, row["mu"], row["sigma"], log=True), dtype=float
            )
            r = np.asarray(row["dlog"], dtype=float)
            np.testing.assert_allclose(
                ours, r, rtol=RTOL_D, atol=ATOL_D,
                err_msg=f"dNBII(log) at mu={row['mu']}, sigma={row['sigma']}",
            )

    def test_negative_x_is_zero_in_both_modes(self, r_probe):
        """R's trailing ifelse runs after the log branch: 0, not -inf."""
        for row in r_probe["dpqr"]:
            mu, sg = row["mu"], row["sigma"]
            ours = np.asarray(dNBII(-1.0, mu, sg), dtype=float)
            ours_log = np.asarray(dNBII(-1.0, mu, sg, log=True), dtype=float)
            r = float(np.asarray(row["d"], dtype=float)[0])
            r_log = float(np.asarray(row["dlog"], dtype=float)[0])
            assert ours == r == 0.0, (mu, sg, ours, r)
            assert ours_log == r_log == 0.0, (mu, sg, ours_log, r_log)


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestPQAgainstLiveR:
    def test_p_matches_live_r(self, r_probe):
        for row in r_probe["dpqr"]:
            x = np.asarray(row["x"], dtype=float)
            ours = np.asarray(pNBII(x, row["mu"], row["sigma"]), dtype=float)
            r = np.asarray(row["p"], dtype=float)
            np.testing.assert_allclose(
                ours, r, rtol=1e-12, atol=ATOL_P,
                err_msg=f"pNBII at mu={row['mu']}, sigma={row['sigma']}",
            )

    def test_p_upper_and_log_match_live_r(self, r_probe):
        for row in r_probe["dpqr"]:
            x = np.asarray(row["x"], dtype=float)
            mu, sg = row["mu"], row["sigma"]
            ours_up = np.asarray(
                pNBII(x, mu, sg, lower_tail=False), dtype=float
            )
            ours_log = np.asarray(pNBII(x, mu, sg, log_p=True), dtype=float)
            r_up = np.asarray(row["pupper"], dtype=float)
            r_log = np.asarray(row["plog"], dtype=float)
            np.testing.assert_allclose(
                ours_up, r_up, rtol=1e-12, atol=ATOL_P,
                err_msg=f"pNBII(upper) at mu={mu}, sigma={sg}",
            )
            np.testing.assert_allclose(
                ours_log, r_log, rtol=1e-12, atol=ATOL_P,
                err_msg=f"pNBII(log.p) at mu={mu}, sigma={sg}",
            )

    def test_negative_q_masked_after_the_flip(self, r_probe):
        """R masks q<0 to 0 *after* lower.tail/log.p, so the upper tail at
        q<0 is 0 (not 1) and log.p at q<0 is 0 (not -inf)."""
        for row in r_probe["dpqr"]:
            mu, sg = row["mu"], row["sigma"]
            ours_up = float(np.asarray(pNBII(-1.0, mu, sg, lower_tail=False)))
            ours_log = float(np.asarray(pNBII(-1.0, mu, sg, log_p=True)))
            r_up = float(np.asarray(row["pupper"], dtype=float)[0])
            r_log = float(np.asarray(row["plog"], dtype=float)[0])
            assert ours_up == r_up == 0.0, (mu, sg, ours_up, r_up)
            assert ours_log == r_log == 0.0, (mu, sg, ours_log, r_log)


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestQAgainstLiveR:
    def test_q_matches_live_r(self, r_probe):
        for row in r_probe["dpqr"]:
            p = np.asarray(P_GRID, dtype=float)
            ours = np.asarray(qNBII(p, row["mu"], row["sigma"]), dtype=float)
            r = np.asarray(row["q"], dtype=float)
            np.testing.assert_array_equal(
                ours, r,
                err_msg=f"qNBII at mu={row['mu']}, sigma={row['sigma']}",
            )

    def test_q_upper_matches_live_r(self, r_probe):
        for row in r_probe["dpqr"]:
            p = np.asarray(P_GRID, dtype=float)
            ours = np.asarray(
                qNBII(p, row["mu"], row["sigma"], lower_tail=False),
                dtype=float,
            )
            r = np.asarray(row["qupper"], dtype=float)
            np.testing.assert_array_equal(
                ours, r,
                err_msg=f"qNBII(upper) at mu={row['mu']}, sigma={row['sigma']}",
            )

    def test_q_log_p_divergence_is_r_s_own_guard(self, r_probe):
        """R's qNBII validates p in [0,1] before converting log.p, so its
        log.p mode rejects every probability below 1; ours converts first
        (the convention the generalised-inverse-Gaussian quantile follows).
        Pin both halves of that contract."""
        for row in r_probe["dpqr"]:
            r_log = row["qlog"]
            assert isinstance(r_log, str) and r_log.startswith("ERR:"), (
                f"R stopped rejecting log-scale p at mu={row['mu']}, "
                f"sigma={row['sigma']}: {r_log}"
            )
            mu, sg = row["mu"], row["sigma"]
            p = np.asarray(P_GRID, dtype=float)
            ours_log = np.asarray(
                qNBII(np.log(p), mu, sg, log_p=True), dtype=float
            )
            ours_plain = np.asarray(qNBII(p, mu, sg), dtype=float)
            np.testing.assert_array_equal(ours_log, ours_plain)


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestRAgainstLiveR:
    def test_sampler_matches_r_distributionally(self, r_probe):
        """R's rNBII is quantile inversion; ours is the gamma-Poisson
        mixture (the convention the type-I sibling follows).  Both seeds
        fixed, so the Kolmogorov-Smirnov statistic is deterministic:
        measured D <= 0.021 over the grid.
        """
        for row in r_probe["r"]:
            mu, sg = row["mu"], row["sigma"]
            r_samples = np.asarray(row["samples"], dtype=float)
            ours = np.asarray(
                rNBII(jax.random.PRNGKey(20261020), 2000, mu, sg), dtype=float
            )
            ks = ks_2samp(ours, r_samples)
            assert ks.statistic <= KS_CRITICAL, (
                f"rNBII({mu}, {sg}): Kolmogorov-Smirnov D={ks.statistic:.4f} "
                f"exceeds the alpha=0.05 critical value {KS_CRITICAL}"
            )
            big = np.asarray(
                rNBII(jax.random.PRNGKey(7), 200000, mu, sg), dtype=float
            )
            # NBII moments: E[Y] = mu, Var(Y) = mu*(1 + sigma).  The
            # tolerance is the sampling noise, not a parity slack: the sd of
            # the sample variance at n=2e5 is sqrt((mu4 - Var^2)/n) with
            # mu4 = 86 for the (2, 0.5) profile -- 0.0196 absolute, so
            # rtol 2e-2 is ~3 sd.
            np.testing.assert_allclose(
                big.mean(), mu, rtol=2e-2,
                err_msg=f"rNBII({mu}, {sg}) mean",
            )
            np.testing.assert_allclose(
                big.var(ddof=1), mu * (1.0 + sg), rtol=2e-2,
                err_msg=f"rNBII({mu}, {sg}) variance",
            )
