"""ZABI / ZIBI: the binomial zero variants with R's denominator (D2.71).

Both families were built by the generic zero-variant factories, which know
nothing about the binomial denominator.  For a Bernoulli (bd = 1) that is
invisible; for anything else it is not, and the two consequences were both
measured:

* ZIBI could not be *measured* at all.  The parity scan draws its fitting
  sample over the family's response range, which for a bd-less discrete family
  is the generic 0..7 grid -- six values a Bernoulli does not have.  Our
  density answered anyway (y*log(mu) + (1-y)*log(1-mu) is finite for y = 5),
  R's dZIBI refused, and the resulting 64.668 vs 78.270 was read as a density
  disagreement until D2.71.
* R's dZABI/dZIBI take bd exactly like dBI, so dBI(y, bd, mu) with bd > 1 has
  to be expressible.

This module pins the replacement: R's own parameter layout ((mu, nu) plus a
fixed bd), R's own d/p/q formulas, and R's own score transcriptions.  The
scores matter: R uses the observed-information approximation
(d2ldm2 = -dldm^2, d2ldd2 = -dldd^2) and, for ZABI's zero mass, the exact
-1/(nu(1-nu)).

The last tests document a *defect on R's side* rather than ours: neither
dZABI nor dZIBI is elementwise.  Their scalar-zero dBI calls collapse to a
length-one answer (row 1's), which R then recycles -- dZIBI in the zero branch,
dZABI in the hurdle correction of the non-zero branch.  See
tools/rs_family_parity_scan._RSDEV_R_DENSITY_RECYCLED.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

import rs_family_parity_scan as scan  # noqa: E402

from omnilss.distributions import resolve_family  # noqa: E402

RSCRIPT = shutil.which("Rscript")

#: One probe grid, shared by every literal below: bd = 7, mu = 0.37, nu = 0.23.
_X = np.array([0.0, 1.0, 2.0, 3.0, 6.0, 7.0])
_MU = 0.37
_NU = 0.23
_BD = 7.0

#: live R 4.6.0 + gamlss.dist 6.1.1, dZABI/dZIBI(..., bd = 7, mu = 0.37,
#: sigma = 0.23, log = TRUE).
_R_LOG_DENSITY = {
    "ZABI": [
        -1.46967597005894, -2.04173306790429, -1.47533759298349,
        -1.4967287829648, -4.70281713664083, -7.18094409944345,
    ],
    "ZIBI": [
        -1.34580464225819, -2.08191964600231, -1.51552417108151,
        -1.53691536106283, -4.74300371473886, -7.22113067754148,
    ],
}

#: Same call, log = FALSE.
_R_CDF = {
    "ZABI": [
        0.23, 0.359803557452409, 0.588505063439987,
        0.8123663259146, 0.999239050908696, 1.0,
    ],
    "ZIBI": [
        0.260330150921586, 0.385020771376995, 0.604713769322238,
        0.819757180009382, 0.999269024546076, 1.0,
    ],
}

#: qZABI/qZIBI at p = 0.1 .. 0.9 in nine steps.
_R_QUANTILE = {
    "ZABI": [0, 0, 1, 2, 2, 3, 3, 3, 4],
    "ZIBI": [0, 0, 1, 2, 2, 2, 3, 3, 4],
}

#: R's own score transcriptions, evaluated at the same grid through the family
#: object: ZABI()$dldm(...) etc.
_R_SCORES = {
    "ZABI": {
        "dldm": [0.0, -7.27671776337572, -2.98671347337143,
                 1.30329081663286, 14.1733036866457, 18.46330797665],
        "dldd": [4.34782608695652, -1.2987012987013, -1.2987012987013,
                 -1.2987012987013, -1.2987012987013, -1.2987012987013],
        "d2ldd2": [-5.64652738565782] * 6,
    },
    "ZIBI": {
        "dldm": [-1.29451650419094, -6.82110682110682, -2.53110253110253,
                 1.75890175890176, 14.6289146289146, 18.9189189189189],
        "dldd": [3.68996902666751, -1.2987012987013, -1.2987012987013,
                 -1.2987012987013, -1.2987012987013, -1.2987012987013],
        # Not constant: R's d2ldd2 is -dldd^2, and dldd at y == 0 differs.
        "d2ldd2": [-13.6158714177656] + [-1.68662506324844] * 5,
    },
}


def _family(name: str):
    return resolve_family(name)


class TestParameterLayout:
    """R's layout: two estimated parameters plus a fixed denominator."""

    @pytest.mark.parametrize("name", ["ZABI", "ZIBI"])
    def test_bd_is_declared_like_bi(self, name):
        fam = _family(name)
        assert fam.parameters == ("mu", "nu", "bd")
        assert tuple(fam.fixed_parameters) == ("bd",)
        assert fam.fixed_parameter_defaults == {"bd": 1.0}
        assert fam.type == "Discrete"
        assert fam.links["mu"] == "logit"
        assert fam.links["nu"] == "logit"
        assert fam.links["bd"] == "identity"

    @pytest.mark.parametrize("name", ["ZABI", "ZIBI"])
    def test_bd_defaults_to_the_bernoulli(self, name):
        """R's dZABI/dZIBI default bd = 1, and so does ours."""
        fam = _family(name)
        assert fam.fixed_parameter_defaults["bd"] == 1.0
        assert fam.d(1.0, 0.4, 0.2) == pytest.approx(fam.d(1.0, 0.4, 0.2, 1.0))


class TestDensityAgainstLiveR:
    @pytest.mark.parametrize("name", ["ZABI", "ZIBI"])
    def test_log_density_matches_the_r_literals(self, name):
        fam = _family(name)
        ours = np.asarray(fam.d(_X, _MU, _NU, _BD, log=True), dtype=float)
        np.testing.assert_allclose(ours, _R_LOG_DENSITY[name], rtol=1e-13)

    @pytest.mark.parametrize("name", ["ZABI", "ZIBI"])
    def test_cdf_matches_the_r_literals(self, name):
        fam = _family(name)
        ours = np.asarray(fam.p(_X, _MU, _NU, _BD), dtype=float)
        np.testing.assert_allclose(ours, _R_CDF[name], rtol=1e-13)

    @pytest.mark.parametrize("name", ["ZABI", "ZIBI"])
    def test_quantiles_match_the_r_literals(self, name):
        fam = _family(name)
        ours = np.asarray(fam.q(np.linspace(0.1, 0.9, 9), _MU, _NU, _BD))
        np.testing.assert_allclose(ours, _R_QUANTILE[name], atol=0)

    @pytest.mark.parametrize("name", ["ZABI", "ZIBI"])
    def test_outside_the_support_the_mass_is_zero(self, name):
        """R sets fy[x > bd] <- 0 and fy[x < 0] <- 0."""
        fam = _family(name)
        for value in (-1.0, _BD + 1.0, 12.0):
            density = float(np.asarray(fam.d(value, _MU, _NU, _BD)))
            assert density == 0.0, (name, value, density)

    def test_bd_changes_the_density(self):
        """The whole point: bd is not decoration."""
        fam = _family("ZIBI")
        one = float(np.asarray(fam.d(1.0, 0.4, 0.2, 1.0)))
        four = float(np.asarray(fam.d(1.0, 0.4, 0.2, 4.0)))
        assert one == pytest.approx(0.8 * 0.4)
        assert four == pytest.approx(0.8 * 4.0 * 0.4 * 0.6 ** 3)
        assert one != pytest.approx(four)


class TestScoresAgainstLiveR:
    @pytest.mark.parametrize("name", ["ZABI", "ZIBI"])
    @pytest.mark.parametrize("slot", ["dldm", "dldd", "d2ldd2"])
    def test_scores_match_the_r_transcriptions(self, name, slot):
        fam = _family(name)
        if slot == "dldm":
            fn = fam.score_functions["mu"]
        elif slot == "dldd":
            fn = fam.score_functions["nu"]
        else:
            fn = fam.hessian_functions["nu"]
        ours = np.asarray(
            fn(y=_X, mu=np.full_like(_X, _MU),
               nu=np.full_like(_X, _NU), bd=np.full_like(_X, _BD)),
            dtype=float,
        )
        np.testing.assert_allclose(ours, _R_SCORES[name][slot], rtol=1e-12)

    def test_the_zero_mass_hessian_is_exact_for_zabi(self):
        """R's ZABI uses -1/(nu(1-nu)), not -dldd^2."""
        fam = _family("ZABI")
        value = np.asarray(
            fam.hessian_functions["nu"](
                y=jnp.zeros(1), mu=jnp.full(1, 0.4),
                nu=jnp.full(1, 0.23), bd=jnp.ones(1),
            )
        )
        assert float(value[0]) == pytest.approx(-1.0 / (0.23 * 0.77), rel=1e-15)


class TestFit:
    def test_a_legal_sample_fits_and_is_self_consistent(self):
        """bd = 4 counts, our own density, RS: the deviance must equal -2 l."""
        from omnilss import gamlss

        rng = np.random.default_rng(11)
        bd = 4.0
        x = rng.normal(size=48)
        y = rng.binomial(4, 1.0 / (1.0 + np.exp(-(-0.3 + 0.6 * x)))).astype(float)
        data = {"y": y, "x": x, "bd": np.full(48, bd)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = gamlss(
                "y ~ x", family="ZIBI", data=data, sigma_formula="~1",
                method="RS", max_iter=20, allow_compat_alias=True,
            )
        assert np.isfinite(model.deviance)
        fam = _family("ZIBI")
        mu = np.asarray(model.fitted_values["mu"], dtype=float)
        nu = np.asarray(model.fitted_values["nu"], dtype=float)
        total = np.asarray(
            fam.g_dev_inc(y=y, mu=mu, nu=nu, bd=np.full(48, bd)), dtype=float
        )
        assert float(total.sum()) == pytest.approx(model.deviance, rel=1e-12)


class TestTheRReferenceIsDefective:
    """Documentation-by-assertion: the defect is on R's side, not ours."""

    @pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
    def test_r_dzibi_recycles_row_one_over_the_whole_vector(self):
        """R's zero branch is evaluated at mu[1] and recycled to every row.

        dBI is dbinom(x, size, prob); with a scalar x and a vector prob R's
        dbinom returns a **scalar**, so the ifelse in dZIBI pastes row 1's base
        mass onto every y == 0 observation.
        """
        code = (
            "suppressPackageStartupMessages(library(gamlss.dist));"
            "cat(sprintf('%.17g ', dZIBI(c(0,0,0), bd = 1, mu = c(0.1,0.2,0.3),"
            " sigma = 0.3, log = TRUE)));"
            "cat('|');"
            "cat(sprintf('%.17g ', dBI(0, bd = 1, mu = c(0.1,0.2,0.3))));"
            "cat('|');"
            "cat(sprintf('%.17g ', dBI(c(0,0,0), bd = 1, mu = c(0.1,0.2,0.3))))"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(code)
            path = fh.name
        try:
            proc = subprocess.run([RSCRIPT, path], capture_output=True,
                                  text=True, timeout=300)
        finally:
            Path(path).unlink(missing_ok=True)
        assert proc.returncode == 0, proc.stderr[-400:]
        log_density, scalar_dbi, vector_dbi = proc.stdout.split("|")
        r_log = [float(v) for v in log_density.split()]
        assert len(r_log) == 3
        # All three identical -- that is the defect.
        assert max(r_log) - min(r_log) < 1e-15
        # ... and it is row 1's value, because dBI(0, bd, mu) collapsed to one.
        assert len([v for v in scalar_dbi.split() if v]) == 1
        assert len([v for v in vector_dbi.split() if v]) == 3
        expected = np.log(0.3 + 0.7 * (1.0 - 0.1))
        assert r_log[0] == pytest.approx(expected, rel=1e-12)

        # Our density is elementwise: three different values.
        fam = _family("ZIBI")
        ours = np.asarray(
            fam.d(np.zeros(3), np.array([0.1, 0.2, 0.3]),
                  np.full(3, 0.3), np.ones(3), log=True),
            dtype=float,
        )
        assert len(set(np.round(ours, 12))) == 3
        np.testing.assert_allclose(
            ours,
            [-0.0725706928348355, -0.15082288973458377, -0.23572233352106997],
            rtol=1e-13,
        )

    @pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
    def test_r_dzabi_recycles_the_hurdle_correction(self):
        """dZABI's non-zero branch carries the same scalar-dBI collapse.

        The correction is -log(1 - dBI(0, bd[x != 0], mu[x != 0])), and that
        inner dBI has a scalar x = 0, so it answers row 1 of the *subset* for
        every positive observation.  On a bd = 1 sample with near-constant mu
        the recycled value is nearly right, which is why ZABI's rsdev passed
        until the families gained bd support.
        """
        code = (
            "suppressPackageStartupMessages(library(gamlss.dist));"
            "cat(sprintf('%.17g ', dZABI(c(1,2), bd = 5, mu = c(0.2,0.6),"
            " sigma = 0.3, log = TRUE)))"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(code)
            path = fh.name
        try:
            proc = subprocess.run([RSCRIPT, path], capture_output=True,
                                  text=True, timeout=300)
        finally:
            Path(path).unlink(missing_ok=True)
        assert proc.returncode == 0, proc.stderr[-400:]
        r_values = [float(v) for v in proc.stdout.split()]
        np.testing.assert_allclose(
            r_values, [-0.85222828787007, -1.427592432773632], rtol=1e-13
        )

        # R's second value is wrong: it reused mu[1]'s correction.
        assert r_values[1] != pytest.approx(-1.814320504613701, rel=1e-6)

        # Ours is the elementwise density.
        fam = _family("ZABI")
        ours = np.asarray(
            fam.d(np.array([1.0, 2.0]), np.array([0.2, 0.6]),
                  np.full(2, 0.3), np.full(2, 5.0), log=True),
            dtype=float,
        )
        np.testing.assert_allclose(
            ours, [-0.85222828787007, -1.814320504613701], rtol=1e-13
        )

    @pytest.mark.parametrize("handled", ["ZABI", "ZIBI"])
    def test_the_registry_records_the_corrected_reference(self, handled):
        """The gate compares against R's density evaluated elementwise."""
        entry = scan._RSDEV_R_DENSITY_RECYCLED[handled]
        # The recycling moves R's number in whichever direction its branch
        # happens to push: dZIBI's zero branch inflates it (211.998 vs
        # 211.785), dZABI's hurdle correction deflates it (217.123 vs
        # 217.373).  What matters is that it moves, and that we are below the
        # corrected value.
        assert entry["reported"] != pytest.approx(entry["valid_at"], rel=1e-6)
        assert entry["dev_ours"] < entry["valid_at"]
        assert set(scan._RSDEV_R_DENSITY_RECYCLED) >= {"ZABI", "ZIBI"}