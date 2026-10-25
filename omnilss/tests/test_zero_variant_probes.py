"""Probe reachability for the zero-variant families (D2.70/D2.71).

ZABI/ZIBI/ZAZIPF skipped their dpqr gate with "unused argument (nu = ...)":
live R names the hurdle mass "sigma" in those three families (their base has
no dispersion of its own, so R reuses the sigma slot) while our factories
name it "nu".  A translation table in the scanner closes the naming gap.

D2.70 left ZIBI out of that table because its gate compared 64.668 against
R's 78.270 with no way to tell which side was wrong.  D2.71 found the answer,
and it was neither side's *density*: the fitting sample was illegal.  The
families were built on the bd-less binomial (a Bernoulli), the generic
discrete probe grid is 0..7, and R's dZABI/dZIBI reject y > bd.  Both families
now take R's binomial denominator as a fixed parameter, the probe passes
bd = 10, and ZABI/ZIBI measure deriv and dpqr cleanly.

ZIBI's rsdev is still registered as a warning, but for a *different* reason
than in D2.70 -- see tools/rs_family_parity_scan._RSDEV_R_DENSITY_RECYCLED
and omnilss/tests/test_binomial_zero_variants_bd.py.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

import rs_family_parity_scan as scan  # noqa: E402

from omnilss.distributions import resolve_family  # noqa: E402

RSCRIPT = shutil.which("Rscript")


class TestNamingMap:
    def test_zero_mass_is_translated_for_the_r_probe(self):
        assert scan._R_ARGNAME_MAP[("ZABI", "nu")] == "sigma"
        assert scan._R_ARGNAME_MAP[("ZAZIPF", "nu")] == "sigma"
        assert scan._R_ARGNAME_MAP[("ZIBI", "nu")] == "sigma"

    def test_job_argnames_use_the_r_name(self):
        jobs, _ = scan._build_jobs(["ZABI", "ZIBI", "ZAZIPF"], 8)
        dps = [j for j in jobs if j["kind"] == "dpqr"]
        assert dps
        for job in dps:
            assert "nu" not in job["argnames"]
            assert job["argnames"].count("sigma") == 1
            assert len(job["argnames"]) == len(job["args"])

    def test_probe_grid_covers_the_binomial_denominator(self):
        """bd is a probe constant, so the legal response runs to bd itself."""
        for family in ("ZABI", "ZIBI"):
            assert scan._SUPPORT_BOUNDS.get(family) == (0, int(scan._PROBE_BD))
            cal = scan.calibrate(family, 8)
            assert cal.ok, family
            assert cal.bd == scan._PROBE_BD
            assert float(np.max(cal.y)) <= scan._PROBE_BD


class TestDensityMatchesR:
    @pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
    def test_zazipf_density_equals_live_r_under_the_translation(self):
        """(mu, nu) with our name == (mu, sigma) with R's name, bit for bit."""
        fam = resolve_family("ZAZIPF")
        ys = [0.0, 1.0, 2.0, 3.0, 5.0]
        ours = np.asarray(fam.d(np.asarray(ys), 0.4, 0.3), dtype=float)

        import subprocess
        import tempfile

        code = (
            "suppressPackageStartupMessages(library(gamlss.dist));"
            "cat(sprintf('%.17g', dZAZIPF(c(0,1,2,3,5), mu = 0.4, sigma = 0.3)))"
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
        expected = np.asarray([float(v) for v in proc.stdout.split()])
        np.testing.assert_allclose(ours, expected, rtol=1e-12)


class TestGates:
    @pytest.mark.parametrize("family", ["ZABI", "ZIBI", "ZAZIPF"])
    def test_derivative_gate_passes(self, family):
        row = scan.gate_derivatives(family)
        assert row["status"] == "pass", row

    @pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
    @pytest.mark.parametrize("family", ["ZABI", "ZIBI"])
    def test_dpqr_gate_passes(self, family):
        """Both families were unmeasurable before D2.71 (skip, then fail)."""
        jobs, index = scan._build_jobs([family], 8)
        results = scan._run_r(jobs)
        row = scan.gate_dpqr(family, results, index)
        assert row["status"] == "pass", row

    def test_the_binomial_denominator_is_a_declared_fixed_parameter(self):
        for family in ("ZABI", "ZIBI"):
            fam = resolve_family(family)
            assert "bd" in fam.parameters
            assert tuple(fam.fixed_parameters) == ("bd",)
            assert fam.fixed_parameter_defaults == {"bd": 1.0}
