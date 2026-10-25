"""Probe reachability for the zero-variant families (D2.70).

ZABI/ZIBI/ZAZIPF skipped their dpqr gate with "unused argument (nu = ...)":
live R names the hurdle mass "sigma" in those three families (their base has
no dispersion of its own, so R reuses the sigma slot) while our factories
name it "nu".  A translation table in the scanner closes the naming gap, and
a {0, 1} probe support closes the second gap -- the families are built on the
bd-less binomial, so R's dZABI/dZIBI default to bd = 1 and reject a 0..7
grid.

ZIBI is deliberately NOT registered: at our own fitted parameters our
density gives 64.668 and live R's gives 78.270, while the first observations
agree to 1e-16 and R then repeats a single value (-1.006054) for the rest --
a discrepancy to investigate, not to hide behind a probe.
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

    def test_zibi_stays_untranslated(self):
        """Its density disagrees with R's; probing it would just turn a skip
        into a fail without identifying the culprit."""
        assert ("ZIBI", "nu") not in scan._R_ARGNAME_MAP

    def test_job_argnames_use_the_r_name(self):
        jobs, _ = scan._build_jobs(["ZABI", "ZAZIPF"], 8)
        dps = [j for j in jobs if j["kind"] == "dpqr"]
        assert dps
        for job in dps:
            assert "nu" not in job["argnames"]
            assert job["argnames"].count("sigma") == 1
            assert len(job["argnames"]) == len(job["args"])

    def test_zabi_probe_grid_is_bernoulli_support(self):
        assert scan._SUPPORT_BOUNDS.get("ZABI") == (0, 1)
        cal = scan.calibrate("ZABI", 8)
        assert cal.ok
        assert set(np.unique(cal.y)) <= {0.0, 1.0}


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
    @pytest.mark.parametrize("family", ["ZABI", "ZAZIPF"])
    def test_derivative_gate_passes(self, family):
        row = scan.gate_derivatives(family)
        assert row["status"] == "pass", row

    def test_zibi_dpqr_stays_skipped(self):
        """Regression guard for the deliberate omission."""
        fam = resolve_family("ZIBI")
        # The family still exposes nu (our name); the probe simply does not
        # claim to measure it against R until the density question is settled.
        assert "nu" in fam.parameters
        assert ("ZIBI", "nu") not in scan._R_ARGNAME_MAP
