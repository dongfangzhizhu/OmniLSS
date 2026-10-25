"""BB's gates were never measured: bd was passed twice (D2.69).

The skip
--------
BB declared bd in *both* parameters and fixed_parameters.  Every probe then
carried bd twice -- once as the fixed probe constant, once as a per-parameter
grid -- so dBB raised "got multiple values", calibration found no finite
grid, and the scanner reported deriv=skip / dpqr=skip for BB.  rsdev passed
throughout, which is why nobody noticed: the fit path resolves bd from the
data and never goes through the probe.

The fix
-------
The probes now treat a declared fixed parameter as a constant, not as a
dimension: calibrate, the d/p/q job builder and the Python-side dpqr all
iterate over "parameters minus fixed_parameters".  BB keeps bd *in*
parameters on purpose -- that is what lets the AD closures and the fit honour
a bd from the data -- so the deriv gate builds its argument vector in
fam.parameters order (and falls back without the trailing bd for families
that bake it into the closure).

Result: BB goes from deriv=skip / dpqr=skip to pass / pass, and the other
fixed-bd families (BI, DBI, ZABB, ZIBB) stay green.
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


class TestProbeParameters:
    def test_bb_declares_bd_once_as_a_probe_dimension(self):
        """bd stays in parameters (the fit needs it) but is not a dimension."""
        fam = resolve_family("BB")
        assert "bd" in fam.parameters
        assert "bd" in (fam.fixed_parameters or ())
        assert "bd" not in scan._family_parameters("BB")

    @pytest.mark.parametrize("family", ["BI", "DBI", "ZABB", "ZIBB", "BB"])
    def test_calibration_ignores_the_fixed_parameter(self, family):
        cal = scan.calibrate(family, 8)
        assert cal.ok, f"{family} calibration must succeed"
        assert cal.bd is not None
        probed = [p for p in resolve_family(family).parameters if p != "bd"]
        assert len(cal.params) == len(probed)

    def test_dpqr_jobs_carry_bd_exactly_once(self):
        jobs, _ = scan._build_jobs(["BB", "BI", "DBI"], 8)
        for job in jobs:
            if job["kind"] != "dpqr":
                continue
            names = job["argnames"]
            assert names.count("bd") == 1, (job["family"], names)
            assert len(names) == len(job["args"]), (job["family"], names)


class TestDerivGate:
    @pytest.mark.parametrize("family", ["BB", "BI", "DBI", "ZABB", "ZIBB"])
    def test_fixed_bd_families_pass_their_derivative_gate(self, family):
        row = scan.gate_derivatives(family)
        assert row["status"] == "pass", row


@pytest.mark.skipif(shutil.which("Rscript") is None,
                    reason="Rscript not on PATH")
class TestDpqrGate:
    def test_bb_matches_live_r_after_the_probe_fix(self):
        """The measurement BB never had: d/p/q against live R."""
        jobs, index = scan._build_jobs(["BB"], 8)
        results = scan._run_r(jobs)
        assert len(results) == len(jobs)
        assert not any("error" in r for r in results), results
        row = scan.evaluate_family("BB", results, index, jobs, True, 8)
        assert row["dpqr"]["status"] == "pass", row["dpqr"]
