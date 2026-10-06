"""Smoke tests for the family parity scanner.

``tools/rs_family_parity_scan.py`` is the tool that turns "check one family
against live R" into "check all of them".  It is only useful if its verdicts
are trustworthy, so these tests pin down the properties that make them so:

1.  Every registered family is discovered -- the registry lists 140+ names
    spread across ``distributions*.py``, and a discovery bug would silently
    shrink coverage while still reporting a clean run.
2.  Calibration derives a usable probe point from the family's own metadata
    (``type`` + ``links``) instead of a hand-maintained whitelist.  The
    whitelist version reported false failures for LOGITNO, GEOMO, ZAGA and the
    other zero-inflated count families.
3.  A family no grid can probe is reported ``skip``, never ``fail``: "we could
    not check this" and "this is wrong" are different statements.
4.  The derivative gate still catches a genuinely wrong score function, i.e.
    calibration did not turn the tool into something that always passes.

Gate 2 (derivatives) is pure Python, so these tests need no R.
"""

import numpy as np
import pytest

from omnilss import distributions as dist


def _scanner():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(dist.__file__).resolve().parents[3] / "tools" / "rs_family_parity_scan.py"
    if not path.exists():  # pragma: no cover - tooling moved
        pytest.skip(f"{path} not found")
    spec = importlib.util.spec_from_file_location("rs_family_parity_scan", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rs_family_parity_scan"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scan():
    return _scanner()


def test_discovery_finds_the_whole_registry(scan):
    """Discovery must not silently drop families.

    ``dir(distributions)`` only exposes the ~19 names that module re-exports;
    the rest live in ``distributions_b*.py``.  The scanner asks the resolver
    for an unknown name and harvests the full list from the error.
    """
    found = scan.discover_families()
    assert len(found) > 100, f"only {len(found)} families discovered"
    for expected in ("NO", "GA", "LOGNO", "BI", "ZAGA", "DPO1"):
        assert expected in found


def test_calibration_derives_a_usable_grid_for_every_family(scan):
    """Every family that can be probed must get a finite, positive density."""
    uncalibrated = []
    for family in scan.discover_families():
        try:
            cal = scan.calibrate(family)
        except Exception:
            # A family that cannot even be resolved is a scanner skip, not a
            # test failure -- the parity report handles those separately.
            continue
        if not cal.ok:
            uncalibrated.append(family)
    # Tolerance: a handful of families are pure placeholders with no working
    # density.  The point of this test is that calibration is not failing
    # wholesale the way the whitelist version did.
    assert len(uncalibrated) < len(scan.discover_families()) // 3, (
        f"too many families could not be calibrated: {uncalibrated[:20]}"
    )


@pytest.mark.parametrize(
    "family",
    ["NO", "GA", "LOGNO", "WEI2", "PE2", "GEOMO", "LOGITNO", "ZAGA", "ZIBI", "BE"],
)
def test_derivative_gate_passes_known_good_families(scan, family):
    """Families previously mis-probed by the whitelist now clear gate 2."""
    info = scan.gate_derivatives(family)
    assert info["status"] == "pass", f"{family}: {info.get('detail')}"


def test_calibration_keeps_the_response_in_support(scan):
    """The chosen grid must actually sit inside the family's support.

    LOGITNO lives on (0, 1): probing it on 0.3..4.0 (the old default) produced
    a bogus 1.35e16 derivative mismatch.
    """
    y, _params, _bd = scan._family_grids("LOGITNO", 6)
    assert np.all((y > 0.0) & (y < 1.0))

    # A count family must be probed on integers.
    y_po, _, _ = scan._family_grids("PO", 6)
    assert np.all(y_po == np.round(y_po))


def test_binomial_families_are_probed_with_a_denominator(scan):
    """BI needs a fixed ``bd``; without it the density is meaningless."""
    _y, _params, bd = scan._family_grids("BI", 6)
    assert bd is not None and bd > 1


@pytest.mark.parametrize("family", ["NO", "GA", "BE", "SST", "LOGITNO", "ZAGA"])
def test_fit_sample_is_in_support_and_not_degenerate(scan, family):
    """The RS fit sample must be spread out, not six repeated values.

    Tiling the calibrated grid gave a response with only six distinct values;
    SST then reported a deviance 80x R's, which was indistinguishable from a
    real defect.
    """
    rng = np.random.default_rng(20261006)
    y_grid, _params, _bd = scan._family_grids(family, 6)
    y = np.asarray(scan._fit_data(family, rng, y_grid)["y"], dtype=np.float64)

    assert y.size == scan.N_OBS
    assert np.all(np.isfinite(y))
    assert len(np.unique(y)) > 20, f"{family}: only {len(np.unique(y))} distinct values"
    assert y.min() >= np.min(y_grid) - 1e-9
    assert y.max() <= np.max(y_grid) + 1e-9


@pytest.mark.parametrize("family", ["PO", "BI"])
def test_count_fit_sample_stays_integral(scan, family):
    """A discrete family must not be fitted on fractional counts."""
    rng = np.random.default_rng(20261006)
    y_grid, _params, _bd = scan._family_grids(family, 6)
    y = np.asarray(scan._fit_data(family, rng, y_grid)["y"], dtype=np.float64)
    assert np.all(y == np.round(y))


def test_derivative_gate_still_flags_a_wrong_score(scan, monkeypatch):
    """Calibration must not have made the gate vacuous.

    A deliberately halved score function has to be reported as a failure.
    """
    fam = dist.resolve_family("NO")
    real = fam.score_functions["mu"]
    monkeypatch.setitem(
        fam.score_functions, "mu", lambda y, mu, sigma: 0.5 * real(y, mu, sigma)
    )
    try:
        info = scan.gate_derivatives("NO")
    finally:
        monkeypatch.setitem(fam.score_functions, "mu", real)
    assert info["status"] == "fail"
    assert "max rel diff" in info["detail"]


def test_gate_reports_skip_not_fail_when_unprobeable(scan, monkeypatch):
    """A family whose density never becomes finite is skipped, not failed.

    Family objects are frozen dataclasses, so the probe is pointed at a stub
    rather than patched in place.
    """

    class _Unprobeable:
        parameters = ("mu",)
        links = {"mu": "log"}
        type = "Continuous"
        fixed_parameters = ()
        score_functions = {"mu": lambda y, mu: np.zeros_like(np.asarray(y, dtype=float))}

        def d(self, y, mu):  # noqa: D102 - stub
            raise RuntimeError("no density available")

    real = dist.resolve_family
    monkeypatch.setattr(
        dist, "resolve_family", lambda name: _Unprobeable() if name == "__STUB__" else real(name)
    )
    info = scan.gate_derivatives("__STUB__")
    assert info["status"] == "skip"
    assert "finite" in info["detail"]
