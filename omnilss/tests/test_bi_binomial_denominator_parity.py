"""BI (binomial) parity with a binomial denominator ``bd`` > 1.

Why this file exists
--------------------
The parity scanner (``tools/rs_family_parity_scan.py``) flagged BI/DBI with a
derivative mismatch of 1.35e16.  The underlying defect was not in the
derivatives at all: ``dBI`` rescaled the response with

    x_count = round(x * bd)

while R computes ``dbinom(x, size = bd, prob = mu)`` -- i.e. R takes ``x`` as
the **number of successes** already.  Every genuine count response was
therefore multiplied by ``bd`` a second time, so ``y = 1`` out of ``bd = 10``
was evaluated as ``y = 10``.

The same rescaling was baked into the family's ``g_dev_inc`` / ``dldm``, and
the deviance, working weights and fitted values inherited it.  Only the
``bd = 1`` (Bernoulli) path was unaffected -- which is why the existing
``test_r_consistency_bi.py`` suite, written entirely against ``bd = 1``, never
saw the bug.

Response convention (verified against R)
----------------------------------------
``gamlss()`` builds the binomial response from a two-column
``cbind(success, failure)`` matrix, taking ``y`` as the success **count** and
``bd = success + failure``.  ``BI()$dldm`` is ``(y - bd*mu)/(mu*(1-mu))``,
which is only dimensionally consistent with counts.  A single-column response
is instead restricted to ``0 <= y <= 1`` and fitted as Bernoulli with
``bd = 1``.

All reference values below were produced by live R (gamlss.dist).
"""

import numpy as np
import pytest

from omnilss.distributions import BI
from omnilss.dpqr_functions import dBI, pBI, qBI

from tests._r_bridge_helper import R_AVAILABLE, _resolve_rscript

# Live R: dBI(c(0,2,4,6,8,10), mu=c(.25,.35,.45,.55,.65,.75), bd=10)
R_D = np.array(
    [
        0.0563135147095,
        0.1756529531060,
        0.2383666466221,
        0.2383666466221,
        0.1756529531060,
        0.0563135147095,
    ]
)
# Live R: pBI(same arguments)
R_P = np.array(
    [
        0.0563135147095,
        0.2616073913832,
        0.5044045916529,
        0.7339620549691,
        0.9140455617228,
        1.0000000000000,
    ]
)
# Live R: qBI(c(.1,.3,.5,.7,.9), mu=c(.25,.35,.45,.55,.65,.75), bd=10)
R_Q = np.array([1.0, 3.0, 4.0, 6.0, 8.0, 6.0])

_Y = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
_MU = np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
_PROBS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
_BD = 10.0


def test_dbi_matches_r_for_counts():
    """dBI takes counts, not proportions: dbinom(y, size=bd, prob=mu)."""
    got = np.asarray(dBI(_Y, _MU, bd=_BD), dtype=np.float64)
    assert got == pytest.approx(R_D, rel=1e-10)


def test_pbi_matches_r_for_counts():
    got = np.asarray(pBI(_Y, _MU, bd=_BD), dtype=np.float64)
    assert got == pytest.approx(R_P, rel=1e-10)


def test_qbi_matches_r_for_counts():
    got = np.asarray(qBI(_PROBS, _MU, bd=_BD), dtype=np.float64)
    assert got == pytest.approx(R_Q, abs=1e-9)


def test_qbi_recycles_arguments_like_r():
    """R recycles shorter arguments to the longest (``rep(x, length = ly)``).

    Live R: qBI(c(.1,.3,.5,.7,.9), mu=0.4, bd=10) == 2 3 4 5 6
    """
    got = np.asarray(qBI(_PROBS, 0.4, bd=_BD), dtype=np.float64)
    assert got == pytest.approx(np.array([2.0, 3.0, 4.0, 5.0, 6.0]), abs=1e-9)


def test_dbi_does_not_rescale_the_response_by_bd():
    """The original defect: ``round(y * bd)`` folded ``bd`` in twice.

    With bd = 10, the response y = 1 must be read as one success out of ten
    (dbinom(1, 10, .5) = 0.009765625), never as ten out of ten
    (dbinom(10, 10, .5) = 0.0009765625).
    """
    one_success = float(dBI(np.array([1.0]), np.array([0.5]), bd=_BD)[0])
    all_success = float(dBI(np.array([10.0]), np.array([0.5]), bd=_BD)[0])
    assert one_success == pytest.approx(0.009765625, rel=1e-9)
    assert all_success == pytest.approx(0.0009765625, rel=1e-9)
    assert one_success != all_success


def test_bernoulli_path_is_untouched():
    """bd = 1 (or None) stays Bernoulli, which is what existing tests cover."""
    y = np.array([0.0, 1.0])
    assert np.asarray(dBI(y, 0.3)) == pytest.approx([0.7, 0.3], rel=1e-12)
    assert np.asarray(dBI(y, 0.3, bd=1.0)) == pytest.approx([0.7, 0.3], rel=1e-12)


def test_family_deviance_matches_r_log_density():
    """G.dev.incr must equal -2 * log dBI, with y a count."""
    fam = BI()
    got = np.asarray(
        fam.g_dev_inc(_Y, _MU, np.full(_Y.shape, _BD)), dtype=np.float64
    )
    with np.errstate(divide="ignore"):
        expected = -2.0 * np.log(R_D)
    assert got == pytest.approx(expected, rel=1e-9)


def test_family_score_is_r_dldm():
    """R: dldm = (y - bd*mu) / (mu*(1-mu)), evaluated on counts."""
    fam = BI()
    got = np.asarray(
        fam.score_functions["mu"](_Y, _MU, np.full(_Y.shape, _BD)), dtype=np.float64
    )
    expected = (_Y - _BD * _MU) / (_MU * (1.0 - _MU))
    assert got == pytest.approx(expected, rel=1e-9)


def test_family_hessian_is_r_d2ldm2():
    """R: d2ldm2 = -(bd/(mu*(1-mu))) -- independent of y."""
    fam = BI()
    got = np.asarray(
        fam.hessian_functions["mu"](_Y, _MU, np.full(_Y.shape, _BD)), dtype=np.float64
    )
    expected = -_BD / (_MU * (1.0 - _MU))
    assert got == pytest.approx(expected, rel=1e-9)


@pytest.mark.skipif(not R_AVAILABLE, reason="R / Rscript not available")
def test_dbi_matches_live_r():
    """End-to-end check against a live R process rather than frozen constants.

    The test bridge only exposes ``call_r_gamlss`` / ``call_r_predict``, so the
    d/p/q comparison goes through the repository's batch R evaluator, which is
    the same code path the parity scanner uses.
    """
    import json
    import subprocess
    import tempfile
    from pathlib import Path

    r_script = Path(__file__).resolve().parents[2] / "tools" / "r_parity_batch.R"
    if not r_script.exists():  # pragma: no cover - moved/removed tooling
        pytest.skip(f"{r_script} not found")
    rscript = _resolve_rscript()
    if rscript is None:  # pragma: no cover - defensive
        pytest.skip("Rscript not on PATH")

    # The batch evaluator walks the arguments element by element, so every
    # vector within one job must share a length.
    jobs = []
    for type_char, arg, first in (
        ("d", "x", _Y),
        ("p", "q", _Y),
        ("q", "p", _PROBS),
    ):
        mu = _MU[: first.size]
        jobs.append(
            {
                "kind": "dpqr",
                "family": "BI",
                "type": type_char,
                "argnames": [arg, "bd", "mu"],
                "args": [first.tolist(), [_BD] * first.size, mu.tolist()],
            }
        )

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        json.dump(jobs, handle)
        payload = handle.name
    try:
        proc = subprocess.run(
            [rscript, str(r_script), payload], capture_output=True, text=True, timeout=300
        )
    finally:
        Path(payload).unlink(missing_ok=True)

    assert proc.returncode == 0, f"R failed: {proc.stderr[:400]}"
    results = json.loads(proc.stdout)
    assert len(results) == len(jobs)
    for job, out in zip(jobs, results):
        assert "error" not in out, f"{job['type']}: {out['error']}"
        got = np.asarray(
            dBI(_Y, _MU, bd=_BD) if job["type"] == "d"
            else pBI(_Y, _MU, bd=_BD) if job["type"] == "p"
            else qBI(_PROBS, _MU[: _PROBS.size], bd=_BD),
            dtype=np.float64,
        )
        r_vals = np.asarray(out["values"], dtype=np.float64)
        assert got == pytest.approx(r_vals, rel=1e-10, abs=1e-9)
