"""SICHEL d/p/q golden reference generated from R gamlss.dist.

The fixture (sichel_dpqr_high_counts_20260912.json) is produced once by
generate_sichel_golden.R with gamlss.dist 6.1.1; tests never call R.

R's pSICHEL upper tail is `1 - cdf` computed in double precision, so at
large counts it saturates (tiny negative values, NaN logs).  The Python
side sums the actual tail masses in log space, which stays accurate.
For large counts the test therefore asserts Python's tail is finite and
bounded by R's artifact, and only requires equality while R's tail is
still numerically meaningful (q <= 100).
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from omnilss.dpqr_functions import dSICHEL, pSICHEL, qSICHEL

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "r_golden"
    / "sichel_dpqr_high_counts_20260912.json"
)

_MEANINGFUL_COUNT = 100  # last count where R's 1-cdf is still exact enough


def _as_float(raw):
    if isinstance(raw, str):
        return float(raw)  # jsonlite strings: "NaN", "Inf", "-Inf"
    return float(raw)


def _entries():
    payload = json.loads(FIXTURE.read_text())
    counts = np.asarray([_as_float(v) for v in payload["counts"]])
    return [(payload, counts, entry) for entry in payload["parameters"]]


def _ids(entry):
    return f"mu{entry['mu']}-sigma{entry['sigma']}-nu{entry['nu']}"


_ALL_ENTRIES = _entries()
_ALL_IDS = [_ids(entry) for _, _, entry in _ALL_ENTRIES]


def _entry_at(index):
    _, counts, entry = _ALL_ENTRIES[index]
    return counts, entry


@pytest.mark.parametrize("index", range(len(_ALL_ENTRIES)), ids=_ALL_IDS)
def test_density_log_matches_r(index):
    counts, entry = _entry_at(index)
    actual = dSICHEL(counts, entry["mu"], entry["sigma"], entry["nu"], log=True)
    expected = np.asarray([_as_float(v) for v in entry["density_log"]])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("index", range(len(_ALL_ENTRIES)), ids=_ALL_IDS)
def test_cdf_and_log_cdf_matches_r(index):
    counts, entry = _entry_at(index)
    actual = pSICHEL(counts, entry["mu"], entry["sigma"], entry["nu"])
    expected = np.asarray([_as_float(v) for v in entry["cdf"]])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-9)

    actual_log = pSICHEL(counts, entry["mu"], entry["sigma"], entry["nu"],
                         log_p=True)
    expected_log = np.asarray([_as_float(v) for v in entry["cdf_log"]])
    meaningful = np.isfinite(expected_log)
    np.testing.assert_allclose(
        np.asarray(actual_log)[meaningful], expected_log[meaningful],
        rtol=1e-6, atol=1e-9)
    # Where R's log(cdf) saturates, Python's log1p-accurate sum stays 0.
    assert np.isfinite(np.asarray(actual_log)[~meaningful]).all()
    np.testing.assert_allclose(
        np.asarray(actual_log)[~meaningful], 0.0, atol=1e-9)


@pytest.mark.parametrize("index", range(len(_ALL_ENTRIES)), ids=_ALL_IDS)
def test_upper_tail_bounded_by_r_artifact(index):
    counts, entry = _entry_at(index)
    r_linear = np.asarray([_as_float(v) for v in entry["cdf_upper"]])
    r_log = np.asarray([_as_float(v) for v in entry["cdf_upper_log"]])
    actual = np.asarray(pSICHEL(counts, entry["mu"], entry["sigma"],
                                entry["nu"], lower_tail=False))
    actual_log = np.asarray(pSICHEL(counts, entry["mu"], entry["sigma"],
                                    entry["nu"], lower_tail=False,
                                    log_p=True))
    assert (actual >= 0).all() and (actual_log <= 0).all()

    meaningful = counts <= _MEANINGFUL_COUNT
    np.testing.assert_allclose(actual[meaningful], r_linear[meaningful],
                               rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(actual_log[meaningful], r_log[meaningful],
                               rtol=1e-6, atol=1e-9)

    # Beyond that, R loses the tail (negative or NaN artifacts).  Python's
    # direct tail sum must remain a valid probability not exceeding any
    # finite artifact R still reports (a NaN bound degenerates to the
    # trivial log-probability bound of 0).
    saturated = counts > _MEANINGFUL_COUNT
    bound = np.where(np.isfinite(r_log[saturated]), r_log[saturated], 0.0)
    assert (actual_log[saturated] <= bound + 1e-9).all()
    assert np.isfinite(actual_log[saturated]).all()


@pytest.mark.parametrize("index", range(len(_ALL_ENTRIES)), ids=_ALL_IDS)
def test_quantiles_match_r(index):
    counts, entry = _entry_at(index)
    probs = jnp.asarray(entry["quantile_probabilities"])
    expected = np.asarray(entry["quantiles_lower"])
    actual = qSICHEL(probs, entry["mu"], entry["sigma"], entry["nu"])
    np.testing.assert_array_equal(actual, expected)

    expected_upper = np.asarray(entry["quantiles_upper"])
    actual_upper = qSICHEL(1e-6, entry["mu"], entry["sigma"], entry["nu"],
                           lower_tail=False)
    np.testing.assert_array_equal(actual_upper, expected_upper)
