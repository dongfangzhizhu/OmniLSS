# SPDX-License-Identifier: GPL-3.0-or-later
"""Family response-domain validation — R parity for ``family$y.valid``.

R source: ``gamlss/R/gamlss-5.R:960`` inside ``gamlss()``::

    ##checking the permissible y values
    if (!family$y.valid(y))
        stop( "response variable out of range")

The check runs *before* any IRLS step, so R never produces a model for
out-of-support data.  Before this module OmniLSS only warned about eta clipping
and returned a model whose deviance was meaningless.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from omnilss import GA, NO, gamlss
from omnilss.distributions import resolve_family
from omnilss.response_domain import (
    ResponseDomainError,
    response_domain_for,
    response_violation_count,
    validate_response,
)

from tests._r_bridge_helper import R_AVAILABLE, R_BRIDGE_CLS as RBridge

# Live-R cases only; the pure-Python predicate tests above always run.
_r_live = pytest.mark.skipif(not R_AVAILABLE, reason="R bridge not available")


# ---------------------------------------------------------------------------
# Predicate-level behaviour (no fitting involved)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("family", "value", "valid"),
    [
        # GA: R `all(y > 0)`
        ("GA", 1.0, True),
        ("GA", 0.0, False),
        ("GA", -0.5, False),
        # ZIP: R `all(y >= 0)` — zero allowed
        ("ZIP", 0.0, True),
        ("ZIP", -1.0, False),
        # BEINF: R `all(y >= 0 & y <= 1)` — both endpoints included
        ("BEINF", 0.0, True),
        ("BEINF", 1.0, True),
        ("BEINF", 1.5, False),
        # BEZI: R `all(y >= 0 & y < 1)` — 1 excluded
        ("BEZI", 0.0, True),
        ("BEZI", 1.0, False),
        # BEOI: R `all(y > 0 & y <= 1)` — 0 excluded
        ("BEOI", 0.0, False),
        ("BEOI", 1.0, True),
        # PARETO: R `all(y > 1)`
        ("PARETO", 1.0, False),
        ("PARETO", 1.5, True),
        # MN4 / MN5: shifted integer supports
        ("MN4", 0.9, False),
        ("MN4", 1.0, True),
        ("MN4", 4.0, True),
        ("MN4", 4.1, False),
        ("MN5", 5.5, False),
        # ZIPF: R `all(y >= 1)`
        ("ZIPF", 0.5, False),
        ("ZIPF", 1.0, True),
    ],
)
def test_predicate_matches_r_y_valid(family, value, valid):
    """Each entry mirrors the R ``y.valid`` body for that family."""
    assert response_violation_count(family, np.array([value])) == (0 if valid else 1)


def test_unbounded_family_accepts_negative_response():
    """NO declares ``y.valid = function(y) TRUE`` — negatives are legitimate."""
    assert response_violation_count("NO", np.array([-10.0, 0.0, 5.0])) == 0


def test_families_without_r_y_valid_stay_permissive():
    """R exports no such family, so Python must not invent a restriction.

    BETA, DAGUM, ALPHA and ZINB are absent from the installed `gamlss.dist`
    6.1.1 export list (checked with `getNamespaceExports`); with no
    `y.valid` to mirror, adding one would reject data R happily fits.

    PO used to be listed here, and that was the D2.128 bug class: live R's
    `PO()$y.valid` is `all(y >= 0)`, so a negative Poisson response is
    refused by R and must be refused here too -- see
    `test_poisson_rejects_negative_response_like_r`.
    """
    for family in ("BETA", "ZINB", "ALPHA", "DAGUM"):
        assert response_violation_count(family, np.array([-1.0, 0.0, 2.0])) == 0


def test_poisson_rejects_negative_response_like_r():
    """R: `PO()$y.valid = all(y >= 0)` (live R 4.6.0 / gamlss.dist 6.1.1)."""
    assert response_violation_count("PO", np.array([0.0, 3.0])) == 0
    assert response_violation_count("PO", np.array([-1.0, 3.0])) == 1


def test_nan_and_inf_rejected_even_for_unbounded_family():
    """No family can fit a non-finite response, including the permissive ones."""
    for family in ("NO", "BETA", "GA"):
        assert response_violation_count(family, np.array([1.0, np.nan])) == 1
        assert response_violation_count(family, np.array([1.0, np.inf])) == 1


def test_error_message_reproduces_r_text():
    with pytest.raises(ResponseDomainError) as excinfo:
        validate_response("GA", np.array([1.0, -2.0, 3.0]))
    message = str(excinfo.value)
    assert "response variable out of range" in message
    assert "GA" in message
    assert excinfo.value.offending == 1
    assert excinfo.value.sample == (-2.0,)


def test_violation_count_reports_all_offenders():
    assert response_violation_count("GA", np.array([0.0, -1.0, -2.0, 1.0])) == 3


def test_empty_response_is_vacuously_valid():
    assert validate_response("GA", np.array([])) == 0


# ---------------------------------------------------------------------------
# Integration with the fitting entry point
# ---------------------------------------------------------------------------

def test_gamlss_rejects_out_of_support_response():
    """GA cannot fit a negative response — R stops, so must we."""
    data = {"y": [1.0, 2.0, 3.0, -1.0], "x": [1.0, 2.0, 3.0, 4.0]}
    with pytest.raises(ResponseDomainError, match="response variable out of range"):
        gamlss("y ~ x", family=GA(), data=data, method="RS")


@pytest.mark.parametrize("method", ["RS", "CG", "MIXED"])
def test_every_method_route_rejects_out_of_support_response(method):
    """The guard sits in ``gamlss()``, so no method can bypass it.

    Checking only the NumPy RS path would leave CG / MIXED / RS_JAX fitting
    data R refuses — the same silent-divergence class D2.2 closed.
    """
    data = {"y": [1.0, 2.0, 3.0, -1.0], "x": [1.0, 2.0, 3.0, 4.0]}
    with pytest.raises(ResponseDomainError):
        gamlss("y ~ x", family=GA(), data=data, method=method)


def test_in_support_response_still_fits():
    """The guard must not reject legitimate data."""
    data = {"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]}
    model = gamlss("y ~ x", family=GA(), data=data, method="RS", max_iter=20)
    assert np.isfinite(model.deviance)


def test_negative_response_fine_for_no_family():
    """NO is unbounded on the response; a negative response is valid data."""
    data = {"y": [-3.0, -1.0, 2.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]}
    model = gamlss("y ~ x", family=NO(), data=data, method="RS", max_iter=20)
    assert np.isfinite(model.deviance)


def test_zero_is_valid_for_zip_but_not_for_ga():
    """The two positive-family rules differ exactly at zero."""
    zero_data = {"y": [0.0, 1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0, 4.0]}
    assert gamlss("y ~ x", family=resolve_family("ZIP"), data=zero_data,
                  method="RS", max_iter=5) is not None
    with pytest.raises(ResponseDomainError):
        gamlss("y ~ x", family=GA(), data=zero_data, method="RS", max_iter=5)


# ---------------------------------------------------------------------------
# Live R bridge — proves the parity claim rather than asserting it
# ---------------------------------------------------------------------------

@_r_live
def test_r_gamlss_also_rejects_ga_with_negative_response():
    """R must stop on the same data, with the same message.

    The bridge surfaces R's ``stop()`` as a ``RuntimeError`` rather than a
    ``{"success": False}`` dict, so the assertion is on the raised message.
    """
    data = {"y": [1.0, 2.0, 3.0, -1.0], "x": [1.0, 2.0, 3.0, 4.0]}
    with pytest.raises(RuntimeError, match="response variable out of range"):
        RBridge().call_r_gamlss(
            data=data, formula="y ~ x", family="GA", sigma_formula="~1", method="RS"
        )


@_r_live
def test_r_gamlss_accepts_the_same_in_support_data_python_accepts():
    """Both sides fit the positive-response GA data — no over-rejection."""
    data = {"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]}
    result = RBridge().call_r_gamlss(
        data=data, formula="y ~ x", family="GA", sigma_formula="~1", method="RS"
    )
    assert result["success"], result.get("error", "R failed")

    py = gamlss("y ~ x", family=GA(), data=data, method="RS", max_iter=20)
    np.testing.assert_allclose(
        float(py.deviance), float(result["deviance"]), rtol=1e-3, atol=1e-3
    )


# ---------------------------------------------------------------------------
# Full-table parity against the pinned live-R dump (D2.129)
#
# `omnilss/tests/fixtures/r_golden/response_domains_y_valid.json` is produced by
# `tools/generate_response_domain_golden.py` from live R; it holds every
# exported family's `deparse(body(y.valid))` plus the mapping from our dispatch
# name to R's name.  These tests need no R at run time and fail loudly on any
# drift between the table and live R.
# ---------------------------------------------------------------------------

GOLDEN_PATH = (
    Path(__file__).parent / "fixtures" / "r_golden" / "response_domains_y_valid.json"
)

#: `deparse(body(y.valid))` text -> predicate name in omnilss.response_domain.
#: A body that is not listed here fails the test, so a new R shape can never be
#: silently mapped to the wrong predicate.
R_BODY_TO_PREDICATE: dict[str, str] = {
    "TRUE": "_all_finite",
    "all(y > 0)": "_all_positive",
    "all(y >= 0)": "_all_non_negative",
    "all(y > 0 & y < 1)": "_in_open_unit",
    "all(y >= 0 & y <= 1)": "_in_closed_unit",
    "all(y >= 0 & y < 1)": "_in_unit_excl_zero",
    "all(y > 0 & y <= 1)": "_in_unit_excl_one",
    "all(y > 1)": "_above_one",
    "all(y >= 1)": "_at_least_one",
    "all(y >= 1 & y <= 4)": "_in_closed_1_4",
    "all(y >= 1 & y <= 5)": "_in_closed_1_5",
    # Coded factor: the level count is the family's business (see
    # _FACTOR_LEVEL_FAMILIES), the response itself is unrestricted.
    "if (is.factor(y)) nlevels(y) == 3 else all(y >= 1 & y <= 3)": "_all_finite",
}

#: OmniLSS families with no R counterpart at all: the 19 generated skeleton
#: families (no gamlss.dist export, no vendored source), ZILG plus the two
#: LOGSHASH variants (absent from gamlss.dist 6.1.1; the vendored development
#: source declares `y.valid = function(y) TRUE` for both).  They stay on the
#: permissive default, and this list pins *why*.
NO_R_REFERENCE_FAMILIES = frozenset(
    {
        "ZILG",
        "LOGSHASH",
        "LOGSHASHo",
    }
    | {
        name
        for name in (
            "BINOM_1_31", "BINOM_2_33", "BINOM_3_33",
            "CONTR_2_12", "CONTR_3_11", "CONTR_4_13",
            "CONTR01_2_13", "CONTR01_4_33",
            "CONTRPLUS_1_1", "CONTRPLUS_2_11", "CONTRPLUS_3_13", "CONTRPLUS_4_33",
            "COUNT_1_22", "COUNT_1_31", "COUNT_2_32", "COUNT_2_32R",
            "COUNT_2_33", "COUNT_3_32", "COUNT_3_33",
        )
    }
)


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _golden_cases(golden: dict) -> list[tuple[str, str, str]]:
    cases = []
    for our_name, r_name in sorted(golden["supported"].items()):
        if r_name is None:
            continue
        body = golden["families"][r_name]
        if body not in R_BODY_TO_PREDICATE:
            raise AssertionError(
                f"unclassified R y.valid body for {r_name!r}: {body!r} -- extend "
                "R_BODY_TO_PREDICATE and the table in omnilss.response_domain"
            )
        cases.append((our_name, r_name, R_BODY_TO_PREDICATE[body]))
    return cases


def test_golden_artifact_has_provenance(golden):
    meta = golden["meta"]
    assert meta["gamlss_dist"] and meta["r_version"]
    assert meta["r_dump_generator"].endswith("r_y_valid_dump.R")
    assert len(golden["families"]) >= 100


def test_table_matches_live_r_y_valid(golden):
    """Every family R restricts resolves to exactly the predicate R declares."""
    mismatches = []
    for our_name, r_name, expected in _golden_cases(golden):
        actual = response_domain_for(our_name).__name__
        if actual != expected:
            mismatches.append(f"{our_name} (R {r_name}): {actual} != {expected}")
    assert not mismatches, "\n".join(mismatches)


def test_unmatched_families_are_the_known_set(golden):
    """No family may lose its R counterpart unnoticed."""
    unmatched = {name for name, r_name in golden["supported"].items() if r_name is None}
    assert unmatched == NO_R_REFERENCE_FAMILIES


def test_every_restricted_r_family_has_a_table_entry(golden):
    """Guard against the D2.128 failure mode: R stops, we fit."""
    missing = [
        our_name
        for our_name, r_name, expected in _golden_cases(golden)
        if expected != "_all_finite"
        and response_domain_for(our_name).__name__ == "_all_finite"
    ]
    assert not missing, f"families R restricts but we accept anything: {missing}"


def test_table_lookup_is_case_insensitive(golden):
    """Case drift between registry key and R name must not silently disable the guard."""
    assert response_domain_for("BCCGUNTR").__name__ == "_all_positive"
    assert response_domain_for("bccguntR").__name__ == "_all_positive"
    assert response_domain_for("exgaus").__name__ == "_all_finite"
