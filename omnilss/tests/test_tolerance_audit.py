"""Phase 4.2: tolerance-system audit — no rtol > 0.01 outside the exception list.

The Release-1 / core-family R-consistency tests must use rtol <= 1e-3 for
fits and <= 1e-5 for d/p/q (tightened from the historical 10-50%).  A small
exception list is retained for non-core families whose Python/R fit
difference is a real, tracked fitting-engine defect (NOT a tolerance-writing
choice); those are documented here so the audit is fail-closed against new
loose tolerances.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent

# Files whose rtol > 0.01 is a documented exception (real fit differences,
# tracked for the fitting-engine work; several also fail earlier on the
# capability gate for placeholder families):
EXEMPT_FILES: dict[str, str] = {
    "test_r_consistency_batch1_bc.py": "BCCG/BCT/BCPE/JSU fit deviance differs ~33% (engine)",
    "test_r_consistency_batch3_4_remaining.py": "BEINF0/1/BEOI/BEZI/SN2 fit differences (engine)",
    "test_r_consistency_batch5.py": "ZIP2 etc. fit differences / placeholder gate",
    "test_r_consistency_batch6.py": "WARING/YULE fit differences (engine)",
    "test_r_consistency_batch7_8.py": "GB2/LNO/NET/PARETO fit differences (engine)",
    "test_r_consistency_exgaus.py": "exGaussian fit differences (engine)",
    "test_r_consistency_gt.py": "GT fit differences (engine)",
    "test_r_consistency_logno.py": "LOGNO fit differences (engine)",
    "test_r_consistency_nbii.py": "NBII fit differences (engine)",
    "test_r_consistency_pareto2.py": "PARETO2 fit differences (engine)",
    "test_r_consistency_rg.py": "RG fit 1.5e-2 (engine)",
    "test_r_consistency_shash.py": "SHASH fit 5e-2 (engine)",
    "test_r_consistency_sn1.py": "SN1 fit 5e-2 (engine)",
    "test_r_consistency_zip.py": "ZIP fit 0.15 / placeholder gate",
}

RTOL_PATTERN = re.compile(r"rtol=([0-9.]+e?-?[0-9]*)")


def _over_tolerance_files() -> list[str]:
    offenders = []
    for path in sorted(TESTS_DIR.glob("test_r_consistency*.py")):
        text = path.read_text(encoding="utf-8")
        over = False
        for match in RTOL_PATTERN.finditer(text):
            try:
                if float(match.group(1)) > 0.01:
                    over = True
                    break
            except ValueError:
                continue
        if over:
            offenders.append(path.name)
    return offenders


def test_no_unlisted_tolerance_overruns() -> None:
    """Every rtol > 0.01 must be in the documented exception list."""
    offenders = _over_tolerance_files()
    unlisted = [f for f in offenders if f not in EXEMPT_FILES]
    assert unlisted == [], (
        f"rtol > 0.01 outside the exception list: {unlisted}; "
        "tighten to <= 1e-3 (fits) / <= 1e-5 (d/p/q) or add to EXEMPT_FILES "
        "with the engine-defect reason"
    )


def test_exception_list_files_still_over_tolerance() -> None:
    """The exception list stays honest: every listed file really is > 0.01."""
    offenders = set(_over_tolerance_files())
    for filename in EXEMPT_FILES:
        assert filename in offenders, (
            f"{filename} is listed as an exception but has no rtol > 0.01; "
            "remove it from EXEMPT_FILES"
        )
