"""Strict audit for OmniLSS migration coverage against the 2026-05-22 plan."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PLAN_DOC = REPO_ROOT / "docs" / "development" / "r-gamlss-coverage-and-plan-2026-05-22.md"


@dataclass(frozen=True)
class Requirement:
    symbol: str
    python_paths: tuple[str, ...]


CORE_REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement("gamlss", ("omnilss/src/omnilss/fitting.py", "omnilss/src/omnilss/gamlss_5.py")),
    Requirement("gamlss.control", ("omnilss/src/omnilss/controls.py",)),
    Requirement("predict.gamlss", ("omnilss/src/omnilss/predict_gamlss_23_12_21.py",)),
    Requirement("predictAll", ("omnilss/src/omnilss/predictAll_22_08_22.py",)),
    Requirement("lpred", ("omnilss/src/omnilss/lpred.py",)),
    Requirement("prodist.gamlss", ("omnilss/src/omnilss/prodist.py",)),
    Requirement("acfResid", ("omnilss/src/omnilss/acfResid.py",)),
    Requirement("summary.gamlss", ("omnilss/src/omnilss/SUMMARY.py",)),
    Requirement("print.gamlss", ("omnilss/src/omnilss/print.py",)),
    Requirement("Q.stats", ("omnilss/src/omnilss/qstats.py",)),
    Requirement("LR.test", ("omnilss/src/omnilss/LR_test_12_06_2013.py",)),
    Requirement("VC.test", ("omnilss/src/omnilss/VuongClarkTest.py",)),
    Requirement("devianceIncr", ("omnilss/src/omnilss/DevianceIncr.py",)),
    Requirement("centiles.pred", ("omnilss/src/omnilss/centile_pred_30_06_22.py",)),
    Requirement("centiles.com", ("omnilss/src/omnilss/centilescom.py",)),
    Requirement("centiles.fan", ("omnilss/src/omnilss/centilesFan.py",)),
    Requirement("centiles", ("omnilss/src/omnilss/centilesPLOT.py",)),
    Requirement("fittedPlot", ("omnilss/src/omnilss/fitted_plot.py",)),
    Requirement("pdf.plot", ("omnilss/src/omnilss/pdfplot.py",)),
    Requirement("term.plot", ("omnilss/src/omnilss/term_plot_new.py",)),
    Requirement("plot2way", ("omnilss/src/omnilss/plot2way.py",)),
)


def _missing(reqs: Iterable[Requirement]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for req in reqs:
        missing = [p for p in req.python_paths if not (REPO_ROOT / p).exists()]
        rows.append(
            {
                "symbol": req.symbol,
                "python_paths": list(req.python_paths),
                "ok": not missing,
                "missing_paths": missing,
            }
        )
    return rows


#: P0-2 calibration.  This audit only checks that a handful of python_paths
#: EXIST on disk (a 2026-05-22 plan-doc check).  ``all_ok`` / ``requirements_ok``
#: are therefore path-existence results, NOT a completion rate, NOT R
#: numerical-equivalence evidence, and NOT validated/parity/production-safe
#: status.  Emitted into the JSON (regenerable, never hand-edited) so any reader
#: sees the caveat inline; the authoritative, regenerable coverage is elsewhere.
CALIBRATION: dict[str, object] = {
    "note": (
        "Historical 2026-05-22 path-existence audit against plan_doc. "
        "'all_ok' and 'requirements_ok'==requirements_total ONLY mean each "
        "listed python_path existed at generation time. This is NOT a "
        "completion rate, NOT R numerical-equivalence / validated / parity / "
        "production-safe evidence, and NOT current OmniLSS coverage. Do not "
        "cite it as any of those."
    ),
    "authoritative_artifacts": [
        "docs/development/r-symbol-coverage.json",
        "docs/development/symbol-coverage-diff.json",
    ],
    "gamlss2_out_of_scope": True,
    "out_of_scope_packages": ["gamlss2"],
    "scope_single_source": "tools/coverage_scope.py",
}


def build_audit() -> dict[str, object]:
    rows = _missing(CORE_REQUIREMENTS)
    ok_count = sum(1 for r in rows if r["ok"])
    return {
        # repo-relative POSIX path (machine-independent, regenerable) instead
        # of an absolute path that changed on every host's regeneration.
        "plan_doc": PLAN_DOC.relative_to(REPO_ROOT).as_posix(),
        "requirements_total": len(rows),
        "requirements_ok": ok_count,
        "requirements_missing": len(rows) - ok_count,
        "all_ok": ok_count == len(rows),
        "calibration": CALIBRATION,
        "rows": rows,
    }


def main() -> int:
    payload = build_audit()
    out = REPO_ROOT / "docs" / "development" / "gamlss-core-migration-audit-2026-05-22.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out)
    return 0 if payload["all_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
