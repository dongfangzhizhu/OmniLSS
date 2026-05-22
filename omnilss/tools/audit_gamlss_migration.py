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


def build_audit() -> dict[str, object]:
    rows = _missing(CORE_REQUIREMENTS)
    ok_count = sum(1 for r in rows if r["ok"])
    return {
        "plan_doc": str(PLAN_DOC),
        "requirements_total": len(rows),
        "requirements_ok": ok_count,
        "requirements_missing": len(rows) - ok_count,
        "all_ok": ok_count == len(rows),
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
