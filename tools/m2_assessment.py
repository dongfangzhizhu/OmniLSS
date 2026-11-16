# SPDX-License-Identifier: GPL-3.0-or-later
"""M2 wrap-up assessment (D2.239).

DEVELOPMENT_PLAN section 22.4 states M2 as:

    P2 convergence defects classified; RS/CG/MIXED layered parity;
    fit/newdata evidence complete; failures explainable.

Three of those four already have committed artifacts.  This tool adds the
missing one -- a single fit/newdata evidence matrix -- and records the whole
M2 verdict from artifacts plus a live replay:

  * P2    -- docs/development/p2-defect-classification-2026-11-15.json
  * layers-- docs/development/layered-parity-2026-11-15.json
  * fit/newdata -- measured here: for every promoted family fixture it checks
    that the golden carries coefficients for each *estimated* parameter, a
    deviance, a convergence schema, a newdata grid and R's predictions; then it
    refits our side and measures (a) the newdata prediction gap vs live R and
    (b) a save_model_json/load_model_json round-trip drift.

The live part is what makes this more than a checklist: a fixture can claim
newdata/predicted keys while our model no longer reproduces them.

Usage::

    python tools/m2_assessment.py --write docs/development/m2-assessment-2026-11-15.json
    python tools/m2_assessment.py --check
    python tools/m2_assessment.py --json --no-live   # static evidence only
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "omnilss" / "src"))
sys.path.insert(0, str(ROOT / "tools"))

GOLDEN_DIR = ROOT / "omnilss" / "tests" / "fixtures" / "r_golden" / "regenerated"
GOLDEN_MANIFEST = GOLDEN_DIR / "manifest.json"
FROZEN = ROOT / "docs" / "development" / "highfreq-families-frozen.json"
P2 = ROOT / "docs" / "development" / "p2-defect-classification-2026-11-15.json"
LAYERED = ROOT / "docs" / "development" / "layered-parity-2026-11-15.json"
DEFAULT_OUT = ROOT / "docs" / "development" / "m2-assessment-2026-11-15.json"

SCHEMA_VERSION = 1
PARAMETERS = ("mu", "sigma", "nu", "tau")
#: Same bounds the golden replay tests use.
PREDICT_RTOL = 2e-4
PREDICT_ATOL = 2e-4
ROUNDTRIP_RTOL = 1e-10
#: The layered-parity cell vocabulary this assessment accepts as classified.
LAYER_CODE_VOCABULARY = (
    "pass",
    "flat_ridge",
    "r_error",
    "r_not_converged",
    "mismatch",
    "py_error",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _promoted() -> list[str]:
    return sorted(item["family"] for item in _read(FROZEN)["tiers"]["promoted"])


def _case_for(family: str, manifest: dict) -> str | None:
    for case, entry in manifest.items():
        if entry["family"] == family:
            return case
    return None


def _static_evidence(payload: dict) -> dict:
    """What the fixture itself claims to record."""
    expected = payload.get("expected", {})
    coefficients = {
        parameter: f"{parameter}_coefficients" in expected
        for parameter in PARAMETERS
        if f"{parameter}_coefficients" in expected
    }
    return {
        "has_deviance": "deviance" in expected,
        "estimated_coefficient_blocks": sorted(coefficients),
        "has_newdata": isinstance(payload.get("newdata"), dict)
        and "x" in payload["newdata"],
        "has_predicted": isinstance(payload.get("predicted"), dict)
        and bool(payload["predicted"]),
        "predicted_parameters": sorted(payload.get("predicted", {})),
        "has_convergence": all(
            key in payload.get("convergence", {})
            for key in ("iter", "converged", "n_cyc")
        )
        if isinstance(payload.get("convergence"), dict)
        else False,
    }


def _live_check(family: str, payload: dict, max_iter: int | None) -> dict:
    from omnilss import gamlss, load_model_json, save_model_json
    from omnilss.operations import deviance

    data = {
        "y": np.asarray(payload["data"]["y"], dtype=float),
        "x": np.asarray(payload["data"]["x"], dtype=float),
    }
    kwargs = {"max_iter": int(max_iter)} if max_iter else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gamlss(payload["formula"], family=family, data=data, **kwargs)
    newdata = {"x": np.asarray(payload["newdata"]["x"], dtype=float)}
    predicted = model.predict_params(newdata)

    # The binding test is the same one the golden replay uses: elementwise
    # allclose(atol + rtol*|ref|).  A plain max-abs gap would flag IG / NBII,
    # whose mu is O(3), where 4.8e-4 is well inside 2e-4 relative.
    worst = 0.0
    worst_param = ""
    worst_rel = 0.0
    missing = []
    within = True
    for parameter, reference in payload["predicted"].items():
        if parameter not in predicted:
            missing.append(parameter)
            continue
        ref = np.asarray(reference, dtype=float)
        got = np.asarray(predicted[parameter], dtype=float)
        if got.shape != ref.shape:
            missing.append(parameter)
            continue
        gap = float(np.max(np.abs(got - ref)))
        rel = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-12)))
        if gap > worst:
            worst, worst_param = gap, parameter
        worst_rel = max(worst_rel, rel)
        if not np.allclose(got, ref, rtol=PREDICT_RTOL, atol=PREDICT_ATOL):
            within = False

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{family}.omnilss"
        save_model_json(model, path)
        restored = load_model_json(path)
    after = restored.predict_params(newdata)
    roundtrip = 0.0
    for parameter, before_value in predicted.items():
        if parameter not in after:
            roundtrip = float("inf")
            break
        roundtrip = max(
            roundtrip,
            float(
                np.max(
                    np.abs(
                        np.asarray(after[parameter], dtype=float)
                        - np.asarray(before_value, dtype=float)
                    )
                )
            ),
        )

    reference_dev = float(payload["expected"]["deviance"])
    our_dev = float(deviance(model))
    return {
        "converged": bool(getattr(model, "converged", False)),
        "iter": int(getattr(model, "iter", 0)),
        "predicted_max_abs_gap": worst,
        "predicted_max_rel_gap": worst_rel,
        "predicted_worst_parameter": worst_param,
        "predicted_missing": missing,
        "predicted_within_tolerance": within,
        "roundtrip_max_abs_drift": roundtrip,
        "deviance_rel": abs(our_dev - reference_dev) / max(abs(reference_dev), 1.0),
    }


def build(live: bool = True) -> dict:
    manifest = _read(GOLDEN_MANIFEST)
    promoted = _promoted()
    rows = []
    for family in promoted:
        case = _case_for(family, manifest)
        if case is None:
            rows.append({"family": family, "case": None, "error": "no golden case"})
            continue
        payload = _read(GOLDEN_DIR / f"{case}.json")
        entry = manifest[case]
        row = {
            "family": family,
            "case": case,
            "static": _static_evidence(payload),
        }
        if live:
            try:
                row["live"] = _live_check(family, payload, entry.get("n_cyc"))
            except Exception as exc:  # noqa: BLE001 - report, do not crash
                row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        rows.append(row)

    def _static_ok(row: dict) -> bool:
        if row.get("error") and "live" not in row:
            return False
        static = row.get("static") or {}
        return bool(
            static.get("has_deviance")
            and static.get("estimated_coefficient_blocks")
            and static.get("has_newdata")
            and static.get("has_predicted")
            and static.get("has_convergence")
        )

    static_complete = [row["family"] for row in rows if _static_ok(row)]
    if live:
        predict_ok = [
            row["family"]
            for row in rows
            if row.get("live")
            and not row["live"]["predicted_missing"]
            and row["live"]["predicted_within_tolerance"]
        ]
        roundtrip_ok = [
            row["family"]
            for row in rows
            if row.get("live")
            and row["live"]["roundtrip_max_abs_drift"] <= ROUNDTRIP_RTOL
        ]
        converged_ok = [
            row["family"] for row in rows if row.get("live") and row["live"]["converged"]
        ]
    else:
        predict_ok = roundtrip_ok = converged_ok = []

    p2 = _read(P2)
    layered = _read(LAYERED)
    # Every layered cell must carry a code from the documented vocabulary; an
    # unknown code would mean a failure nobody classified.
    unknown_layer_codes = sorted(
        {cell["code"] for cell in layered["cells"]} - set(LAYER_CODE_VOCABULARY)
    )

    criteria = {
        "p2_defects_classified": {
            "met": bool(p2["verdict"]["all_failures_classified"]),
            "detail": (
                f"{p2['verdict']['non_pass_total']} non-pass entries across three gates, "
                f"unclassified={len(p2['unclassified'])}"
            ),
        },
        "layered_parity_our_side": {
            "met": bool(layered["verdict"]["our_side_parity_ok"]),
            "detail": (
                f"{layered['scope']['cells']} cells; counts={layered['counts']['by_code']}; "
                f"mismatch={len(layered['problems']['mismatch'])}, "
                f"py_error={len(layered['problems']['py_error'])}"
            ),
        },
        "fit_newdata_evidence_static": {
            "met": len(static_complete) == len(rows),
            "detail": (
                f"{len(static_complete)}/{len(rows)} promoted fixtures carry deviance + "
                "coefficient blocks + convergence + newdata + predicted"
            ),
        },
        "fit_newdata_evidence_live": {
            "met": bool(live) and len(predict_ok) == len(rows),
            "detail": (
                f"{len(predict_ok)}/{len(rows)} refits reproduce R's newdata predictions "
                f"within atol={PREDICT_ATOL:g}" if live else "live replay skipped"
            ),
        },
        "serialization_roundtrip": {
            "met": bool(live) and len(roundtrip_ok) == len(rows),
            "detail": (
                f"{len(roundtrip_ok)}/{len(rows)} JSON round-trips drift <= "
                f"{ROUNDTRIP_RTOL:g}" if live else "live replay skipped"
            ),
        },
        "failures_explainable": {
            "met": bool(p2["verdict"]["all_failures_classified"])
            and not unknown_layer_codes,
            "detail": (
                "every non-pass carries a category: P2 classification "
                f"({p2['verdict']['non_pass_total']} entries) plus "
                f"layered r_error={layered['problems']['r_error']}, "
                f"r_not_converged={len(layered['problems']['r_not_converged'])}, "
                f"flat_ridge={len(layered['problems']['flat_ridge'])}"
            ),
        },
    }
    if live:
        criteria["convergence_complete"] = {
            "met": len(converged_ok) == len(rows),
            "detail": f"{len(converged_ok)}/{len(rows)} refits report converged=True",
        }

    gate = tuple(criteria)
    m2_gate_met = all(criteria[name]["met"] for name in gate)

    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": "M2",
        "target_version": "0.6.0",
        "live_replay": bool(live),
        "promoted_families": promoted,
        "sources": {
            "golden_manifest": "omnilss/tests/fixtures/r_golden/regenerated/manifest.json",
            "frozen_families": "docs/development/highfreq-families-frozen.json",
            "p2_classification": "docs/development/p2-defect-classification-2026-11-15.json",
            "layered_parity": "docs/development/layered-parity-2026-11-15.json",
        },
        "tolerances": {
            "predict_atol": PREDICT_ATOL,
            "predict_rtol": PREDICT_RTOL,
            "roundtrip_rtol": ROUNDTRIP_RTOL,
        },
        "counts": {
            "promoted_families": len(rows),
            "static_evidence_complete": len(static_complete),
            "live_predictions_ok": len(predict_ok),
            "live_roundtrip_ok": len(roundtrip_ok),
            "live_converged_ok": len(converged_ok),
            "live_errors": sorted(row["family"] for row in rows if row.get("error")),
        },
        "rows": rows,
        "criteria": criteria,
        "verdict": {
            "m2_gate_met": m2_gate_met,
            "gate_criteria": list(gate),
            "notes": (
                "M2 is met when the P2 classification is exhaustive, our three fitting "
                "routes have no mismatch, every promoted fixture carries fit + newdata + "
                "convergence evidence, our refits reproduce R's newdata predictions and "
                "survive a JSON round-trip, and every non-pass has a category. flat_ridge "
                "and R-side errors are classified, not counted as our failures."
            ),
        },
    }


def _stable(artifact: dict) -> dict:
    return {
        "schema_version": artifact["schema_version"],
        "milestone": artifact["milestone"],
        "live_replay": artifact["live_replay"],
        "promoted_families": artifact["promoted_families"],
        "counts": artifact["counts"],
        "criteria": artifact["criteria"],
        "verdict": artifact["verdict"],
    }


def _render_markdown(a: dict) -> str:
    c = a["counts"]
    lines = [
        f"# M2 assessment (target {a['target_version']})",
        "",
        f"- live replay: {a['live_replay']}",
        f"- promoted families: {c['promoted_families']}",
        f"- static fit+newdata evidence: {c['static_evidence_complete']}/{c['promoted_families']}",
        f"- live newdata predictions ok: {c['live_predictions_ok']}/{c['promoted_families']}",
        f"- live JSON round-trip ok: {c['live_roundtrip_ok']}/{c['promoted_families']}",
        f"- live converged: {c['live_converged_ok']}/{c['promoted_families']}",
        f"- live errors: {c['live_errors'] or '(none)'}",
        "",
        "| criterion | met | detail |",
        "|---|---|---|",
    ]
    for name, item in a["criteria"].items():
        lines.append(f"| {name} | {'yes' if item['met'] else 'no'} | {item['detail']} |")
    lines += [
        "",
        f"**m2_gate_met: {a['verdict']['m2_gate_met']}**",
        "",
        a["verdict"]["notes"],
        "",
        "## Per-family evidence",
        "",
        "| family | case | dev | coef blocks | newdata | predicted | convergence | pred gap | roundtrip |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in a["rows"]:
        if row.get("case") is None:
            lines.append(f"| {row['family']} | (missing) | | | | | | | |")
            continue
        s = row.get("static", {})
        live = row.get("live", {})
        gap = f"{live.get('predicted_max_abs_gap', float('nan')):.2e}" if live else "-"
        trip = f"{live.get('roundtrip_max_abs_drift', float('nan')):.1e}" if live else "-"
        lines.append(
            f"| {row['family']} | {row['case']} | {s.get('has_deviance')} | "
            f"{len(s.get('estimated_coefficient_blocks', []))} | {s.get('has_newdata')} | "
            f"{s.get('has_predicted')} | {s.get('has_convergence')} | {gap} | {trip} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_artifact(artifact: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    path.with_suffix(".md").write_text(_render_markdown(artifact), encoding="utf-8")


def check_artifact(path: Path) -> tuple[bool, list[str]]:
    problems: list[str] = []
    artifact = _read(path)
    for key in ("schema_version", "milestone", "live_replay", "counts", "criteria", "verdict"):
        if key not in artifact:
            problems.append(f"missing key: {key}")
    if problems:
        return False, problems
    current = build(live=bool(artifact["live_replay"]))
    if _stable(artifact) != _stable(current):
        for key in ("promoted_families", "counts", "criteria", "verdict"):
            if artifact.get(key) != current.get(key):
                problems.append(f"{key} drifted vs the current tree")
    return (not problems), problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--path", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-live", action="store_true", help="static evidence only")
    args = parser.parse_args(argv)

    if args.check:
        ok, problems = check_artifact(args.path)
        print(f"M2 assessment check: {args.path}")
        if ok:
            print("result: OK")
            return 0
        for problem in problems:
            print(f"  - {problem}")
        print("result: FAIL")
        return 1

    artifact = build(live=not args.no_live)
    log = (lambda *a: print(*a, file=sys.stderr)) if args.json else print
    if args.write is not None:
        write_artifact(artifact, args.write)
        log(f"wrote {args.write}")
        log(f"wrote {args.write.with_suffix('.md')}")
    if args.json or args.write is None:
        print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False))
    counts = artifact["counts"]
    log(
        f"families={counts['promoted_families']} static={counts['static_evidence_complete']} "
        f"predict={counts['live_predictions_ok']} roundtrip={counts['live_roundtrip_ok']} "
        f"converged={counts['live_converged_ok']} errors={counts['live_errors']}"
    )
    log(f"m2_gate_met={artifact['verdict']['m2_gate_met']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
