# SPDX-License-Identifier: GPL-3.0-or-later
"""M3 wrap-up assessment (D2.245).

DEVELOPMENT_PLAN section 22.4 states M3 as:

    P3 core API + family/link framework + admission smoothing closed;
    *.fix / contrasts on every route.

Each of those now has a deliverable, and this tool ties them together.  Two of
the criteria are checked with a *live probe* rather than by trusting a file:

  * gen.Family    -- build logNO and fit it; the RS deviance must match the live
                     R reference 140.3830665004935 (D2.243).
  * *.fix         -- RS / CG / MIXED with sigma.fix and sigma.start=0.5 must all
                     reach R's 67.1324138595735 with no sigma coefficient, and
                     CG / MIXED must rebuild categorical predictions from the
                     contrasts golden (D2.244).

The other three read committed artifacts: P3-1 contracts, the admission
inventory, and the M1/M2 verdicts (M3 must not break them).

Usage::

    python tools/m3_assessment.py --write docs/development/m3-assessment-2026-11-15.json
    python tools/m3_assessment.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "omnilss" / "src"))

CONTRACTS = ROOT / "docs" / "development" / "p3-contracts-2026-11-15.json"
ADMISSION = ROOT / "docs" / "development" / "m3-admission-evidence-2026-11-15.json"
M1 = ROOT / "docs" / "development" / "m1-assessment-2026-11-15.json"
M2 = ROOT / "docs" / "development" / "m2-assessment-2026-11-15.json"
CONTRASTS_GOLDEN = ROOT / "omnilss" / "tests" / "fixtures" / "r_golden" / "contrasts_2026-11-01.json"
DEFAULT_OUT = ROOT / "docs" / "development" / "m3-assessment-2026-11-15.json"

SCHEMA_VERSION = 1
R_LOGNO_DEV = 140.3830665004935
R_FIXED_SIGMA_DEV = 67.132413859573532
FIXED_SIGMA_START = 0.5
DEV_REL_TOL = 1e-6


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _probe_gen_family() -> dict:
    from omnilss import gamlss
    from omnilss.family_generator import Family

    rng = np.random.default_rng(42)
    n = 80
    x = rng.normal(size=n)
    y = np.exp(0.5 + 0.3 * x + rng.normal(scale=0.4, size=n))
    model = gamlss(
        "y ~ x", family=Family("NO", "log"), data={"y": y, "x": x},
        method="RS", max_iter=100,
    )
    gap = abs(float(model.g_dev) - R_LOGNO_DEV) / abs(R_LOGNO_DEV)
    return {
        "ok": bool(model.converged) and gap <= DEV_REL_TOL,
        "converged": bool(model.converged),
        "deviance_rel_gap_within_tolerance": gap <= DEV_REL_TOL,
    }


def _probe_fix_routes() -> dict:
    from omnilss import NO, gamlss

    rng = np.random.default_rng(5)
    n = 60
    x = rng.normal(size=n)
    y = 1.0 + 0.4 * x + rng.normal(scale=0.5, size=n)
    data = {"y": y, "x": x}
    rows = {}
    for method, kwargs in (
        ("RS", {"max_iter": 100}),
        ("CG", {"max_iter": 100}),
        ("MIXED", {"mixed_n1": 1, "mixed_n2": 100}),
    ):
        model = gamlss(
            "y ~ x", family=NO(), data=data, method=method,
            sigma_fix=True, sigma_start=FIXED_SIGMA_START, **kwargs,
        )
        gap = abs(float(model.g_dev) - R_FIXED_SIGMA_DEV) / abs(R_FIXED_SIGMA_DEV)
        rows[method] = {
            "deviance_matches_r": gap <= DEV_REL_TOL,
            "sigma_not_estimated": "sigma" not in model.coefficients,
            "sigma_frozen_at_start": bool(
                np.allclose(
                    np.asarray(model.fitted_values["sigma"], dtype=float),
                    FIXED_SIGMA_START,
                    rtol=0.0,
                    atol=1e-12,
                )
            ),
        }
    return {
        "ok": all(all(row.values()) for row in rows.values()),
        "routes": rows,
        "sigma_start": FIXED_SIGMA_START,
    }


def _probe_categorical_prediction() -> dict:
    from omnilss import NO, gamlss
    from omnilss.prediction import predict_params

    golden = _read(CONTRASTS_GOLDEN)
    data = {key: np.asarray(value) for key, value in golden["data"].items()}
    if data["g"].dtype.kind in "UO" or data["g"].dtype == object:
        data["g"] = data["g"].astype(str)
    newdata = {"g": np.array(["a", "c"]), "x": np.array([0.0, 1.0])}
    rows = {}
    for method, kwargs in (
        ("CG", {"max_iter": 30}),
        ("MIXED", {"mixed_n1": 1, "mixed_n2": 30}),
    ):
        model = gamlss(
            "y ~ g + x", family=NO(), data=data, method=method,
            contrasts="contr.treatment", **kwargs,
        )
        reference = np.asarray(golden["treatment"]["pred_mu"], dtype=float)
        predicted = np.asarray(predict_params(model, newdata)["mu"], dtype=float)
        target = float(golden["treatment"]["deviance"])
        rows[method] = {
            "deviance_matches_golden": abs(float(model.g_dev) - target) / abs(target)
            <= 1e-6,
            "predictions_match_golden": bool(
                np.allclose(predicted, reference, rtol=1e-8, atol=1e-8)
            ),
        }
    return {"ok": all(all(row.values()) for row in rows.values()), "routes": rows}


def build() -> dict:
    contracts = _read(CONTRACTS)
    admission = _read(ADMISSION)
    m1 = _read(M1)
    m2 = _read(M2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gen_family = _probe_gen_family()
        fix_routes = _probe_fix_routes()
        categorical = _probe_categorical_prediction()

    admitted = int(admission["counts"]["admitted"])
    promoted = int(admission["counts"]["promoted_families"])

    criteria = {
        "p3_contracts_frozen": {
            "met": bool(contracts["verdict"]["contract_frozen"]),
            "detail": (
                "family / link / formula / prediction contracts are frozen and "
                "checked against the live package (D2.242)"
            ),
        },
        "gen_family_fits_like_r": {
            "met": bool(gen_family["ok"]),
            "detail": (
                "logNO RS fit reproduces R deviance 140.3831 (D2.243 derivative "
                "wrapper); converged="
                + str(gen_family["converged"])
            ),
        },
        "fix_on_every_route": {
            "met": bool(fix_routes["ok"]),
            "detail": (
                "sigma.fix=TRUE with sigma.start=0.5 reaches R deviance "
                "67.1324138595735 on RS/CG/MIXED with no sigma coefficient (D2.244)"
            ),
        },
        "categorical_prediction_rebuilt": {
            "met": bool(categorical["ok"]),
            "detail": (
                "CG/MIXED factor fits rebuild predictions from the contrasts golden "
                "for both the deviance and pred_mu (D2.244 intercept fix)"
            ),
        },
        "admission_closed": {
            "met": admitted == promoted and promoted > 0,
            "detail": f"{admitted}/{promoted} promoted families are production_safe (D2.241)",
        },
        "prior_milestones_intact": {
            "met": bool(m1["verdict"]["m1_gate_met"]) and bool(m2["verdict"]["m2_gate_met"]),
            "detail": "M1 and M2 gate verdicts are still true",
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": "M3",
        "target_version": "0.9.0",
        "decision": "D2.245",
        "deliverables": {
            "P3-1": "docs/development/p3-contracts-2026-11-15.json",
            "P3-2": "omnilss/src/omnilss/family_generator.py + test_family_generator.py",
            "P3-3": "omnilss/tests/test_p3_fix_methods.py",
            "P3-4": "docs/development/m3-admission-evidence-2026-11-15.json",
        },
        "probes": {
            "gen_family": gen_family,
            "fix_routes": fix_routes,
            "categorical_prediction": categorical,
        },
        "criteria": criteria,
        "counts": {
            "criteria": len(criteria),
            "met": sum(1 for item in criteria.values() if item["met"]),
            "admitted_families": admitted,
            "promoted_families": promoted,
        },
        "verdict": {
            "m3_gate_met": all(item["met"] for item in criteria.values()),
            "notes": (
                "M3 is met when P3-1..P3-4 are closed and the two behavioural claims "
                "(gen.Family fitting, fix across routes) hold on a live probe, not "
                "only in a test name. M4 (surrounding packages, services, docs, "
                "release artifacts) is untouched by this verdict."
            ),
        },
    }


def _stable(artifact: dict) -> dict:
    return {
        "schema_version": artifact["schema_version"],
        "milestone": artifact["milestone"],
        "deliverables": artifact["deliverables"],
        "probes": artifact["probes"],
        "criteria": artifact["criteria"],
        "counts": artifact["counts"],
        "verdict": artifact["verdict"],
    }


def _render_markdown(a: dict) -> str:
    lines = [
        "# M3 assessment (target 0.9.0)",
        "",
        f"- decision: {a['decision']}",
        f"- criteria met: {a['counts']['met']}/{a['counts']['criteria']}",
        f"- admitted families: {a['counts']['admitted_families']}/{a['counts']['promoted_families']}",
        "",
        "| criterion | met | detail |",
        "|---|---|---|",
    ]
    for name, item in a["criteria"].items():
        lines.append(f"| {name} | {'yes' if item['met'] else 'no'} | {item['detail']} |")
    lines += ["", "## Live probes", ""]
    gf = a["probes"]["gen_family"]
    lines.append(f"- gen.Family: ok={gf['ok']}, converged={gf['converged']}")
    lines += ["", "| route | fix deviance | sigma not estimated | sigma frozen |", "|---|---|---|---|"]
    for route, row in a["probes"]["fix_routes"]["routes"].items():
        lines.append(
            f"| {route} | {row['deviance_matches_r']} | {row['sigma_not_estimated']} | "
            f"{row['sigma_frozen_at_start']} |"
        )
    lines += ["", "| route | categorical deviance | categorical prediction |", "|---|---|---|"]
    for route, row in a["probes"]["categorical_prediction"]["routes"].items():
        lines.append(
            f"| {route} | {row['deviance_matches_golden']} | {row['predictions_match_golden']} |"
        )
    lines += [
        "",
        f"**m3_gate_met: {a['verdict']['m3_gate_met']}**",
        "",
        a["verdict"]["notes"],
        "",
        "## Deliverables",
        "",
    ]
    for key, value in a["deliverables"].items():
        lines.append(f"- {key}: {value}")
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
    for key in ("schema_version", "milestone", "deliverables", "probes", "criteria", "counts", "verdict"):
        if key not in artifact:
            problems.append(f"missing key: {key}")
    if problems:
        return False, problems
    current = build()
    if _stable(artifact) != _stable(current):
        for key in ("probes", "criteria", "counts", "verdict", "deliverables"):
            if artifact.get(key) != current.get(key):
                problems.append(f"{key} drifted vs the current tree")
    return (not problems), problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--path", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        ok, problems = check_artifact(args.path)
        print(f"M3 assessment check: {args.path}")
        if ok:
            print("result: OK")
            return 0
        for problem in problems:
            print(f"  - {problem}")
        print("result: FAIL")
        return 1

    artifact = build()
    log = (lambda *a: print(*a, file=sys.stderr)) if args.json else print
    if args.write is not None:
        write_artifact(artifact, args.write)
        log(f"wrote {args.write}")
        log(f"wrote {args.write.with_suffix('.md')}")
    if args.json or args.write is None:
        print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False))
    counts = artifact["counts"]
    log(f"criteria={counts['met']}/{counts['criteria']} m3_gate_met={artifact['verdict']['m3_gate_met']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
