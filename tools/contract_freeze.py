# SPDX-License-Identifier: GPL-3.0-or-later
"""P3-1: freeze the family / link / formula / prediction contracts (D2.242).

DEVELOPMENT_PLAN P3-1 asks to freeze the four contracts M3 builds on.  A
contract that only lives in prose cannot fail a build, so this tool pins it as a
checked-in artifact and verifies it against the live package:

  family      FamilyDefinition's dataclass fields and, for every promoted family,
              its parameter tuple, nopar, type, fixed parameters, estimable
              parameters, d/p/q/r callability and declared link keys.
  link        LinkGamlss' five R slots, the supported link names, and a
              round-trip probe per link.  logshiftto1 / logshiftto2 reproduce
              R's asymmetric branch and therefore round-trip to mu + 1e-5 by
              design -- freezing that offset is the point, because "fixing" it
              would silently diverge from R.
  formula     ParsedFormula's fields and the parsed structure of canonical
              formulas (linear / smooth / tensor terms and the intercept flag).
  prediction  predict_params / predict_response behaviour on a real NO fit:
              keys are the estimable parameters, shapes follow newdata, a
              missing column raises PredictionSchemaError (a ValueError), an
              unused column is tolerated.

The tool is fail-closed in both directions: a new supported link with no probe
fails, and a changed family parameter tuple fails --check.

Usage::

    python tools/contract_freeze.py --write docs/development/p3-contracts-2026-11-15.json
    python tools/contract_freeze.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import fields
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "omnilss" / "src"))

SCHEMA_VERSION = 1
FROZEN_FAMILIES = ROOT / "docs" / "development" / "highfreq-families-frozen.json"
DEFAULT_OUT = ROOT / "docs" / "development" / "p3-contracts-2026-11-15.json"

#: Probe domain per link and the documented inverse offset.  logshiftto0 / Slog
#: round-trip exactly; logshiftto1 / 2 land on mu + 1e-5 exactly as R does.
LINK_PROBES: dict[str, dict[str, object]] = {
    "identity": {"domain": [-2.0, -1.0, 0.0, 1.0, 2.0], "offset": 0.0},
    "log": {"domain": [0.25, 0.5, 1.0, 2.0], "offset": 0.0},
    "logit": {"domain": [0.1, 0.3, 0.5, 0.7, 0.9], "offset": 0.0},
    "probit": {"domain": [0.1, 0.3, 0.5, 0.7, 0.9], "offset": 0.0},
    "cauchit": {"domain": [0.1, 0.3, 0.5, 0.7, 0.9], "offset": 0.0},
    "cloglog": {"domain": [0.1, 0.3, 0.5, 0.7, 0.9], "offset": 0.0},
    "inverse": {"domain": [0.25, 0.5, 1.0, 2.0], "offset": 0.0},
    "1/mu^2": {"domain": [0.25, 0.5, 1.0, 2.0], "offset": 0.0},
    "mu^2": {"domain": [0.25, 0.5, 1.0, 2.0], "offset": 0.0},
    "sqrt": {"domain": [0.25, 0.5, 1.0, 2.0], "offset": 0.0},
    "slog": {"domain": [0.25, 0.5, 1.0, 2.0], "offset": 0.0},
    "logshiftto0": {"domain": [0.25, 0.5, 1.0, 2.0], "offset": 0.0},
    "logshiftto1": {"domain": [1.5, 2.0, 3.0, 4.0], "offset": 1e-5},
    "logshiftto2": {"domain": [2.5, 3.0, 4.0, 5.0], "offset": 1e-5},
    "[-1,1]": {"domain": [0.1, 0.3, 0.5, 0.7, 0.9], "offset": 0.0},
    "(0,2]": {"domain": [0.25, 0.5, 1.5, 2.0], "offset": 0.0},
    "(0,5]": {"domain": [0.5, 1.0, 3.0, 5.0], "offset": 0.0},
    "(1,3]": {"domain": [1.5, 2.0, 2.5, 3.0], "offset": 0.0},
}
ROUNDTRIP_TOL = 1e-9

FORMULA_CASES = (
    "y ~ x",
    "y ~ x + z",
    "y ~ x - 1",
    "y ~ 1",
    "y ~ pb(x)",
    "y ~ s(x, k=3)",
    "y ~ te(x, z)",
)


def _promoted() -> list[str]:
    data = json.loads(FROZEN_FAMILIES.read_text(encoding="utf-8"))
    return sorted(item["family"] for item in data["tiers"]["promoted"])


def _family_area() -> dict:
    from dataclasses import fields as dataclass_fields

    from omnilss.distribution_registry import resolve
    from omnilss.families import FamilyDefinition

    families = {}
    for name in _promoted():
        family = resolve(name)
        families[name] = {
            "parameters": list(family.parameters),
            "nopar": int(family.nopar),
            "type": str(family.type),
            "fixed_parameters": sorted(family.fixed_parameters or ()),
            "estimable_parameters": list(family.estimable_parameters),
            "density_callable": callable(family.d),
            "cdf_callable": callable(family.p),
            "quantile_callable": callable(family.q),
            "sampler_callable": callable(family.r),
            "link_keys": sorted((family.links or {}).keys()),
        }
    return {
        "dataclass_fields": sorted(field.name for field in dataclass_fields(FamilyDefinition)),
        "members": ["pdf", "nopar", "estimable_parameters", "from_mapping"],
        "families": families,
    }


def _link_area() -> dict:
    from omnilss.gamlss_family import LinkGamlss, _LINK_FACTORIES, make_link_gamlss

    supported = sorted(_LINK_FACTORIES)
    probes = {}
    for name, spec in LINK_PROBES.items():
        link = make_link_gamlss(name)
        mu = np.asarray(spec["domain"], dtype=float)
        eta = np.asarray(link.linkfun(mu), dtype=float)
        back = np.asarray(link.linkinv(eta), dtype=float)
        derivative = np.asarray(link.mu_eta(eta), dtype=float)
        target = mu + float(spec["offset"])
        error = float(np.max(np.abs(back - target)))
        probes[name] = {
            "roundtrip_within_tolerance": error <= ROUNDTRIP_TOL,
            "mu_eta_finite_nonzero": bool(
                np.all(np.isfinite(derivative)) and np.all(derivative != 0.0)
            ),
            "offset": float(spec["offset"]),
            "domain": [float(value) for value in spec["domain"]],
        }
    return {
        "class_members": ["name", "linkfun", "linkinv", "mu_eta", "valideta", "as_dict"],
        "supported_links": supported,
        "probes": probes,
    }


def _formula_area() -> dict:
    from omnilss.formula_parser import ParsedFormula, parse_formula

    cases = []
    for formula in FORMULA_CASES:
        parsed = parse_formula(formula)
        cases.append(
            {
                "formula": formula,
                "response": parsed.response,
                "linear_variables": [term.variable for term in parsed.linear_terms],
                "smooth_terms": [
                    {"smoother": term.smoother, "variable": term.variable}
                    for term in parsed.smooth_terms
                ],
                "tensor_terms": [
                    {"smoother": term.smoother, "variables": list(term.variables)}
                    for term in parsed.tensor_terms
                ],
                "has_intercept": bool(parsed.has_intercept),
            }
        )
    return {
        "fields": sorted(field.name for field in fields(ParsedFormula)),
        "cases": cases,
    }


def _prediction_area() -> dict:
    from omnilss import gamlss
    from omnilss.prediction import (
        PredictionSchemaError,
        predict_params,
        predict_response,
    )

    rng = np.random.default_rng(20261115)
    n = 40
    x = rng.normal(size=n)
    y = 1.0 + 0.4 * x + rng.normal(scale=0.6, size=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gamlss("y ~ x", family="NO", data={"y": y, "x": x}, max_iter=10)
    grid = np.array([-0.5, 0.0, 0.5])
    predicted = predict_params(model, {"x": grid})
    # The estimable set belongs to the family; the fitted model exposes it only
    # through its family object.
    estimable = list(model.family.estimable_parameters)

    missing_raises = False
    missing_error = None
    try:
        predict_params(model, {"z": grid})
    except PredictionSchemaError as exc:
        missing_raises = True
        missing_error = type(exc).__name__
    except Exception as exc:  # noqa: BLE001 - a different error is a contract break
        missing_error = f"{type(exc).__name__}: {exc}"[:200]

    extra = predict_params(model, {"x": grid, "unused": np.full(grid.shape, 9.0)})
    return {
        "functions": [
            "predict_params",
            "predict_response",
            "predict_quantiles",
            "centiles",
            "build_prediction_design_matrix",
        ],
        "schema_error_is_value_error": issubclass(PredictionSchemaError, ValueError),
        "keys_are_estimable_parameters": sorted(predicted) == sorted(estimable),
        "values_follow_newdata_length": all(
            len(np.ravel(value)) == grid.size for value in predicted.values()
        ),
        "missing_column_raises_schema_error": missing_raises,
        "missing_column_error": missing_error,
        "extra_column_tolerated": sorted(extra) == sorted(estimable),
        "predict_response_length_follows_newdata": len(
            np.ravel(predict_response(model, {"x": grid}))
        )
        == grid.size,
        "estimated_parameters": estimable,
        "newdata_points": int(grid.size),
    }


def build() -> dict:
    family = _family_area()
    link = _link_area()
    formula = _formula_area()
    prediction = _prediction_area()

    # Two legal fixed-parameter shapes exist: BI's bd is a density-only argument
    # listed outside the parameter tuple, while NET's nu/tau are inside the tuple
    # but marked fixed.  The invariant that covers both is: nopar counts the
    # parameter tuple, and the estimable tuple is exactly the parameters minus
    # the fixed ones.
    family_ok = all(
        row["nopar"] == len(row["parameters"])
        and row["estimable_parameters"]
        == [p for p in row["parameters"] if p not in row["fixed_parameters"]]
        and row["density_callable"]
        and row["cdf_callable"]
        and row["quantile_callable"]
        and row["sampler_callable"]
        for row in family["families"].values()
    )
    link_ok = (
        link["supported_links"] == sorted(LINK_PROBES)
        and all(
            probe["roundtrip_within_tolerance"] and probe["mu_eta_finite_nonzero"]
            for probe in link["probes"].values()
        )
    )
    prediction_ok = all(
        prediction[key]
        for key in (
            "schema_error_is_value_error",
            "keys_are_estimable_parameters",
            "values_follow_newdata_length",
            "missing_column_raises_schema_error",
            "extra_column_tolerated",
            "predict_response_length_follows_newdata",
        )
    )
    formula_ok = bool(formula["cases"]) and formula["fields"] == sorted(
        ["response", "linear_terms", "smooth_terms", "tensor_terms", "has_intercept"]
    )

    criteria = {
        "family_contract_holds": {
            "met": family_ok,
            "detail": (
                f"{len(family['families'])} promoted families: nopar == len(parameters), "
                "estimable == parameters minus fixed (covers BI's bd outside the tuple "
                "and NET's fixed nu/tau inside it), d/p/q/r callable"
            ),
        },
        "link_contract_holds": {
            "met": link_ok,
            "detail": (
                f"{len(link['supported_links'])} supported links each have a probe; "
                "round-trip within 1e-9 (including the documented 1e-5 offset for "
                "logshiftto1/2) and a finite non-zero mu_eta"
            ),
        },
        "formula_contract_holds": {
            "met": formula_ok,
            "detail": (
                f"ParsedFormula fields frozen; {len(formula['cases'])} canonical formulas "
                "parse to a recorded structure"
            ),
        },
        "prediction_contract_holds": {
            "met": prediction_ok,
            "detail": (
                "predict_params keys == estimable parameters, shapes follow newdata, "
                "missing column raises PredictionSchemaError, extra column tolerated"
            ),
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": "M3",
        "task": "P3-1",
        "decision": "D2.242",
        "areas": {
            "family": family,
            "link": link,
            "formula": formula,
            "prediction": prediction,
        },
        "sources": {
            "family": "omnilss/families.py + omnilss/distribution_registry.py",
            "link": "omnilss/gamlss_family.py",
            "formula": "omnilss/formula_parser.py",
            "prediction": "omnilss/prediction.py",
            "families": "docs/development/highfreq-families-frozen.json",
        },
        "criteria": criteria,
        "verdict": {
            "contract_frozen": all(item["met"] for item in criteria.values()),
            "notes": (
                "Frozen at the public shape M3 builds on. The link offsets are part of "
                "the contract, not a defect: logshiftto1/2 mirror R's asymmetric branch. "
                "A new link without a probe, a changed parameter tuple or a changed "
                "formula parse fails --check."
            ),
        },
    }


def _stable(artifact: dict) -> dict:
    return {
        "schema_version": artifact["schema_version"],
        "milestone": artifact["milestone"],
        "task": artifact["task"],
        "areas": artifact["areas"],
        "criteria": artifact["criteria"],
        "verdict": artifact["verdict"],
    }


def _render_markdown(a: dict) -> str:
    family = a["areas"]["family"]
    link = a["areas"]["link"]
    formula = a["areas"]["formula"]
    prediction = a["areas"]["prediction"]
    lines = [
        "# P3-1 frozen contracts (family / link / formula / prediction)",
        "",
        f"- decision: {a['decision']}",
        f"- promoted families: {len(family['families'])}",
        f"- supported links: {len(link['supported_links'])}",
        f"- formula cases: {len(formula['cases'])}",
        f"- contract frozen: {a['verdict']['contract_frozen']}",
        "",
        "| criterion | met | detail |",
        "|---|---|---|",
    ]
    for name, item in a["criteria"].items():
        lines.append(f"| {name} | {'yes' if item['met'] else 'no'} | {item['detail']} |")
    lines += ["", "## Families", "", "| family | parameters | nopar | estimable | fixed | links |", "|---|---|---|---|---|---|"]
    for name, row in family["families"].items():
        lines.append(
            f"| {name} | {', '.join(row['parameters'])} | {row['nopar']} | "
            f"{', '.join(row['estimable_parameters'])} | "
            f"{', '.join(row['fixed_parameters']) or '-'} | {', '.join(row['link_keys'])} |"
        )
    lines += ["", "## Links", "", "| link | offset | round-trip | mu_eta |", "|---|---|---|---|"]
    for name, probe in link["probes"].items():
        lines.append(
            f"| {name} | {probe['offset']:g} | {probe['roundtrip_within_tolerance']} | "
            f"{probe['mu_eta_finite_nonzero']} |"
        )
    lines += ["", "## Formulas", "", "| formula | linear | smooth | tensor | intercept |", "|---|---|---|---|---|"]
    for case in formula["cases"]:
        smooth = ", ".join(f"{t['smoother']}({t['variable']})" for t in case["smooth_terms"]) or "-"
        tensor = ", ".join(f"{t['smoother']}({','.join(t['variables'])})" for t in case["tensor_terms"]) or "-"
        lines.append(
            f"| {case['formula']} | {', '.join(case['linear_variables']) or '-'} | "
            f"{smooth} | {tensor} | {case['has_intercept']} |"
        )
    lines += ["", "## Prediction", ""]
    for key in (
        "schema_error_is_value_error",
        "keys_are_estimable_parameters",
        "values_follow_newdata_length",
        "missing_column_raises_schema_error",
        "extra_column_tolerated",
        "predict_response_length_follows_newdata",
    ):
        lines.append(f"- {key}: {prediction[key]}")
    lines += ["", a["verdict"]["notes"], ""]
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
    artifact = json.loads(path.read_text(encoding="utf-8"))
    for key in ("schema_version", "milestone", "task", "areas", "criteria", "verdict"):
        if key not in artifact:
            problems.append(f"missing key: {key}")
    if problems:
        return False, problems
    current = build()
    if _stable(artifact) != _stable(current):
        for key in ("areas", "criteria", "verdict"):
            if artifact.get(key) != current.get(key):
                problems.append(f"{key} drifted vs the live package")
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
        print(f"P3-1 contract check: {args.path}")
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
    log(
        f"families={len(artifact['areas']['family']['families'])} "
        f"links={len(artifact['areas']['link']['supported_links'])} "
        f"formulas={len(artifact['areas']['formula']['cases'])} "
        f"contract_frozen={artifact['verdict']['contract_frozen']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
