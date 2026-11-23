"""Fail-closed gate for the Group B R-consistency suite (D2.308/R4).

The suite has a small, registered set of legitimate skips (vendored gamlss.*
sources absent, R not converging).  This gate parses the pytest log produced by
the run step and fails when:

* fewer than min_passed assertions ran (R was not really exercised),
* skipped tests exceed baseline_skipped + allowed_growth (mass skip),
* a skip reason does not start with an allowed prefix (new hidden skip),
* the log shows R itself was unusable.

It never re-runs the suite; it only inspects the existing log.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO_ROOT / "docs" / "development" / "group-b-skip-baseline.json"
PASSED_RE = re.compile(r"(\d+) passed")
SKIPPED_RE = re.compile(r"(\d+) skipped")
REASON_RE = re.compile(r"SKIPPED \[\d+\] [^:]+:\d+: (.+)")
R_UNUSABLE_MARKERS = (
    "Rscript: command not found",
    "Rscript: No such file or directory",
    "executable file not found in $PATH: Rscript",
)


def load_baseline(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"unsupported baseline schema_version: {data.get('schema_version')!r}")
    for key in ("baseline_skipped", "allowed_growth", "min_passed", "allowed_reason_prefixes"):
        if key not in data:
            raise ValueError(f"baseline missing {key}")
    if not data["allowed_reason_prefixes"]:
        raise ValueError("baseline has no allowed_reason_prefixes")
    return data


def evaluate(log_text: str, baseline: dict, min_passed: int | None = None) -> tuple[int, int, list[str]]:
    passed_match = PASSED_RE.search(log_text)
    skipped_match = SKIPPED_RE.search(log_text)
    passed = int(passed_match.group(1)) if passed_match else 0
    skipped = int(skipped_match.group(1)) if skipped_match else 0
    floor = baseline["min_passed"] if min_passed is None else min_passed
    problems: list[str] = []
    if passed < floor:
        problems.append(f"passed={passed} < required floor {floor}")
    limit = int(baseline["baseline_skipped"]) + int(baseline["allowed_growth"])
    if skipped > limit:
        problems.append(
            f"skipped={skipped} exceeds baseline {baseline['baseline_skipped']} + "
            f"allowed growth {baseline['allowed_growth']} (= {limit})"
        )
    prefixes = tuple(baseline["allowed_reason_prefixes"])
    unknown = sorted(
        {
            match.group(1).strip()
            for match in REASON_RE.finditer(log_text)
            if not match.group(1).strip().startswith(prefixes)
        }
    )
    if unknown:
        problems.append("unknown skip reason(s): " + "; ".join(unknown))
    for marker in R_UNUSABLE_MARKERS:
        if marker in log_text:
            problems.append(f"R is not usable: log contains {marker!r}")
    return passed, skipped, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--min-passed", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        baseline = load_baseline(args.baseline)
        log_text = args.log.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"::error::group_b_gate: {exc}")
        return 1
    passed, skipped, problems = evaluate(log_text, baseline, args.min_passed)
    if problems:
        for problem in problems:
            print(f"::error::group_b_gate: {problem}")
        return 1
    print(
        f"OK: Group B ran against pinned R (passed={passed}, skipped={skipped}, "
        f"baseline={baseline['baseline_skipped']} +{baseline['allowed_growth']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
