"""Validate that the three-month execution checklist is complete and ordered.

This tool enforces the execution rule used by the core engineering roadmap:
work should proceed strictly by week, and each week in the 12-week plan
must be marked complete before follow-up work starts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CHECKLIST_PATH = ROOT / "docs/development/three-month-core-engineering-checklist-2026-05-20.md"

_WEEK_PATTERN = re.compile(r"^- \[(?P<done>[ xX])\] Week\s+(?P<week>\d+)\b")


def validate_three_month_checklist(path: Path = CHECKLIST_PATH) -> dict[str, Any]:
    """Return machine-readable validation status for the three-month checklist."""

    lines = path.read_text(encoding="utf-8").splitlines()
    week_states: list[tuple[int, bool, str]] = []
    issues: list[dict[str, Any]] = []

    for lineno, line in enumerate(lines, start=1):
        match = _WEEK_PATTERN.match(line.strip())
        if not match:
            continue
        week = int(match.group("week"))
        done = match.group("done").lower() == "x"
        week_states.append((week, done, line.strip()))
        if not done:
            issues.append(
                {
                    "code": "WEEK_NOT_COMPLETE",
                    "week": week,
                    "line": lineno,
                    "message": f"Week {week} is not marked complete.",
                }
            )

    expected_weeks = list(range(1, 13))
    observed_weeks = [week for week, _, _ in week_states]
    unique_weeks_in_order = list(dict.fromkeys(observed_weeks))
    if unique_weeks_in_order != expected_weeks:
        issues.append(
            {
                "code": "WEEK_SEQUENCE_MISMATCH",
                "expected": expected_weeks,
                "observed": unique_weeks_in_order,
                "message": "Checklist week entries are missing, duplicated, or out of order.",
            }
        )

    checklist_ref = str(path)
    if path.is_relative_to(ROOT):
        checklist_ref = str(path.relative_to(ROOT))

    return {
        "ok": not issues,
        "checklist": checklist_ref,
        "weeks_observed": unique_weeks_in_order,
        "issues": issues,
    }


if __name__ == "__main__":
    report = validate_three_month_checklist()
    if report["ok"]:
        print("Three-month checklist is complete and in strict Week 1-12 order.")
        raise SystemExit(0)

    print("Three-month checklist validation failed.")
    for issue in report["issues"]:
        print(f"- {issue['code']}: {issue['message']}")
    raise SystemExit(1)
