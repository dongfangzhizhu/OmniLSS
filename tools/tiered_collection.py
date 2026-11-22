# SPDX-License-Identifier: GPL-3.0-or-later
"""D0.3: tiered pytest collection gate for this project's own test surface.

Four tiers cover every ``test_*.py`` owned by this repository:

=============  ==========================================================
tier           what it collects
=============  ==========================================================
omnilss        ``omnilss/tests`` (main suite; ``omnilss/pyproject.toml`` config)
omnilss-server ``omnilss-server`` (L3 FastAPI service tests)
omnilss-pro    ``omnilss-pro`` (src-layout commercial prototype)
benchmarks     ``benchmarks`` benchmark/diagnostic scripts
=============  ==========================================================

``ondil/`` and ``gamlss-python/`` are **reference projects** compared in the
README; they are deliberately not tiers (see ``REFERENCE_PROJECTS``) and the
test suite asserts they can never become tiers.

Gate semantics (fail-closed, matching the deliverable plan D0.3):

* ``--mode collect`` (default): each tier must collect with **zero errors** and
  at least one test; ``collected``/``errors``/``skipped`` are reported per tier
  and as totals. ``skipped`` here counts *collection-time* skips (module-level
  ``importorskip``/``skip``), i.e. tests that would never run.
* ``--mode run``: the tier must run green; ``passed``/``failed``/``skipped``/
  ``errors`` are reported.
* ``--strict-skips``: additionally fail on any skipped test (the D7 rule
  "no hidden skips"); CI enables it with the full dev extras installed.

Exit code 0 only when every selected tier passes its gate, so CI can simply run
this command. ``--json`` emits the machine-readable report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Reference projects kept for comparison only; never tiers, never run here.
REFERENCE_PROJECTS = ("ondil", "gamlss-python")

#: Collection exclusions inside the benchmarks tier: standalone diagnostic
#: scripts with import-time dependencies on data files / the R bridge that are
#: not part of any suite. Tracked explicitly so NEW breakage still fails.
BENCHMARK_KNOWN_EXCLUSIONS = (
    "diagnostic/test_bi_performance.py",
    # Import-time Rscript calls (same class as the exclusions below).
    "diagnostic/test_convergence_tolerance.py",
    "diagnostic/test_initial_values.py",
    "diagnostic/test_bi_r_consistency.py",
    "diagnostic/test_python_with_r_data.py",
    "diagnostic/test_verbose_python.py",
    "diagnostic/test_with_r_data.py",
)


@dataclass(frozen=True)
class Tier:
    """One independently collected test surface of this repository."""

    name: str
    cwd: str
    args: tuple[str, ...] = ()
    ignore: tuple[str, ...] = ()
    description: str = ""

    @property
    def root(self) -> Path:
        return REPO_ROOT / self.cwd


TIERS: dict[str, Tier] = {
    tier.name: tier
    for tier in (
        Tier("omnilss", "omnilss", ("tests",), (), "主套件（omnilss/pyproject.toml 配置）"),
        Tier("omnilss-server", "omnilss-server", (), (), "L3 FastAPI 服务层测试"),
        Tier("omnilss-pro", "omnilss-pro", (), (), "商业扩展原型（src 布局）"),
        Tier(
            "benchmarks",
            "benchmarks",
            (),
            BENCHMARK_KNOWN_EXCLUSIONS,
            "基准/诊断脚本（5 个已知缺数据文件的脚本显式排除）",
        ),
    )
}

_COLLECTED_RE = re.compile(r"(\d+) tests? collected")
_FILE_COUNT_RE = re.compile(r"(?m)^(.*\.py): (\d+)$")
_COLLECT_SKIP_RE = re.compile(r"(\d+) skipped")
_SKIPPED_LINE_RE = re.compile(r"(?m)^SKIPPED\s+\[(\d+)\]")
_ERROR_LINE_RE = re.compile(r"(?m)^ERROR\s+")
_SUMMARY_RE = re.compile(r"(\d+) (passed|skipped|failed|errors?|xfailed|xpassed)\b")


def build_command(tier: Tier, *, mode: str = "collect", python: str | None = None) -> list[str]:
    """Build the pytest command for one tier (runs in ``tier.root``)."""
    command = [python or sys.executable, "-m", "pytest", "--no-header", "-p", "no:cacheprovider"]
    if mode == "collect":
        command.extend(["-q", "--collect-only"])
    for path in tier.ignore:
        command.append(f"--ignore={path}")
    command.extend(tier.args)
    return command


def parse_collection(output: str) -> dict:
    """Extract ``collected``/``errors``/``skipped`` from ``--collect-only -q``.

    Quiet levels may be doubled by pytest config, which suppresses the final
    ``N tests collected`` line; in that case the per-file ``path: count`` lines
    are summed instead. Both shapes are seen in CI and in tier directories that
    ship their own pytest config.

    ``skipped`` counts *collection-time* skips (module-level ``importorskip`` /
    ``skip`` markers), i.e. tests that would never run. They are reported and,
    with ``--strict-skips``, treated as gate failures.
    """
    match = _COLLECTED_RE.search(output)
    if match:
        collected = int(match.group(1))
    else:
        collected = sum(int(count) for _path, count in _FILE_COUNT_RE.findall(output))
    skip_match = _COLLECT_SKIP_RE.search(output)
    skipped = (
        int(skip_match.group(1))
        if skip_match
        else sum(int(count) for count in _SKIPPED_LINE_RE.findall(output))
    )
    return {
        "collected": collected,
        "errors": len(_ERROR_LINE_RE.findall(output)),
        "skipped": skipped,
    }


def parse_summary(output: str) -> dict:
    """Extract the final run summary (last occurrence of each stat wins)."""
    stats: dict[str, int] = {}
    for count, name in _SUMMARY_RE.findall(output):
        key = "error" if name.startswith("error") else name
        stats[key] = int(count)
    return {
        "passed": stats.get("passed", 0),
        "failed": stats.get("failed", 0),
        "skipped": stats.get("skipped", 0),
        "errors": stats.get("error", 0),
    }


def evaluate_gate(
    *,
    mode: str,
    exit_code: int,
    collected: int,
    errors: int,
    failed: int,
    skipped: int = 0,
    strict_skips: bool = False,
) -> list[str]:
    """Return the list of gate violations (empty means the tier passed)."""
    reasons: list[str] = []
    if errors:
        reasons.append(f"collection/execution errors: {errors}")
    if failed:
        reasons.append(f"failed tests: {failed}")
    if strict_skips and skipped:
        reasons.append(f"skipped tests are not allowed (--strict-skips): {skipped}")
    if exit_code == 5 or collected == 0:
        reasons.append("no tests collected")
    elif exit_code != 0:
        reasons.append(f"pytest exit code {exit_code}")
    return reasons


def run_tier(
    tier: Tier,
    *,
    mode: str = "collect",
    runner=None,
    python: str | None = None,
    timeout: float = 1800.0,
    strict_skips: bool = False,
) -> dict:
    """Run one tier and return its gate result (never raises on pytest failure)."""
    runner = runner or subprocess.run
    command = build_command(tier, mode=mode, python=python)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        result = runner(
            command,
            cwd=str(tier.root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        output = (result.stdout or "") + (result.stderr or "")
        exit_code = int(result.returncode)
    except FileNotFoundError as error:  # pragma: no cover - python binary missing
        output, exit_code = f"runner unavailable: {error}", 127
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or "") + (error.stderr or "") if isinstance(error.stdout, str) else ""
        output += f"\nTIMEOUT after {timeout}s"
        exit_code = 124
    if mode == "collect":
        parsed = parse_collection(output)
        counts = {
            "collected": parsed["collected"],
            "errors": parsed["errors"],
            "skipped": parsed["skipped"],
            "passed": None,
            "failed": None,
        }
    else:
        summary = parse_summary(output)
        counts = {
            "collected": summary["passed"] + summary["failed"] + summary["skipped"] + summary["errors"],
            "errors": summary["errors"],
            "skipped": summary["skipped"],
            "passed": summary["passed"],
            "failed": summary["failed"],
        }
    reasons = evaluate_gate(
        mode=mode,
        exit_code=exit_code,
        collected=counts["collected"],
        errors=counts["errors"],
        failed=counts["failed"] or 0,
        skipped=counts["skipped"],
        strict_skips=strict_skips,
    )
    return {
        "tier": tier.name,
        "cwd": tier.cwd,
        "description": tier.description,
        "mode": mode,
        "command": command,
        "exit_code": exit_code,
        **counts,
        "strict_skips": strict_skips,
        "excluded_from_collection": list(tier.ignore),
        "ok": not reasons,
        "reasons": reasons,
        "log_tail": [line for line in output.splitlines() if line.strip()][-25:],
    }


def build_report(results: list[dict], *, mode: str, strict_skips: bool = False) -> dict:
    """Aggregate per-tier results into the CI-facing report."""
    totals = {
        "tiers": len(results),
        "collected": sum(entry["collected"] for entry in results),
        "errors": sum(entry["errors"] for entry in results),
        "skipped": sum(entry["skipped"] for entry in results),
        "failed": sum(entry["failed"] or 0 for entry in results),
    }
    return {
        "mode": mode,
        "strict_skips": strict_skips,
        "tiers": results,
        "totals": totals,
        "reference_projects_excluded": list(REFERENCE_PROJECTS),
        "ok": all(entry["ok"] for entry in results),
    }


def render_report(report: dict) -> str:
    """Human readable rendering of a tiered collection report."""
    suffix = " --strict-skips" if report.get("strict_skips") else ""
    lines = [f"tiered collection ({report['mode']}{suffix})"]
    for entry in report["tiers"]:
        status = "ok" if entry["ok"] else "FAIL"
        lines.append(
            f"[{status}] {entry['tier']:<15} collected={entry['collected']:<5} "
            f"errors={entry['errors']:<3} skipped={entry['skipped']:<4} exit={entry['exit_code']}"
        )
        for reason in entry["reasons"]:
            lines.append(f"    - {reason}")
    totals = report["totals"]
    lines.append(
        f"totals: tiers={totals['tiers']} collected={totals['collected']} "
        f"errors={totals['errors']} skipped={totals['skipped']} failed={totals['failed']}"
    )
    lines.append(f"reference projects not run here: {', '.join(report['reference_projects_excluded'])}")
    lines.append(f"result: {'OK' if report['ok'] else 'FAIL-CLOSED'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, runner=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tier",
        action="append",
        choices=sorted(TIERS),
        dest="tiers",
        help="run only this tier (repeatable; default: all four)",
    )
    parser.add_argument(
        "--mode",
        choices=("collect", "run"),
        default="collect",
        help="collect = zero-error collection gate (default); run = execute the tier",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--list", action="store_true", help="list the tiers and exit")
    parser.add_argument(
        "--strict-skips",
        action="store_true",
        help="fail when any tier reports skipped tests (D7 'no hidden skips')",
    )
    parser.add_argument("--timeout", type=float, default=1800.0, help="per-tier timeout in seconds")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # readable reports on Windows consoles
        except (ValueError, OSError):  # pragma: no cover - wrapped streams (pytest capsys)
            pass
    if args.list:
        for tier in TIERS.values():
            print(f"{tier.name:<16} {tier.cwd:<16} {tier.description}")
        print(f"reference projects (never tiers): {', '.join(REFERENCE_PROJECTS)}")
        return 0
    selected = [TIERS[name] for name in (args.tiers or list(TIERS))]
    results = [
        run_tier(
            tier,
            mode=args.mode,
            runner=runner,
            timeout=args.timeout,
            strict_skips=args.strict_skips,
        )
        for tier in selected
    ]
    report = build_report(results, mode=args.mode, strict_skips=args.strict_skips)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(render_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
