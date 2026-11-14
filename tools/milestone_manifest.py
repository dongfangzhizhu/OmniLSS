# SPDX-License-Identifier: GPL-3.0-or-later
"""D2.226 / M0: fixed milestone manifest (release evidence bundle).

DEVELOPMENT_PLAN section 22.4 requires every milestone to carry a *fixed
manifest* covering R / Python / JAX / NumPy / SciPy / x64 versions, a source
hash, seeds, tolerances and the commands, plus --require-r, no hidden
skips, the coverage diff and a reproducible commit.

This tool assembles that manifest from evidence already committed in the repo
(coverage artifacts, the pinned golden template, the golden manifest, the
development plan, git HEAD) plus live environment probing and optional gate
results, then writes a deterministic JSON (and a Markdown summary).
--check verifies the committed manifest still describes the current tree.

Nothing here re-runs the multi-hour suite: the caller passes a collection
report produced by tools/tiered_collection.py --mode collect --strict-skips
--json; the golden gate is probed with tools/golden_manifest_gate.py.

Usage::

    python tools/milestone_manifest.py --check
    python tools/milestone_manifest.py --write docs/development/milestone-manifest-m0.json \
        --collection-report tmp/collection_m0.json --require-r --verify-coverage
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVERAGE_JSON = ROOT / "docs" / "development" / "r-symbol-coverage.json"
COVERAGE_DIFF = ROOT / "docs" / "development" / "symbol-coverage-diff.json"
TEMPLATE = ROOT / "omnilss" / "tests" / "fixtures" / "r_golden" / "manifest_template.json"
GOLDEN_MANIFEST = ROOT / "omnilss" / "tests" / "fixtures" / "r_golden" / "regenerated" / "manifest.json"
PLAN = ROOT / "DEVELOPMENT_PLAN.md"
DEFAULT_OUT = ROOT / "docs" / "development" / "milestone-manifest-m0.json"

MILESTONE = "M0"
VERSION = "0.4.0"
SCHEMA_VERSION = 1

#: gate ids whose ok must be true for the milestone to be claimed.
GATE_KEYS = (
    "p0_clear",
    "coverage_regenerable",
    "tiered_collection",
    "devcontainer",
    "golden_manifest_gate",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _head() -> dict:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
    }


def _python_env() -> dict:
    import jax
    import numpy
    import scipy

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "jax": jax.__version__,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
    }


_R_PROBE = (
    'cat(R.version.string, "\\n", sep="");'
    'cat(as.character(tryCatch(packageVersion("gamlss"), error=function(e) "MISSING")), "\\n", sep="");'
    'cat(as.character(tryCatch(packageVersion("gamlss.dist"), error=function(e) "MISSING")), "\\n", sep="")'
)


def _r_env() -> dict:
    try:
        proc = subprocess.run(
            ["Rscript", "-e", _R_PROBE], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "r_version": None, "gamlss_version": None,
                "gamlss_dist_version": None}
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if proc.returncode != 0 or len(lines) < 3:
        return {"available": False, "r_version": None, "gamlss_version": None,
                "gamlss_dist_version": None}
    return {
        "available": True,
        "r_version": lines[0],
        "gamlss_version": lines[1],
        "gamlss_dist_version": lines[2],
    }


def _env_pinned_ok(env: dict, r_env: dict, pinned: dict) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    if not env["python"].startswith(pinned["python_version_prefix"]):
        mismatches.append(f"python {env['python']} !~ {pinned['python_version_prefix']}*")
    if not env["jax"].startswith(pinned["jax_version_prefix"]):
        mismatches.append(f"jax {env['jax']} !~ {pinned['jax_version_prefix']}*")
    if env["jax_enable_x64"] is not bool(pinned["jax_enable_x64"]):
        mismatches.append(f"jax_enable_x64 {env['jax_enable_x64']} != {pinned['jax_enable_x64']}")
    if not r_env["available"]:
        mismatches.append("R not available (--require-r would fail closed)")
    else:
        if not r_env["r_version"].startswith(pinned["R_version_prefix"]):
            mismatches.append(f"R {r_env['r_version']} !~ {pinned['R_version_prefix']}*")
        if r_env["gamlss_version"] != pinned["gamlss_version"]:
            mismatches.append(f"gamlss {r_env['gamlss_version']} != {pinned['gamlss_version']}")
        if r_env["gamlss_dist_version"] != pinned["gamlss_dist_version"]:
            mismatches.append(
                f"gamlss.dist {r_env['gamlss_dist_version']} != {pinned['gamlss_dist_version']}"
            )
    return (not mismatches), mismatches


def _coverage() -> dict:
    cov = _read_json(COVERAGE_JSON)
    diff = _read_json(COVERAGE_DIFF)
    stats = cov.get("stats", {})
    return {
        "stats": {
            "surface_missing": stats.get("surface_missing"),
            "surface_matched": stats.get("surface_matched"),
            "in_scope_exports": stats.get("in_scope_exports"),
        },
        "provenance": cov.get("provenance", {}),
        "diff_ok": bool(diff.get("ok")),
        "ratchet_ok": bool(diff.get("ratchet_ok")),
        "ceiling_ok": bool(diff.get("ceiling_ok")),
        "current_counts": diff.get("current_counts", {}),
    }


def _seeds() -> dict[str, int]:
    manifest = _read_json(GOLDEN_MANIFEST)
    return {ky: int(manifest[ky]["seed"]) for ky in sorted(manifest)}


def _p0_status() -> dict:
    text = PLAN.read_text(encoding="utf-8")
    rows = re.findall(r"^- \[([ x])\] (P0-\d+)", text, re.M)
    checked = sorted(pid for mark, pid in rows if mark == "x")
    open_items = sorted(pid for mark, pid in rows if mark == " ")
    return {"ok": not open_items and len(checked) >= 6,
            "checked": checked, "open": open_items}


def _devcontainer() -> dict:
    setup = ROOT / ".codex" / "setup.sh"
    dockerfile = ROOT / ".devcontainer" / "Dockerfile"
    evidence: list[str] = []
    ok = True
    if setup.is_file():
        text = setup.read_text(encoding="utf-8")
        evidence.append("setup.sh present")
        if "OMNILSS_R_SOURCE" in text:
            evidence.append("setup.sh bundles R sources best-effort")
        else:
            ok = False
            evidence.append("setup.sh missing R source bundling")
    else:
        ok = False
        evidence.append("setup.sh MISSING")
    if dockerfile.is_file():
        text = dockerfile.read_text(encoding="utf-8")
        if re.search(r"^COPY\s+.*\bR/", text, re.M):
            ok = False
            evidence.append("Dockerfile hard-COPYs gitignored R/ (fresh clone breaks)")
        else:
            evidence.append("Dockerfile does not COPY R/")
        if "install.packages" in text:
            evidence.append("gamlss installed via install.packages (online build)")
    else:
        ok = False
        evidence.append("Dockerfile MISSING")
    for name in ("devcontainer.json", "devcontainer.gpu.json"):
        path = ROOT / ".devcontainer" / name
        if path.is_file() and ".codex/setup.sh" in path.read_text(encoding="utf-8"):
            evidence.append(f"{name} references setup.sh")
        else:
            ok = False
            evidence.append(f"{name} does NOT reference setup.sh")
    return {"ok": ok, "mode": "static + fresh-clone simulation (no docker in CI env)",
            "evidence": evidence}


def _collection_report(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {"ok": None, "reason": "no report supplied", "totals": {}}
    data = _read_json(path)
    return {"ok": bool(data.get("ok")), "totals": data.get("totals", {}),
            "command": "python tools/tiered_collection.py --mode collect --strict-skips --json"}


def _golden_gate(require_r: bool) -> dict:
    cmd = [sys.executable, str(ROOT / "tools" / "golden_manifest_gate.py")]
    if require_r:
        cmd.append("--require-r")
    else:
        cmd.append("--check")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    text = (proc.stdout + proc.stderr).strip().splitlines()
    result_line = next((ln for ln in text if ln.strip().lower().startswith("result:")), "")
    label_line = next((ln for ln in text if ln.strip().lower().startswith("label:")), "")
    return {
        "ok": proc.returncode == 0 and "OK" in result_line.upper(),
        "mode": "--require-r" if require_r else "--check",
        "label": label_line.split(":", 1)[1].strip() if label_line else None,
        "result": result_line.split(":", 1)[1].strip() if result_line else None,
        "command": "python tools/golden_manifest_gate.py " + ("--require-r" if require_r else "--check"),
    }


def _coverage_gate(verify: bool) -> dict:
    if not verify:
        diff = _read_json(COVERAGE_DIFF)
        ok = bool(diff.get("ok") and diff.get("ratchet_ok") and diff.get("ceiling_ok"))
        return {"ok": ok, "mode": "diff flags (pass --verify-coverage to run the gate)",
                "command": "python tools/generate_symbol_coverage.py --check"}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "generate_symbol_coverage.py"), "--check"],
        cwd=ROOT, capture_output=True, text=True,
    )
    return {"ok": proc.returncode == 0, "mode": "--check executed",
            "command": "python tools/generate_symbol_coverage.py --check"}


def build_manifest(
    *,
    collection_report: Path | None = None,
    require_r: bool = False,
    verify_coverage: bool = False,
    generated_at: str | None = None,
) -> dict:
    template = _read_json(TEMPLATE)
    pinned = template["pinned"]
    env = _python_env()
    r_env = _r_env()
    pinned_ok, mismatches = _env_pinned_ok(env, r_env, pinned)
    coverage = _coverage()
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "version": VERSION,
        "generated_at_utc": generated_at or _utc_now(),
        "source": _head(),
        "environment": env,
        "r_environment": r_env,
        "pinned_expected": pinned,
        "environment_pinned_ok": pinned_ok,
        "environment_mismatches": mismatches,
        "seeds": _seeds(),
        "tolerances": template["tolerances"],
        "coverage": {
            "stats": coverage["stats"],
            "provenance": coverage["provenance"],
            "diff_ok": coverage["diff_ok"],
            "ratchet_ok": coverage["ratchet_ok"],
            "ceiling_ok": coverage["ceiling_ok"],
            "current_counts": coverage["current_counts"],
            "command": "python tools/generate_coverage_artifacts.py --date <date>",
        },
        "gates": {
            "p0_clear": _p0_status(),
            "coverage_regenerable": _coverage_gate(verify_coverage),
            "tiered_collection": _collection_report(collection_report),
            "devcontainer": _devcontainer(),
            "golden_manifest_gate": _golden_gate(require_r),
        },
        "notes": (
            "M0 gate bundle per DEVELOPMENT_PLAN 22.4. Reproducibility is anchored by "
            "coverage provenance content hashes (HEAD-independent); source.commit is "
            "informational. Regenerate after any symbol/test change."
        ),
    }


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _render_markdown(m: dict) -> str:
    gates = m["gates"]
    rows = []
    for key in GATE_KEYS:
        g = gates[key]
        rows.append(f"| {key} | {'PASS' if g.get('ok') else 'FAIL'} | {g.get('mode', '')} |")
    seeds = ", ".join(f"{k}={v}" for k, v in m["seeds"].items())
    return "\n".join([
        f"# Milestone {m['milestone']} manifest (v{m['version']})",
        "",
        f"- generated: {m['generated_at_utc']}",
        f"- source commit: {m['source']['commit']} (branch {m['source']['branch']})",
        f"- environment pinned: {m['environment_pinned_ok']}"
        + ("" if m["environment_pinned_ok"] else f" -- {m['environment_mismatches']}"),
        f"- python {m['environment']['python']} / jax {m['environment']['jax']} "
        f"(x64={m['environment']['jax_enable_x64']}) / numpy {m['environment']['numpy']} "
        f"/ scipy {m['environment']['scipy']}",
        f"- R: {m['r_environment'].get('r_version')} / gamlss "
        f"{m['r_environment'].get('gamlss_version')} / gamlss.dist "
        f"{m['r_environment'].get('gamlss_dist_version')}",
        f"- coverage: missing={m['coverage']['stats']['surface_missing']} "
        f"matched={m['coverage']['stats']['surface_matched']} "
        f"(diff ok={m['coverage']['diff_ok']})",
        f"- seeds: {seeds}",
        f"- tolerances: {m['tolerances']}",
        "",
        "| gate | result | mode |",
        "|---|---|---|",
        *rows,
        "",
        m["notes"],
        "",
    ])


def write_manifest(m: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    md = path.with_suffix(".md")
    md.write_text(_render_markdown(m), encoding="utf-8")


def check_manifest(path: Path) -> tuple[bool, list[str]]:
    """Validate that a committed manifest still describes the current tree."""
    problems: list[str] = []
    m = _read_json(path)
    for key in ("schema_version", "milestone", "version", "environment", "gates",
                "seeds", "tolerances", "coverage", "pinned_expected"):
        if key not in m:
            problems.append(f"missing key: {key}")
    if problems:
        return False, problems
    if m["schema_version"] != SCHEMA_VERSION:
        problems.append(f"schema_version {m['schema_version']} != {SCHEMA_VERSION}")
    if not m.get("environment_pinned_ok"):
        problems.append(f"environment not pinned: {m.get('environment_mismatches')}")
    for key in GATE_KEYS:
        gate = m["gates"].get(key)
        if gate is None:
            problems.append(f"gate missing: {key}")
        elif gate.get("ok") is not True:
            problems.append(f"gate not passing: {key} -> {gate.get('ok')}")
    current = _coverage()
    if m["coverage"]["stats"] != current["stats"]:
        problems.append(f"coverage stats drift: {m['coverage']['stats']} != {current['stats']}")
    if m["coverage"]["provenance"] != current["provenance"]:
        problems.append("coverage provenance drift (regenerate the milestone manifest)")
    if m["seeds"] != _seeds():
        problems.append("golden seed set drift (regenerate golden manifest)")
    template = _read_json(TEMPLATE)
    if m["tolerances"] != template["tolerances"]:
        problems.append("tolerance drift vs manifest_template.json")
    if m["pinned_expected"] != template["pinned"]:
        problems.append("pinned environment drift vs manifest_template.json")
    return (not problems), problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None,
                        help="write the manifest JSON (and .md) here")
    parser.add_argument("--check", action="store_true",
                        help="validate an existing manifest against the current tree")
    parser.add_argument("--path", type=Path, default=DEFAULT_OUT,
                        help="manifest path for --check (default: M0 artifact)")
    parser.add_argument("--collection-report", type=Path, default=None,
                        help="JSON from tiered_collection --json")
    parser.add_argument("--require-r", action="store_true",
                        help="probe the golden gate with --require-r (fail-closed)")
    parser.add_argument("--verify-coverage", action="store_true",
                        help="run generate_symbol_coverage.py --check instead of trusting diff flags")
    parser.add_argument("--json", action="store_true", help="print the manifest to stdout")
    args = parser.parse_args(argv)

    if args.check:
        ok, problems = check_manifest(args.path)
        print(f"milestone manifest check: {args.path}")
        if ok:
            print("result: OK")
            return 0
        for problem in problems:
            print(f"  - {problem}")
        print("result: FAIL")
        return 1

    manifest = build_manifest(
        collection_report=args.collection_report,
        require_r=args.require_r,
        verify_coverage=args.verify_coverage,
    )
    # With --json, stdout must carry ONLY the JSON; human lines go to stderr.
    log = (lambda *a: print(*a, file=sys.stderr)) if args.json else print
    if args.write is not None:
        write_manifest(manifest, args.write)
        log(f"wrote {args.write}")
        log(f"wrote {args.write.with_suffix('.md')}")
    if args.json or args.write is None:
        print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    for key in GATE_KEYS:
        gate = manifest["gates"][key]
        log(f"  {key}: {'PASS' if gate.get('ok') else 'FAIL'} ({gate.get('mode', '')})")
    log(f"environment pinned: {manifest['environment_pinned_ok']}")
    return 0 if all(manifest["gates"][k].get("ok") for k in GATE_KEYS) and manifest["environment_pinned_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
