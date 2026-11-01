# SPDX-License-Identifier: GPL-3.0-or-later
"""Pin the live-R response-support table as a golden artifact.

R's `gamlss()` refuses to fit data outside `family$y.valid(y)`
(`stop("response variable out of range")`, gamlss/R/gamlss-5.R).  OmniLSS
mirrors that guard in `omnilss/src/omnilss/response_domain.py`, and its table
may only contain predicates R itself declares.

This tool runs `tools/r_y_valid_dump.R` against the installed
`gamlss.dist`, matches every OmniLSS registry family to its R counterpart
(exact name, then case-insensitive, then the four kernel aliases), and writes:

    {
      "meta":       {R / gamlss / gamlss.dist versions, generation date},
      "families":   {"<R family>": "<deparse(body(y.valid))>"},
      "supported":  {"<OmniLSS family.name>": "<R family>" | null},
      "unmatched":  ["<OmniLSS family.name>", ...]
    }

`supported` is the bridge between our dispatch name (what
`gamlss()` passes to `validate_response`) and R's name, so the test never has
to re-derive it.

Usage::

    python tools/generate_response_domain_golden.py
    python tools/generate_response_domain_golden.py --dump tmp/yvalid.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = (
    REPO_ROOT
    / "omnilss"
    / "tests"
    / "fixtures"
    / "r_golden"
    / "response_domains_y_valid.json"
)

#: Registry names whose kernels are an R family under another name.  Mirrors
#: `tools/rs_family_parity_scan.py::_R_FAMILY_ALIASES` (the scanner is the
#: canonical table; it is duplicated here because importing it pulls in JAX and
#: builds all 146 families, ~3 minutes).
KERNEL_ALIASES = {
    "IGA": "IG",
    "GENERALISEDPOISSON": "GPO",
    "DPO1": "DPO",
    "DELAPORT": "DEL",
}


def _dump_live_r(dest: Path) -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise SystemExit("Rscript not found on PATH; live R is required")
    subprocess.run(
        [rscript, str(REPO_ROOT / "tools" / "r_y_valid_dump.R"), str(dest)],
        check=True,
        cwd=REPO_ROOT,
    )


def _our_family_names() -> dict[str, str]:
    """registry key -> family.name (the string `gamlss()` dispatch uses)."""
    sys.path.insert(0, str(REPO_ROOT / "omnilss" / "src"))
    from omnilss.distribution_registry import list_families, resolve

    out: dict[str, str] = {}
    for key in list_families():
        family = resolve(key)
        # A family is only reachable through fitting if it resolves; failures
        # here mean the registry is broken, which is not this tool's problem.
        out[key] = getattr(family, "name", key)
    return out


def _match(name: str, r_names: dict[str, str], lower_index: dict[str, str]) -> str | None:
    if name in r_names:
        return name
    alias = KERNEL_ALIASES.get(name.upper())
    if alias is not None and alias in r_names:
        return alias
    return lower_index.get(name.lower())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help=f"artifact path (default {DEFAULT_OUT})"
    )
    parser.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="reuse an existing r_y_valid_dump.R output instead of invoking R",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the artifact on disk differs from a freshly generated one",
    )
    args = parser.parse_args(argv)

    if args.dump is not None:
        dump_path = args.dump
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory()
        dump_path = Path(cleanup.name) / "y_valid.json"
        _dump_live_r(dump_path)

    try:
        dump = json.loads(dump_path.read_text(encoding="utf-8"))
        r_families: dict[str, str] = dump["families"]
        meta = dict(dump["meta"])
        meta["generator"] = "tools/generate_response_domain_golden.py"
        meta["r_dump_generator"] = "tools/r_y_valid_dump.R"

        lower_index: dict[str, list[str]] = {}
        for r_name in r_families:
            lower_index.setdefault(r_name.lower(), []).append(r_name)

        supported: dict[str, str | None] = {}
        ambiguous: list[str] = []
        for key, family_name in sorted(_our_family_names().items()):
            candidates = lower_index.get(family_name.lower(), [])
            if len(candidates) > 1 and family_name not in r_families:
                # e.g. PARETO1o vs PARETO1 would collide if R ever exported
                # two families differing only in case -- refuse to guess.
                ambiguous.append(f"{family_name}: {candidates}")
                supported[family_name] = None
                continue
            supported[family_name] = _match(family_name, r_families, {
                k: v[0] for k, v in lower_index.items()
            })
        if ambiguous:
            raise SystemExit("ambiguous case-insensitive matches: " + "; ".join(ambiguous))

        artifact = {
            "meta": meta,
            "families": dict(sorted(r_families.items())),
            "supported": dict(sorted(supported.items())),
        }
        text = json.dumps(artifact, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    finally:
        if cleanup is not None:
            cleanup.cleanup()

    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != text:
            print(f"{args.out} is stale; re-run without --check", file=sys.stderr)
            return 1
        print(f"{args.out} is up to date")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    matched = sum(1 for v in supported.values() if v is not None)
    print(f"wrote {args.out}")
    print(f"  R families: {len(r_families)}  supported: {matched}/{len(supported)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
