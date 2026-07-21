"""Generate the authoritative Release 1 R capability evidence matrix.

Only vendored ``R/gamlss`` and ``R/gamlss.dist`` namespaces are parsed.  The
JSON matrix is the source for the Markdown report and all five metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "validation" / "r-gamlss-release1"
CORE_PACKAGES = ("gamlss", "gamlss.dist")
EXCLUDED_PACKAGES: Mapping[str, str] = {
    "gamlss.add": "Release 1 excludes add-on terms from statistical migration metrics.",
    "gamlss.cens": "Release 1 excludes censored-response extensions.",
    "gamlss.tr": "Release 1 excludes truncated-distribution extensions.",
    "gamlss2": "Release 1 excludes the next-generation interface package.",
    "gamlss.mx": "Release 1 excludes mixture-model extensions.",
    "gamlss.spatial": "Release 1 excludes spatial-model extensions.",
    "gamlss.foreach": "Release 1 excludes parallel foreach integration.",
    "gamlss.prepdata": "Release 1 excludes data-preparation helpers.",
    "gamlss.ggplots": "Release 1 excludes plotting integrations.",
    "gamlss.data": "Release 1 excludes bundled data from migration completion metrics.",
    "WeatherGermany": "Release 1 excludes the external example data package.",
}
CAPABILITY_STATUSES = frozenset({
    "missing", "placeholder", "compat_alias", "experimental", "implemented",
    "unsupported_with_rationale",
})
_STATUS_BY_IMPLEMENTATION = {
    "missing": "missing", "placeholder": "placeholder",
    "compat_alias": "compat_alias", "experimental": "experimental",
    "implemented": "implemented", "unsupported_with_rationale": "unsupported_with_rationale",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _description_version(path: Path) -> str:
    match = re.search(r"^Version:\s*(.+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise ValueError(f"missing Version field in {path}")
    return match.group(1).strip()


def parse_namespace_exports(path: Path) -> tuple[str, ...]:
    """Parse unique exports from an R NAMESPACE while ignoring comments."""
    uncommented = "\n".join(line.split("#", 1)[0] for line in path.read_text(encoding="utf-8").splitlines())
    exports: set[str] = set()
    for match in re.finditer(r"\bexport\s*\((.*?)\)", uncommented, re.DOTALL):
        exports.update(token.strip() for token in match.group(1).split(",") if token.strip())
    return tuple(sorted(exports))


def source_evidence(repo_root: Path = REPO_ROOT) -> dict[str, dict[str, str]]:
    """Return fixed-source versions and hashes for the two permitted packages."""
    evidence: dict[str, dict[str, str]] = {}
    for package in CORE_PACKAGES:
        source = repo_root / "R" / package
        namespace = source / "NAMESPACE"
        description = source / "DESCRIPTION"
        evidence[package] = {
            "source_path": str(source.relative_to(repo_root)).replace("\\", "/"),
            "version": _description_version(description),
            "namespace_sha256": _sha256(namespace),
            "description_sha256": _sha256(description),
        }
    return evidence


def _family_metadata() -> dict[str, dict[str, str]]:
    from omnilss.distribution_registry import _BUILTIN_FAMILY_FACTORIES
    from omnilss.family_capabilities import implementation_status_for_family

    metadata: dict[str, dict[str, str]] = {}
    for family, (module, factory) in _BUILTIN_FAMILY_FACTORIES.items():
        implementation_status = implementation_status_for_family(family).value
        metadata[family] = {
            "python_entry": f"{module}.{factory}",
            "kernel_owner": module,
            "implementation_origin": f"Python factory {module}.{factory}",
            "status": _STATUS_BY_IMPLEMENTATION[implementation_status],
        }
    return metadata


def build_matrix(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Build the complete, explicit Release 1 capability evidence matrix."""
    evidence = source_evidence(repo_root)
    family_metadata = _family_metadata()
    entries: list[dict[str, Any]] = []
    for package in CORE_PACKAGES:
        namespace = repo_root / "R" / package / "NAMESPACE"
        for symbol in parse_namespace_exports(namespace):
            public = family_metadata.get(symbol.upper())
            entry: dict[str, Any] = {
                "r_package": package,
                "r_source_version": evidence[package]["version"],
                "r_symbol": symbol,
                "python_entry": None,
                "status": "missing",
                "implementation_origin": "No Release 1 Python implementation metadata.",
                "kernel_owner": None,
                "evidence_tier": "E0",
                "evidence_references": [f"R/{package}/NAMESPACE"],
                "exclusion_reason": None,
            }
            if public is not None:
                entry.update(public)
                entry["evidence_tier"] = "E1"
                entry["evidence_references"] = [
                    f"R/{package}/NAMESPACE",
                    "omnilss/src/omnilss/distribution_registry.py",
                    "omnilss/src/omnilss/family_capabilities.py",
                ]
            entries.append(entry)
    entries.sort(key=lambda item: (item["r_package"], item["r_symbol"]))
    matrix = {
        "schema_version": 1,
        "release": "Release 1",
        "included_r_packages": list(CORE_PACKAGES),
        "excluded_r_packages": [
            {"package": package, "reason": reason}
            for package, reason in EXCLUDED_PACKAGES.items()
        ],
        "allowed_statuses": sorted(CAPABILITY_STATUSES),
        "source_evidence": evidence,
        "entries": entries,
    }
    matrix["metrics"] = compute_metrics(matrix)
    validate_matrix(matrix)
    return matrix


def compute_metrics(matrix: Mapping[str, Any]) -> dict[str, int]:
    entries = list(matrix["entries"])
    public = [entry for entry in entries if entry["python_entry"]]
    return {
        "exported_symbol_coverage": sum(entry["status"] != "missing" for entry in entries),
        "callable_python_entry_coverage": len(public),
        "dedicated_kernel_coverage": sum(entry["status"] == "implemented" for entry in public),
        "e3_family_count": sum(entry["evidence_tier"] == "E3" for entry in public),
        "e4_feature_count": sum(entry["evidence_tier"] == "E4" for entry in entries),
    }


def validate_matrix(matrix: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when the matrix violates Release 1 traceability."""
    if tuple(matrix.get("included_r_packages", ())) != CORE_PACKAGES:
        raise ValueError("Release 1 may include only gamlss and gamlss.dist")
    excluded = matrix.get("excluded_r_packages", [])
    if len(excluded) != 11 or {item["package"] for item in excluded} != set(EXCLUDED_PACKAGES):
        raise ValueError("Release 1 must serialize exactly the eleven excluded packages")
    entries = matrix.get("entries", [])
    keys = [(entry["r_package"], entry["r_symbol"]) for entry in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("each R export must have exactly one matrix entry")
    for entry in entries:
        if entry["status"] not in CAPABILITY_STATUSES:
            raise ValueError(f"invalid capability status for {entry['r_symbol']}")
        if entry["python_entry"] is not None and not all((
            entry["kernel_owner"], entry["implementation_origin"], entry["evidence_references"],
        )):
            raise ValueError(f"public family {entry['r_symbol']} lacks traceability metadata")


def markdown_report(matrix: Mapping[str, Any]) -> str:
    metrics = matrix["metrics"]
    lines = [
        "# Release 1 Capability Evidence Matrix",
        "",
        "This report is derived exclusively from `r-export-coverage.json`.",
        "",
        "## Scope",
        "",
        "Included fixed R sources: " + ", ".join(matrix["included_r_packages"]) + ".",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {value} |" for name, value in metrics.items())
    lines.extend(["", "## Excluded packages", "", "| Package | Reason |", "| --- | --- |"])
    lines.extend(f"| {item['package']} | {item['reason']} |" for item in matrix["excluded_r_packages"])
    lines.extend(["", "## Fixed R source evidence", "", "| Package | Version | Namespace SHA-256 |", "| --- | --- | --- |"])
    lines.extend(
        f"| {package} | {item['version']} | `{item['namespace_sha256']}` |"
        for package, item in matrix["source_evidence"].items()
    )
    return "\n".join(lines) + "\n"


def write_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR, repo_root: Path = REPO_ROOT) -> tuple[Path, Path, Path]:
    matrix = build_matrix(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "r-export-coverage.json"
    report_path = output_dir / "r-export-coverage.md"
    evidence_path = output_dir / "r-source-versions.json"
    matrix_path.write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.write_text(markdown_report(matrix), encoding="utf-8")
    evidence_path.write_text(json.dumps(matrix["source_evidence"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return matrix_path, report_path, evidence_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    for path in write_artifacts(args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
