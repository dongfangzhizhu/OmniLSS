# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate the D0.1 real per-symbol R <-> Python coverage matrix.

Unlike ``generate_r_export_coverage.py`` (registry inventory only), this tool
parses the checked-in R package sources and matches every exported R symbol
against the OmniLSS Python export surface, the family registry and the test /
golden-fixture evidence, producing a four-layer status per symbol:

``surface``   symbol reachable from Python (public export, module definition or
              distribution-family registry entry).
``kernel``    implementation depth (``implemented`` / ``stub`` / registry
              implementation status / ``missing``).
``parity``    R-consistency evidence (``r_consistency_test`` / ``tested`` /
              family ``r_consistency`` capability tier / ``unverified``).
``admission`` production admission (``exported_api`` / ``production_safe`` /
              ``blocked`` / ``internal`` / ``not_available``).

The JSON + Markdown artifacts are fully regenerable from source: no timestamps
and no repository HEAD hash are embedded. Provenance is content based because
``R/`` is a vendored (git-ignored) upstream source tree:

* ``provenance.*_sha256``           hashes of the parsed R/py/test inputs;
* ``stats.per_package[].package_version``  upstream R package version
  (``R/<pkg>/DESCRIPTION``) plus the NAMESPACE hash;
* ``symbols[].r_source_sha256``     hash of the R file that defines the symbol;
* ``symbols[].python_source_commit`` last commit touching the primary OmniLSS
  module that provides the symbol (tracked source, so a real commit exists).

``--check`` verifies the checked-in artifacts are up to date without rewriting
them (used by CI).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
R_ROOT = REPO_ROOT / "R"
OMNILSS_ROOT = REPO_ROOT / "omnilss" / "src" / "omnilss"
TESTS_ROOT = REPO_ROOT / "omnilss" / "tests"
GOLDEN_MANIFEST = TESTS_ROOT / "fixtures" / "r_golden" / "regenerated" / "manifest.json"
OUT_JSON = REPO_ROOT / "docs" / "development" / "r-symbol-coverage.json"
OUT_MD = REPO_ROOT / "docs" / "development" / "r-symbol-coverage.md"

#: P0-6: scope declarations come from a single source of truth so every
#: coverage report stays in sync.  Dual import: this module is loaded both as a
#: namespace package (``tools.generate_symbol_coverage`` from pytest / the diff
#: report) and as a direct script (``python tools/generate_symbol_coverage.py``).
try:
    from tools.coverage_scope import (  # noqa: E402
        OUT_OF_SCOPE_PACKAGES,
        scope_line,
        scope_payload,
    )
except ModuleNotFoundError:  # pragma: no cover - direct-script fallback
    from coverage_scope import (  # noqa: E402
        OUT_OF_SCOPE_PACKAGES,
        scope_line,
        scope_payload,
    )

SCHEMA_VERSION = 1

_R_DEF_RE = re.compile(r"(?m)^\s*([A-Za-z.][A-Za-z0-9._]*)\s*(?:<<-|<-|=)\s*function\b")
_EXPORT_CALL_RE = re.compile(r"\b(export[A-Za-z]*)\s*\(([^)]*)\)")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*")
_CLEAN_NAME_STRIP = "\"'"
_DPQR_PREFIXES = ("d", "p", "q", "r")

#: Human readable legend written into both artifacts.
LAYER_LEGEND = {
    "surface": {
        "public_api": "top-level `omnilss.__all__` export",
        "module_export": "exported from a submodule `__all__`",
        "module_def": "defined in an omnilss module, not exported",
        "registry": "family present in the OmniLSS capability registry (kernels in the kernel layer)",
        "missing": "no Python surface found",
    },
    "kernel": {
        "implemented": "Python implementation present",
        "stub": "Python definition raises NotImplementedError",
        "missing": "no Python implementation",
        "registry impl status": "family-linked symbols use the registry implementation_status",
    },
    "parity": {
        "r_consistency_test": "a `test_r_consistency*` module references the symbol",
        "capability:<tier>": "family `r_consistency` capability tier (validated / experimental / ...)",
        "golden:<tier>": "family has a fixed-R golden fixture plus the capability tier above",
        "tested": "referenced by non-R-consistency tests only",
        "unverified": "no test evidence found",
    },
    "admission": {
        "exported_api": "part of the supported public API",
        "internal": "module-internal helper, no admission claim",
        "not_available": "no Python symbol, production routing cannot select it",
        "production_safe": "family admitted for production routing",
        "blocked": "family registered but not production safe",
    },
}


def _norm(symbol: str) -> str:
    """Normalise an R symbol for cross-language matching (``a.b`` -> ``a_b``)."""
    return symbol.replace(".", "_").replace("-", "_")


def _rel(path: Path) -> str:
    """Repo-relative POSIX path, or the absolute path when outside the repo."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:  # pragma: no cover - only synthetic paths outside the repo
        return path.as_posix()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _hash_files(paths: list[Path]) -> str:
    """Stable sha256 over the sorted (relative path, normalised content) pairs.

    Content is read in text mode so the digest is independent of the working
    tree line endings (``.gitattributes`` pins ``eol=lf``, but local checkouts
    may still differ).
    """
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(_rel(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_read_text(path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: Path, cache: dict[str, str]) -> str:
    """Short content hash of one file (line-ending normalised), cached."""
    key = _rel(path)
    if key not in cache:
        cache[key] = hashlib.sha256(_read_text(path).encode("utf-8")).hexdigest()[:16]
    return cache[key]


def r_package_version(package_dir: Path) -> str:
    """Return the upstream ``Version:`` of a vendored R package DESCRIPTION."""
    description = package_dir / "DESCRIPTION"
    if not description.exists():
        return "unknown"
    for line in _read_text(description).splitlines():
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def _git_commit_for(path: Path, cache: dict[str, str]) -> str:
    """Return the abbreviated commit that last touched ``path`` (tracked files)."""
    key = _rel(path)
    if key in cache:
        return cache[key]
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", key],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        commit = result.stdout.strip().splitlines()[0] if result.stdout.strip() else "unknown"
    except OSError:  # pragma: no cover - git not installed
        commit = "unknown"
    cache[key] = commit
    return commit


def r_package_dirs() -> dict[str, Path]:
    """Return ``{package: package_dir}`` for every checked-in R package."""
    if not R_ROOT.is_dir():
        return {}
    return {
        path.name: path
        for path in sorted(R_ROOT.iterdir())
        if path.is_dir() and any(path.glob("NAMESPACE"))
    }


def _clean_r_name(raw: str) -> str:
    """Normalise one entry of an ``export(...)`` call, dropping patterns."""
    name = raw.strip().strip(_CLEAN_NAME_STRIP).strip()
    if not name or name.startswith("^") or "\\" in name:
        return ""
    return name


def parse_namespace(path: Path) -> dict:
    """Parse the ``export*()`` directives of an R ``NAMESPACE`` file."""
    stripped = "\n".join(line.split("#", 1)[0] for line in _read_text(path).splitlines())
    exports: set[str] = set()
    auxiliary: set[str] = set()
    for match in _EXPORT_CALL_RE.finditer(stripped):
        directive, body = match.group(1), match.group(2)
        names = {_clean_r_name(part) for part in body.split(",")}
        names.discard("")
        if directive == "export":
            exports |= names
        else:
            auxiliary |= names
    return {
        "exports": sorted(exports),
        "auxiliary_exports": sorted(auxiliary),
        "s3method_count": len(re.findall(r"(?m)^\s*S3method\s*\(", stripped)),
    }


def index_r_definitions(package_dir: Path) -> dict[str, list[str]]:
    """Map every R symbol defined as ``name <- function(...)`` to its sources."""
    definitions: dict[str, list[str]] = {}
    sources = sorted({path.resolve() for path in package_dir.rglob("*.[Rr]")})
    for path in sources:
        rel = _rel(path)
        for number, line in enumerate(_read_text(path).splitlines(), start=1):
            match = _R_DEF_RE.match(line)
            if match:
                definitions.setdefault(match.group(1), []).append(f"{rel}:{number}")
    return definitions


def _is_stub(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when the function body only raises ``NotImplementedError``."""
    body = [
        statement
        for statement in node.body
        if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant))
    ]
    if len(body) != 1 or not isinstance(body[0], ast.Raise):
        return False
    exc = body[0].exc
    if isinstance(exc, ast.Call):
        exc = exc.func
    return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"


def index_python_surface() -> dict:
    """Index ``__all__`` exports, public definitions and stubs of ``omnilss``."""
    exported: dict[str, set[str]] = {}
    defined: dict[str, set[str]] = {}
    stubs: set[str] = set()
    for path in sorted(OMNILSS_ROOT.rglob("*.py")):
        rel = path.relative_to(OMNILSS_ROOT).as_posix()
        try:
            tree = ast.parse(_read_text(path))
        except SyntaxError:  # pragma: no cover - invalid generated module
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    defined.setdefault(node.name, set()).add(rel)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_stub(node):
                    stubs.add(node.name)
            elif isinstance(node, ast.Assign):
                if not any(getattr(target, "id", None) == "__all__" for target in node.targets):
                    continue
                if not isinstance(node.value, (ast.List, ast.Tuple)):
                    continue
                for element in node.value.elts:
                    if isinstance(element, ast.Constant) and isinstance(element.value, str):
                        exported.setdefault(element.value, set()).add(rel)
    return {"exported": exported, "defined": defined, "stubs": stubs}


def index_test_evidence() -> dict[str, list[str]]:
    """Invert ``omnilss/tests`` so every identifier maps to referencing files."""
    index: dict[str, set[str]] = {}
    for path in sorted(TESTS_ROOT.rglob("test_*.py")):
        rel = _rel(path)
        for token in set(_TOKEN_RE.findall(_read_text(path))):
            index.setdefault(token, set()).add(rel)
            if "." in token:
                index.setdefault(_norm(token), set()).add(rel)
    return {token: sorted(files) for token, files in index.items()}


def load_family_facts(emit=print) -> dict:
    """Collect registry status, family-class presence and golden coverage.

    ``resolve_family`` is deliberately NOT used here: instantiating family
    objects costs up to 10s per family (JAX/backend initialisation), which would
    make artifact regeneration unacceptable. The family class presence check and
    the registry ``implementation_status`` (which already distinguishes
    ``placeholder`` stub kernels from real ones) are cheap and are the declared
    evidence for the ``kernel`` layer.
    """
    src = REPO_ROOT / "omnilss" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from omnilss.family_capabilities import family_capability_names, get_family_capability

    facts: dict[str, dict] = {}
    for name in family_capability_names():
        cap = get_family_capability(name)
        try:
            parity = cap.status("r_consistency").name.lower()
        except Exception:  # pragma: no cover - feature always declared today
            parity = "unknown"
        facts[name] = {
            "implementation_status": cap.implementation_status.value,
            "r_consistency": parity,
            "production_safe": bool(cap.is_production_safe),
        }
    golden: dict[str, list[str]] = {}
    if GOLDEN_MANIFEST.exists():
        try:
            manifest = json.loads(_read_text(GOLDEN_MANIFEST))
        except json.JSONDecodeError as error:
            emit(f"warning: golden manifest unreadable: {error}")
            manifest = {}
        if isinstance(manifest, dict):
            for case, entry in manifest.items():
                if isinstance(entry, dict) and entry.get("family"):
                    golden.setdefault(entry["family"], []).append(case)
    return {
        "capabilities": facts,
        "golden": {name: sorted(cases) for name, cases in golden.items()},
    }


def _family_for(symbol: str, capabilities: dict) -> tuple[str | None, str | None]:
    """Return ``(family, kind)`` when the R symbol belongs to a registered family."""
    if symbol in capabilities:
        return symbol, "family"
    if len(symbol) > 1 and symbol[0] in _DPQR_PREFIXES and symbol[1:].isupper() and symbol[1:] in capabilities:
        return symbol[1:], "dpqr"
    return None, None


def _python_source_commit(modules: list[str], git_cache: dict[str, str]) -> str:
    """Last commit touching the primary OmniLSS module that provides the symbol."""
    for rel in modules:
        source = OMNILSS_ROOT / rel
        if source.exists():
            return _git_commit_for(source, git_cache)
    return "unknown"


def _r_source_sha256(definitions: list[str], hash_cache: dict[str, str]) -> str:
    """Content hash of the R file that defines the symbol (vendored source)."""
    if not definitions:
        return ""
    source = REPO_ROOT / definitions[0].split(":")[0]
    return _file_sha256(source, hash_cache) if source.exists() else ""


def _classify_layers(
    *,
    modules: list[str],
    exported_modules: set[str],
    python_symbol: str | None,
    family: str | None,
    surface: dict,
    tests_files: list[str],
    facts: dict,
) -> dict[str, str]:
    """Assign the four-layer status of one R symbol."""
    is_public = "__init__.py" in exported_modules
    r_consistency_hit = any("/test_r_consistency" in item for item in tests_files)
    if family:
        info = facts["capabilities"][family]
        if is_public:
            surface_status = "public_api"
        elif exported_modules:
            surface_status = "module_export"
        elif modules:
            surface_status = "module_def"
        else:
            surface_status = "registry"
        if r_consistency_hit:
            parity_status = "r_consistency_test"
        elif facts["golden"].get(family):
            parity_status = f"golden:{info['r_consistency']}"
        else:
            parity_status = f"capability:{info['r_consistency']}"
        return {
            "surface": surface_status,
            "kernel": info["implementation_status"],
            "parity": parity_status,
            "admission": "production_safe" if info["production_safe"] else "blocked",
        }
    if not modules:
        return {
            "surface": "missing",
            "kernel": "missing",
            "parity": "unverified",
            "admission": "not_available",
        }
    surface_status = "public_api" if is_public else ("module_export" if exported_modules else "module_def")
    kernel_status = "stub" if python_symbol in surface["stubs"] else "implemented"
    if r_consistency_hit:
        parity_status = "r_consistency_test"
    elif tests_files:
        parity_status = "tested"
    else:
        parity_status = "unverified"
    return {
        "surface": surface_status,
        "kernel": kernel_status,
        "parity": parity_status,
        "admission": "exported_api" if is_public else "internal",
    }


def _build_row(
    *,
    symbol: str,
    package: str,
    package_version: str,
    definitions: dict[str, list[str]],
    surface: dict,
    tests: dict,
    facts: dict,
    git_cache: dict[str, str],
    hash_cache: dict[str, str],
    in_scope: bool,
) -> dict:
    """Build one per-symbol matrix row."""
    family, family_kind = _family_for(symbol, facts["capabilities"])
    python_symbol = None
    modules: list[str] = []
    exported_modules: set[str] = set()
    for candidate in (symbol, _norm(symbol)):
        exported = surface["exported"].get(candidate, set())
        defined = surface["defined"].get(candidate, set())
        if exported or defined:
            python_symbol = candidate
            exported_modules = set(exported)
            modules = sorted(set(exported) | set(defined))
            break
    tests_files = sorted(set(tests.get(symbol, [])) | set(tests.get(_norm(symbol), [])))
    r_definitions = definitions.get(symbol, [])
    layers = _classify_layers(
        modules=modules,
        exported_modules=exported_modules,
        python_symbol=python_symbol,
        family=family,
        surface=surface,
        tests_files=tests_files,
        facts=facts,
    )
    return {
        "r_symbol": symbol,
        "r_package": package,
        "r_package_version": package_version,
        "r_kind": "function" if r_definitions else "object",
        "r_family": family,
        "r_family_kind": family_kind,
        "r_source_sha256": _r_source_sha256(r_definitions, hash_cache),
        "python_symbol": python_symbol,
        "python_modules": modules,
        "python_exported_from": sorted(exported_modules),
        "python_source_commit": _python_source_commit(modules, git_cache),
        "layers": layers,
        "evidence": {
            "r_definitions": r_definitions[:3],
            "tests": tests_files[:5],
            "test_count": len(tests_files),
            "golden_cases": facts["golden"].get(family, []) if family else [],
        },
        "in_scope": in_scope,
    }


def _counter(values) -> dict[str, int]:
    """Count values into a sorted mapping (deterministic artifact output)."""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_coverage(emit=print) -> dict:
    """Build the complete per-symbol matrix without writing any file."""
    surface = index_python_surface()
    tests = index_test_evidence()
    facts = load_family_facts(emit)
    git_cache: dict[str, str] = {}
    hash_cache: dict[str, str] = {}
    packages = r_package_dirs()
    rows: list[dict] = []
    package_stats: dict[str, dict] = {}
    namespace_paths: list[Path] = []
    for package, directory in packages.items():
        namespace_path = directory / "NAMESPACE"
        namespace_paths.append(namespace_path)
        namespace = parse_namespace(namespace_path)
        definitions = index_r_definitions(directory)
        package_version = r_package_version(directory)
        in_scope = package not in OUT_OF_SCOPE_PACKAGES
        package_rows = [
            _build_row(
                symbol=symbol,
                package=package,
                package_version=package_version,
                definitions=definitions,
                surface=surface,
                tests=tests,
                facts=facts,
                git_cache=git_cache,
                hash_cache=hash_cache,
                in_scope=in_scope,
            )
            for symbol in namespace["exports"]
        ]
        rows.extend(package_rows)
        package_stats[package] = {
            "in_scope": in_scope,
            "package_version": package_version,
            "namespace_sha256": _hash_files([namespace_path])[:16],
            "exported": len(namespace["exports"]),
            "defined_functions": sum(1 for row in package_rows if row["r_kind"] == "function"),
            "surface_matched": sum(1 for row in package_rows if row["layers"]["surface"] != "missing"),
            "auxiliary_exports": len(namespace["auxiliary_exports"]),
            "s3method_count": namespace["s3method_count"],
        }
    rows.sort(key=lambda row: (row["r_package"], row["r_symbol"]))
    r_names = {row["r_symbol"] for row in rows} | {_norm(row["r_symbol"]) for row in rows}
    python_only = sorted(
        name
        for name, modules in surface["exported"].items()
        if "__init__.py" in modules and name not in r_names and _norm(name) not in r_names
    )
    stats = {
        "r_packages": len(packages),
        "r_exports_total": len(rows),
        "in_scope_exports": sum(1 for row in rows if row["in_scope"]),
        "surface_matched": sum(1 for row in rows if row["layers"]["surface"] != "missing"),
        "surface_missing": sum(1 for row in rows if row["layers"]["surface"] == "missing"),
        "family_linked": sum(1 for row in rows if row["r_family"]),
        "dpqr_linked": sum(1 for row in rows if row["r_family_kind"] == "dpqr"),
        "python_only_public_exports": len(python_only),
        "per_layer": {
            layer: _counter(row["layers"][layer] for row in rows)
            for layer in ("surface", "kernel", "parity", "admission")
        },
        "per_package": package_stats,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "tools/generate_symbol_coverage.py",
        "scope": {
            "r_packages": sorted(packages),
            **scope_payload(),
        },
        "provenance": {
            "r_namespace_sha256": _hash_files(namespace_paths),
            "python_surface_sha256": _hash_files(sorted(OMNILSS_ROOT.rglob("*.py"))),
            "test_suite_sha256": _hash_files(sorted(TESTS_ROOT.rglob("test_*.py"))),
            "golden_manifest_sha256": (
                _hash_files([GOLDEN_MANIFEST]) if GOLDEN_MANIFEST.exists() else "absent"
            ),
        },
        "layer_legend": LAYER_LEGEND,
        "stats": stats,
        "symbols": rows,
        "python_only_public_exports": python_only,
    }


def render_markdown(coverage: dict) -> str:
    """Render the Markdown companion of the JSON matrix."""
    stats = coverage["stats"]
    out_of_scope = ", ".join(coverage["scope"]["out_of_scope_packages"]) or "none"
    lines = [
        "# R -> Python per-symbol coverage matrix (D0.1)",
        "",
        scope_line(),
        "",
        "Generated by `tools/generate_symbol_coverage.py`. The artifact is regenerable from"
        " source: it embeds no timestamp and no repository HEAD hash. Each row records the"
        " defining R file hash (vendored source) and the last commit of the primary OmniLSS"
        " module, so drift is detectable per symbol.",
        "",
        "Layer values are decoded in the `layer_legend` object of `r-symbol-coverage.json`;"
        " the table below uses the short form `surface/kernel/parity/admission`.",
        "",
        "## Summary",
        "",
        f"- R packages parsed: {stats['r_packages']} (out of scope: {out_of_scope})",
        f"- R exported symbols: {stats['r_exports_total']} (in scope: {stats['in_scope_exports']})",
        f"- Python surface matched: {stats['surface_matched']} / missing: {stats['surface_missing']}",
        f"- Family-linked symbols: {stats['family_linked']} (dpqr kernels: {stats['dpqr_linked']})",
        f"- Python-only public exports: {stats['python_only_public_exports']}",
        "",
    ]
    for layer in ("surface", "kernel", "parity", "admission"):
        rendered = ", ".join(f"`{key}` {value}" for key, value in stats["per_layer"][layer].items())
        lines.append(f"- **{layer}**: {rendered}")
    lines += [
        "",
        "## Per-package coverage",
        "",
        "R packages are vendored (git-ignored) sources, so provenance is the upstream"
        " version plus the NAMESPACE content hash.",
        "",
        "| R package | version | NAMESPACE sha256 | in scope | exports | defined functions | surface matched |",
        "|---|---|---|---|---|---|---|",
    ]
    for package, entry in stats["per_package"].items():
        lines.append(
            f"| {package} | {entry['package_version']} | `{entry['namespace_sha256']}` | "
            f"{'yes' if entry['in_scope'] else 'no'} | {entry['exported']} | "
            f"{entry['defined_functions']} | {entry['surface_matched']} |"
        )
    missing = [row for row in coverage["symbols"] if row["layers"]["surface"] == "missing"]
    lines += ["", "## Symbols without any Python surface", ""]
    if not missing:
        lines.append("_None._")
    else:
        lines += ["| R symbol | package | R kind | R definitions |", "|---|---|---|---|"]
        for row in missing[:400]:
            definitions = ", ".join(row["evidence"]["r_definitions"]) or "-"
            lines.append(f"| {row['r_symbol']} | {row['r_package']} | {row['r_kind']} | {definitions} |")
        if len(missing) > 400:
            lines.append(f"| ... | | | {len(missing) - 400} more rows in the JSON artifact |")
    lines += [
        "",
        "## Full matrix",
        "",
        "| R symbol | package | Python symbol | S | K | P | A | tests |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in coverage["symbols"]:
        layers = row["layers"]
        lines.append(
            f"| {row['r_symbol']} | {row['r_package']} | {row['python_symbol'] or '-'} | "
            f"{layers['surface']} | {layers['kernel']} | {layers['parity']} | {layers['admission']} | "
            f"{row['evidence']['test_count']} |"
        )
    lines += [
        "",
        "## Python-only public exports",
        "",
        f"{stats['python_only_public_exports']} names exported from `omnilss.__all__` have no R"
        " counterpart; they are Python-side additions (dataclasses, backends, utilities), not"
        " coverage gaps. The JSON artifact lists them all.",
        "",
    ]
    for name in coverage["python_only_public_exports"][:40]:
        lines.append(f"- `{name}`")
    if stats["python_only_public_exports"] > 40:
        lines.append(f"- ... {stats['python_only_public_exports'] - 40} more in the JSON artifact")
    return "\n".join(lines) + "\n"


def dump_json(coverage: dict) -> str:
    """Serialise the matrix with one compact line per symbol row.

    Pretty printing 1080 rows with ``indent=2`` produces ~34k lines and makes
    every regeneration unreviewable. One row per line keeps the artifact valid
    JSON, diff friendly (D0.4 compares row by row) and still readable.
    """
    parts = []
    for key, value in coverage.items():
        if key == "symbols":
            rows = ",\n".join(f"    {json.dumps(row, ensure_ascii=False)}" for row in value)
            parts.append(f'  "symbols": [\n{rows}\n  ]')
        else:
            parts.append(
                f'  {json.dumps(key, ensure_ascii=False)}: '
                + json.dumps(value, indent=2, ensure_ascii=False).replace("\n", "\n  ")
            )
    return "{\n" + ",\n".join(parts) + "\n}\n"


def write_coverage(coverage: dict, output: Path = OUT_JSON, markdown: Path = OUT_MD) -> None:
    """Write the JSON matrix and its Markdown companion (LF endings, deterministic)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dump_json(coverage), encoding="utf-8", newline="\n")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(coverage), encoding="utf-8", newline="\n")


def check_artifacts(emit=print) -> int:
    """Fail when the checked-in artifacts drift from a fresh regeneration."""
    coverage = build_coverage(emit)
    expected = {OUT_JSON: dump_json(coverage), OUT_MD: render_markdown(coverage)}
    stale = [
        _rel(path)
        for path, content in expected.items()
        if (_read_text(path) if path.exists() else None) != content
    ]
    if stale:
        emit("stale coverage artifacts: " + ", ".join(stale))
        emit("run: python tools/generate_symbol_coverage.py")
        return 1
    emit("symbol coverage artifacts are up to date")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify artifacts are up to date, write nothing")
    parser.add_argument("--output", type=Path, default=OUT_JSON)
    parser.add_argument("--markdown", type=Path, default=OUT_MD)
    args = parser.parse_args(argv)
    if args.check and args.output == OUT_JSON and args.markdown == OUT_MD:
        return check_artifacts()
    coverage = build_coverage()
    write_coverage(coverage, args.output, args.markdown)
    stats = coverage["stats"]
    print(f"Generated: {args.output}")
    print(f"Generated: {args.markdown}")
    print(
        f"R exports: {stats['r_exports_total']} | surface matched: {stats['surface_matched']} | "
        f"missing: {stats['surface_missing']} | family-linked: {stats['family_linked']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
