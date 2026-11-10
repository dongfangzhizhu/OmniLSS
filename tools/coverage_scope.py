# SPDX-License-Identifier: GPL-3.0-or-later
"""Single source of truth for OmniLSS 1.0 scope declarations (P0-6).

Every coverage artifact must state, explicitly, which R packages are **out of
scope** for OmniLSS 1.0.  Keeping the constant, the human-readable notice and
the JSON payload in one module prevents the reports from drifting apart (the
P0-6 acceptance criterion is "gamlss2 out-of-scope annotated in EVERY report").

The generators are loaded in two different ways -- as a namespace package
(``tools.generate_symbol_coverage``, e.g. from pytest) and as a directly-run
script (``python tools/xxx.py``, sys.path[0] == tools/) -- so importers use a
try/except dual import; see ``_load_scope`` patterns in each generator.
"""
from __future__ import annotations

#: R packages outside the OmniLSS 1.0 scope (deliverable plan section 1.2).
#: ``gamlss2`` has a separate spec / version / evidence chain and does not
#: block 1.0.  Kept as a tuple so the value is immutable and reviewable.
OUT_OF_SCOPE_PACKAGES: tuple[str, ...] = ("gamlss2",)

#: Why out-of-scope symbols still appear in the matrix (and in raw counts).
SCOPE_COUNTING_NOTE = (
    "Out-of-scope exports are still enumerated in the per-symbol matrix for "
    "completeness and ARE included in raw totals (e.g. r_exports_total / "
    "surface_missing); they are NOT OmniLSS 1.0 delivery targets and do not "
    "count against 1.0 completeness. Filter on in_scope / in_scope_exports "
    "for the 1.0 figure."
)

#: The one-line human notice embedded at the top of every Markdown report.
def scope_line() -> str:
    packages = ", ".join(f"`{p}`" for p in OUT_OF_SCOPE_PACKAGES) or "none"
    return (
        f"> **Scope (P0-6):** {packages} is **OUT OF SCOPE** for OmniLSS 1.0 "
        "(separate spec / version / evidence chain; it does not block 1.0). "
        + SCOPE_COUNTING_NOTE
    )


#: The JSON block embedded at the top level of every JSON coverage report.
def scope_payload() -> dict:
    return {
        "out_of_scope_packages": sorted(OUT_OF_SCOPE_PACKAGES),
        "in_scope_definition": (
            "A symbol is in scope for 1.0 iff its R package is not listed in "
            "out_of_scope_packages. See stats.in_scope_exports for the count."
        ),
        "counting_note": SCOPE_COUNTING_NOTE,
        "notice": (
            "gamlss2 is out of scope for OmniLSS 1.0: separate spec / version / "
            "evidence, does not block 1.0 (deliverable plan sections 1.2, 22.6)."
        ),
    }
