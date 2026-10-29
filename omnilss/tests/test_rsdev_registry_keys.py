"""Registry tables must not repeat a family key (D2.96).

Every rsdev evidence table is a dict literal.  In a literal, the LAST
occurrence of a key wins -- so a family registered twice silently keeps only
the later entry, and an edit applied to the earlier one does nothing at all.
That is exactly what happened with GT: it had two entries (one from D2.68
with pre-D2.83 numbers, one added later), a re-measurement was applied to the
first, and the gate kept reporting the stale verdict until the entry was
moved to the end of the table.

These tests assert the tables are key-unique, and that a family appears in at
most one rsdev table (a family cannot be both "R stopped early" and
"R diverged" -- the two are different diagnoses).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

import rs_family_parity_scan as scan  # noqa: E402

_TABLES = (
    "_RSDEV_R_DIVERGED",
    "_RSDEV_R_STOPPED_EARLY",
    "_RSDEV_PATH_DIVERGENT",
    "_RSDEV_R_DENSITY_RECYCLED",
)

_SOURCE = (TOOLS / "rs_family_parity_scan.py").read_text(encoding="utf-8")


def _source_keys(table_name: str) -> list[str]:
    """The keys written in the literal, duplicates included."""
    head = _SOURCE.index(table_name + ": dict")
    open_brace = _SOURCE.index(" = {", head)
    depth = 0
    cursor = open_brace
    while True:
        char = _SOURCE[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        cursor += 1
    body = _SOURCE[open_brace:cursor]
    return re.findall(r'^    "([A-Za-z0-9_]+)": dict\(', body, re.M)


def test_no_table_repeats_a_family_key():
    for table_name in _TABLES:
        keys = _source_keys(table_name)
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        assert duplicates == [], (table_name, duplicates)


def test_the_runtime_table_matches_what_is_written():
    """A silent override would show up as source keys outnumbering runtime keys."""
    for table_name in _TABLES:
        written = _source_keys(table_name)
        runtime = getattr(scan, table_name)
        assert len(written) == len(runtime), (table_name, len(written), len(runtime))


def test_a_family_is_diagnosed_in_at_most_one_table():
    seen: dict[str, str] = {}
    for table_name in _TABLES:
        for family in getattr(scan, table_name):
            assert family not in seen, (family, seen[family], table_name)
            seen[family] = table_name