from __future__ import annotations

import random
from pathlib import Path

import pytest

from tools.generate_release1_evidence_matrix import (
    CAPABILITY_STATUSES,
    CORE_PACKAGES,
    EXCLUDED_PACKAGES,
    build_matrix,
    markdown_report,
    parse_namespace_exports,
    source_evidence,
    validate_matrix,
    write_artifacts,
)


# Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_matrix_covers_every_fixed_core_r_export_once():
    matrix = build_matrix(REPO_ROOT)
    expected = {
        (package, symbol)
        for package in CORE_PACKAGES
        for symbol in parse_namespace_exports(REPO_ROOT / "R" / package / "NAMESPACE")
    }
    actual = {(entry["r_package"], entry["r_symbol"]) for entry in matrix["entries"]}
    assert actual == expected
    assert len(actual) == len(matrix["entries"])


def test_matrix_explicitly_limits_scope_and_serializes_all_exclusions():
    matrix = build_matrix(REPO_ROOT)
    assert matrix["included_r_packages"] == list(CORE_PACKAGES)
    exclusions = {item["package"]: item["reason"] for item in matrix["excluded_r_packages"]}
    assert exclusions == dict(EXCLUDED_PACKAGES)
    assert all(exclusions.values())


def test_entries_use_only_release_statuses_and_public_families_are_traceable():
    matrix = build_matrix(REPO_ROOT)
    for entry in matrix["entries"]:
        assert entry["status"] in CAPABILITY_STATUSES
        if entry["python_entry"]:
            assert entry["kernel_owner"]
            assert entry["implementation_origin"]
            assert entry["evidence_references"]
            assert entry["r_symbol"]


def test_fixed_r_source_versions_are_recorded_with_hashes():
    evidence = source_evidence(REPO_ROOT)
    assert evidence["gamlss"]["version"] == "5.5-3"
    assert evidence["gamlss.dist"]["version"] == "6.1-8"
    assert all(len(item["namespace_sha256"]) == 64 for item in evidence.values())


def test_artifacts_are_json_authoritative_and_markdown_is_derived(tmp_path: Path):
    matrix_path, report_path, evidence_path = write_artifacts(tmp_path, REPO_ROOT)
    assert matrix_path.exists() and evidence_path.exists()
    matrix = build_matrix(REPO_ROOT)
    assert report_path.read_text(encoding="utf-8") == markdown_report(matrix)
    assert set(matrix["metrics"]) == {
        "exported_symbol_coverage", "callable_python_entry_coverage",
        "dedicated_kernel_coverage", "e3_family_count", "e4_feature_count",
    }


def test_property_matrix_completeness_statuses_scope_and_traceability_100_runs():
    """Property test: invariant Release 1 metadata holds across 100 executions."""
    for seed in range(100):
        random.Random(seed).shuffle(list(CORE_PACKAGES))
        matrix = build_matrix(REPO_ROOT)
        validate_matrix(matrix)
        assert set(matrix["included_r_packages"]) == set(CORE_PACKAGES)
        assert {row["package"] for row in matrix["excluded_r_packages"]} == set(EXCLUDED_PACKAGES)
        assert all(row["status"] in CAPABILITY_STATUSES for row in matrix["entries"])
        assert all(
            row["kernel_owner"] and row["implementation_origin"] and row["evidence_references"]
            for row in matrix["entries"] if row["python_entry"]
        )


def test_validator_rejects_missing_public_family_traceability():
    matrix = build_matrix(REPO_ROOT)
    entry = next(item for item in matrix["entries"] if item["python_entry"])
    entry["kernel_owner"] = None
    with pytest.raises(ValueError, match="lacks traceability"):
        validate_matrix(matrix)
