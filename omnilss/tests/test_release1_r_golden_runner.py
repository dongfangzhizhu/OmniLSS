from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.run_release1_r_golden import (
    REPO_ROOT,
    REQUIRED_MANIFEST_FIELDS,
    GoldenCase,
    case_directory,
    default_no_case,
    run_no_case,
    validate_manifest,
)


# Validates: Requirements 2.1, 2.2, 2.7, 3.7

def _manifest(case: GoldenCase) -> dict[str, object]:
    return {
        "gamlss_source": {"version": "5.5-3", "path": "R/gamlss"},
        "gamlss_dist_source": {"version": "6.1-8", "path": "R/gamlss.dist"},
        "r_version": "R version test", "python_version": "3.12", "jax_version": "0.test",
        "jaxlib_version": "0.test", "numpy_version": "2.test", "scipy_version": "1.test",
        "commit_id": "abc", "dirty_worktree": False, "precision_policy": "jax-x64",
        "seed": case.seed, "formula": case.formula, "parameters": dict(case.parameters),
        "tolerances": dict(case.tolerances), "test_id": case.test_id, "result_path": "result.json",
    }


def test_deterministic_case_path_uses_worktree_or_commit(tmp_path: Path):
    case = default_no_case()
    assert case_directory(case, tmp_path, "a" * 40, False) == tmp_path / ("a" * 40) / case.test_id
    assert case_directory(case, tmp_path, "a" * 40, True) == tmp_path / "worktree" / case.test_id


def test_manifest_validator_requires_every_provenance_field():
    manifest = _manifest(default_no_case())
    validate_manifest(manifest)
    assert REQUIRED_MANIFEST_FIELDS <= set(manifest)
    manifest.pop("commit_id")
    with pytest.raises(ValueError, match="commit_id"):
        validate_manifest(manifest)


def test_x64_is_not_imported_at_module_startup_and_is_configured_on_demand():
    code = (
        "import sys; import tools.run_release1_r_golden as runner; "
        "assert 'jax' not in sys.modules; runner.enable_jax_x64(); "
        "import jax, jax.numpy as jnp; "
        "assert jax.config.jax_enable_x64 and jnp.asarray(1.0).dtype == jnp.float64"
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT / "omnilss", text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr


def test_property_provenance_and_seed_are_complete_for_100_cases(tmp_path: Path):
    """Property 3: every randomized case records complete replay provenance."""
    for seed in range(100):
        case = GoldenCase(
            test_id=f"no-{seed}", family="NO", operation="d", seed=seed,
            formula=None, parameters={"x": [float(seed)], "mu": 0.0, "sigma": 1.0},
            tolerances={"absolute": 1e-12},
        )
        manifest = _manifest(case)
        validate_manifest(manifest)
        assert REQUIRED_MANIFEST_FIELDS <= set(manifest)
        assert manifest["seed"] == seed
        assert case_directory(case, tmp_path, "commit", seed % 2 == 1).name == case.test_id


def test_fixed_r_no_case_writes_readable_complete_artifacts(tmp_path: Path):
    try:
        result_path, markdown_path, manifest_path = run_no_case(default_no_case(), tmp_path)
    except RuntimeError as error:
        pytest.fail(f"fixed R NO baseline must run: {error}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    assert REQUIRED_MANIFEST_FIELDS <= set(manifest)
    assert result["status"] == "passed"
    assert result["r_result"]["density"]
    assert result["python_result"]["density"]
    assert markdown_path.read_text(encoding="utf-8").startswith("# R Golden Artifact:")
