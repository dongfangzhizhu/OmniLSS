from __future__ import annotations

import json

from omnilss.benchmarks import BenchmarkRunner, BenchmarkSpec, default_dataset_registry, write_report


def test_dataset_registry_has_versioned_dataset():
    reg = default_dataset_registry()
    assert "normal_linear" in reg.names()
    assert reg.get("normal_linear").version


def test_benchmark_runner_emits_record_and_report(tmp_path):
    runner = BenchmarkRunner()
    record = runner.run(BenchmarkSpec(benchmark_type="correctness", family="NO", method="RS", dataset="normal_linear", n=32))
    assert record.converged
    out = tmp_path / "report.json"
    write_report(out, [record])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0.0"
    assert payload["records"][0]["benchmark_type"] == "correctness"
