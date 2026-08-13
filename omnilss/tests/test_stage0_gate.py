"""Phase 0 task 0.6 tests: stage-0 admission/completion gate.

Covers `tools/stage0_gate.py`:
- artifact verification (golden manifest provenance, fixture coverage,
  R reference results status, presence of key Phase 0 outputs);
- completion-record verification (record exists / completed / valid commit /
  Chinese subject / declared files tracked in the repository);
- fail-closed suite execution (require-r semantics);
- relaxed record command for historical tasks.

Validates: .kiro tasks.md 0.6; requirements 1.1-1.7, 2.1-2.8, 3.7, 7.1-7.4.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage0_gate import (  # noqa: E402
    STAGE0_TASKS,
    verify_completion_records,
    verify_stage0_artifacts,
)
from tools.stage_completion_checker import (  # noqa: E402
    STATUS_COMPLETED,
    STATUS_INCOMPLETE,
    CompletionRecord,
)

MANIFEST = "omnilss/tests/fixtures/r_golden/regenerated/manifest.json"
FIXTURE = "omnilss/tests/fixtures/r_golden/regenerated/no_y_x_seed_20260524.json"
GENERATE_R = "omnilss/tests/fixtures/r_golden/generate_golden.R"
R_REF = "benchmarks/r_reference_results.json"
RECORDS = "docs/reports/completion_records.json"
EVIDENCE = "docs/reports/phase0-task-0.5-evidence.json"

FIXTURE_NAMES = ("no_y_x_seed_20260524", "ga_y_x_seed_20260525")
PROVENANCE = {
    "family": "NO",
    "formula": "y ~ x",
    "seed": 20260524,
    "n": 90,
    "R_version": "R version 4.6.0",
    "gamlss_version": "5.5.0",
    "gamlss_dist_version": "6.1.1",
    "content_hash": "85523b6b",
}


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _good_root(tmp: Path) -> Path:
    root = tmp / "repo"
    _write(root / MANIFEST, json.dumps({n: dict(PROVENANCE) for n in FIXTURE_NAMES}))
    for name in FIXTURE_NAMES:
        _write(root / f"omnilss/tests/fixtures/r_golden/regenerated/{name}.json", "{}")
    _write(root / GENERATE_R, "# generate_golden.R\n")
    _write(root / R_REF, json.dumps({"status": "generated_by_r", "results": []}))
    _write(root / RECORDS, json.dumps({}))
    _write(root / EVIDENCE, "{}")
    return root


class TestArtifactVerification(unittest.TestCase):
    """Unit tests for verify_stage0_artifacts."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = _good_root(Path(self._tmp.name))

    def test_complete_artifact_set_passes(self):
        report = verify_stage0_artifacts(self.root)
        self.assertTrue(report.ok, [i.detail for i in report.issues])

    def test_missing_manifest_fails(self):
        (self.root / MANIFEST).unlink()
        report = verify_stage0_artifacts(self.root)
        self.assertFalse(report.ok)
        self.assertTrue(any("manifest 缺失" in i.detail for i in report.issues))

    def test_corrupt_manifest_fails(self):
        _write(self.root / MANIFEST, "{not json")
        report = verify_stage0_artifacts(self.root)
        self.assertFalse(report.ok)
        self.assertTrue(any("不可读" in i.detail for i in report.issues))

    def test_manifest_missing_provenance_fails(self):
        bad = {name: {"family": "NO"} for name in FIXTURE_NAMES}  # no provenance
        _write(self.root / MANIFEST, json.dumps(bad))
        report = verify_stage0_artifacts(self.root)
        self.assertFalse(report.ok)
        self.assertTrue(any("缺 provenance 字段" in i.detail for i in report.issues))

    def test_uncovered_fixture_fails(self):
        # A checked-in fixture not present in the manifest.
        _write(self.root / "omnilss/tests/fixtures/r_golden/regenerated/extra_y_x_seed_1.json", "{}")
        report = verify_stage0_artifacts(self.root)
        self.assertFalse(report.ok)
        self.assertTrue(any("未被 manifest 覆盖" in i.detail for i in report.issues))

    def test_missing_r_reference_fails(self):
        (self.root / R_REF).unlink()
        report = verify_stage0_artifacts(self.root)
        self.assertFalse(report.ok)
        self.assertTrue(any("R 参考结果" in i.detail for i in report.issues))

    def test_r_reference_without_generated_by_r_fails(self):
        _write(self.root / R_REF, json.dumps({"status": "placeholder"}))
        report = verify_stage0_artifacts(self.root)
        self.assertFalse(report.ok)
        self.assertTrue(any("status != generated_by_r" in i.detail for i in report.issues))

    def test_missing_completion_records_fails(self):
        (self.root / RECORDS).unlink()
        report = verify_stage0_artifacts(self.root)
        self.assertFalse(report.ok)
        self.assertTrue(any("完成记录 缺失" in i.detail for i in report.issues))


class _GitHelper(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "repo"
        self.root.mkdir()

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def _init_repo(self, files: dict[str, str], subject: str) -> str:
        self._git("init", "-q")
        self._git("config", "user.name", "Tester")
        self._git("config", "user.email", "tester@example.com")
        for rel, content in files.items():
            _write(self.root / rel, content)
        self._git("add", "-A")
        self.assertEqual(self._git("commit", "-q", "-m", subject).returncode, 0)
        return self._git("rev-parse", "HEAD").stdout.strip()

    def _record(
        self, task_id: str, commit_id: str, subject: str,
        impl=("src/x.py",), test=("tests/test_x.py",), artifact=(FIXTURE,),
    ) -> CompletionRecord:
        return CompletionRecord(
            task_id=task_id,
            commit_id=commit_id,
            subject=subject,
            status=STATUS_COMPLETED,
            verification_summary="",
            artifact_paths=tuple(artifact),
            impl_paths=tuple(impl),
            test_paths=tuple(test),
        )

    def _write_records(self, records: dict[str, CompletionRecord]) -> None:
        path = self.root / RECORDS
        _write(
            path,
            json.dumps({tid: rec.to_dict() for tid, rec in records.items()}, ensure_ascii=False),
        )


class TestCompletionRecordVerification(_GitHelper):
    """Unit tests for verify_completion_records."""

    def _verify(self, tasks=STAGE0_TASKS):
        return verify_completion_records(self.root, tasks=tasks)

    def test_all_tasks_recorded_passes(self):
        files = {"src/x.py": "x\n", "tests/test_x.py": "x\n", FIXTURE: "{}\n"}
        sha = self._init_repo(files, "完成Phase任务0.1：测试提交")
        records = {tid: self._record(tid, sha, "完成Phase任务0.1：测试提交") for tid in STAGE0_TASKS}
        self._write_records(records)
        report = self._verify()
        self.assertTrue(report.ok, [i.detail for i in report.issues])
        self.assertEqual(len([i for i in report.issues if i.ok]), len(STAGE0_TASKS))

    def test_missing_record_fails(self):
        sha = self._init_repo({"src/x.py": "x\n", "tests/test_x.py": "x\n", FIXTURE: "{}\n"},
                              "完成Phase任务0.1：测试提交")
        records = {tid: self._record(tid, sha, "完成Phase任务0.1：测试提交")
                   for tid in ("0.1", "0.2")}
        self._write_records(records)
        report = self._verify(tasks=["0.1", "0.2", "0.3"])
        self.assertFalse(report.ok)
        self.assertTrue(any("缺少 completion record" in i.detail and "0.3" in i.detail
                            for i in report.issues))

    def test_incomplete_record_fails(self):
        sha = self._init_repo({"src/x.py": "x\n", "tests/test_x.py": "x\n", FIXTURE: "{}\n"},
                              "完成Phase任务0.1：测试提交")
        rec = self._record("0.1", sha, "完成Phase任务0.1：测试提交")
        rec.status = STATUS_INCOMPLETE
        self._write_records({"0.1": rec})
        report = self._verify(tasks=["0.1"])
        self.assertFalse(report.ok)
        self.assertTrue(any("未完成" in i.detail for i in report.issues))

    def test_invalid_commit_fails(self):
        self._init_repo({"src/x.py": "x\n", "tests/test_x.py": "x\n", FIXTURE: "{}\n"},
                        "完成Phase任务0.1：测试提交")
        self._write_records({"0.1": self._record("0.1", "0" * 40, "完成Phase任务0.1：测试提交")})
        report = self._verify(tasks=["0.1"])
        self.assertFalse(report.ok)
        self.assertTrue(any("commit" in i.detail and "无效" in i.detail for i in report.issues))

    def test_non_chinese_subject_fails(self):
        sha = self._init_repo({"src/x.py": "x\n", "tests/test_x.py": "x\n", FIXTURE: "{}\n"},
                              "finish task 0.1")
        self._write_records({"0.1": self._record("0.1", sha, "finish task 0.1")})
        report = self._verify(tasks=["0.1"])
        self.assertFalse(report.ok)
        self.assertTrue(any("非中文" in i.detail for i in report.issues))

    def test_declared_files_must_be_tracked(self):
        sha = self._init_repo({"src/x.py": "x\n", "tests/test_x.py": "x\n", FIXTURE: "{}\n"},
                              "完成Phase任务0.1：测试提交")
        rec = self._record("0.1", sha, "完成Phase任务0.1：测试提交",
                           impl=("src/missing.py",))
        self._write_records({"0.1": rec})
        report = self._verify(tasks=["0.1"])
        self.assertFalse(report.ok)
        self.assertTrue(any("不在仓库" in i.detail for i in report.issues))

    def test_no_declared_files_fails(self):
        sha = self._init_repo({"src/x.py": "x\n", "tests/test_x.py": "x\n", FIXTURE: "{}\n"},
                              "完成Phase任务0.1：测试提交")
        rec = self._record("0.1", sha, "完成Phase任务0.1：测试提交",
                           impl=(), test=(), artifact=())
        self._write_records({"0.1": rec})
        report = self._verify(tasks=["0.1"])
        self.assertFalse(report.ok)
        self.assertTrue(any("未声明" in i.detail for i in report.issues))


class TestRecordCommand(_GitHelper):
    """Unit tests for the relaxed historical record command."""

    def _run_record(self, argv: list[str]) -> int:
        from tools.stage0_gate import main

        return main(argv)

    def test_record_historical_task(self):
        files = {"src/x.py": "x\n", "tests/test_x.py": "x\n", FIXTURE: "{}\n"}
        sha = self._init_repo(files, "完成阶段0.4：历史提交")
        store = self.root / "records.json"
        rc = self._run_record(
            ["record", "--task", "0.4", "--commit", sha,
             "--store", str(store), "--repo", str(self.root),
             "--summary", "历史任务补记"]
        )
        self.assertEqual(rc, 0)
        data = json.loads(store.read_text(encoding="utf-8"))
        self.assertEqual(data["0.4"]["status"], STATUS_COMPLETED)
        self.assertIn("src/x.py", data["0.4"]["impl_paths"])
        self.assertIn("tests/test_x.py", data["0.4"]["test_paths"])
        self.assertEqual(data["0.4"]["verification_summary"], "历史任务补记")

    def test_record_with_explicit_paths(self):
        files = {"omnilss/src/omnilss/_precision.py": "x\n",
                 "omnilss/tests/test_x64_precision.py": "x\n"}
        sha = self._init_repo(files, "完成阶段0.7：x64 精度")
        store = self.root / "records.json"
        rc = self._run_record(
            ["record", "--task", "0.7", "--commit", sha,
             "--store", str(store), "--repo", str(self.root),
             "--impl", "omnilss/src/omnilss/_precision.py",
             "--test", "omnilss/tests/test_x64_precision.py",
             "--artifact", "omnilss/src/omnilss/_precision.py"]
        )
        self.assertEqual(rc, 0)
        data = json.loads(store.read_text(encoding="utf-8"))
        self.assertEqual(data["0.7"]["status"], STATUS_COMPLETED)
        self.assertEqual(data["0.7"]["artifact_paths"], ["omnilss/src/omnilss/_precision.py"])

    def test_record_invalid_commit_fails(self):
        store = self.root / "records.json"
        self._init_repo({"src/x.py": "x\n"}, "完成阶段0.4：历史提交")
        rc = self._run_record(
            ["record", "--task", "0.4", "--commit", "0" * 40,
             "--store", str(store), "--repo", str(self.root)]
        )
        self.assertEqual(rc, 1)
        data = json.loads(store.read_text(encoding="utf-8"))
        self.assertEqual(data["0.4"]["status"], STATUS_INCOMPLETE)
        self.assertTrue(data["0.4"]["reasons"])

    def test_record_declares_missing_tracked_file_fails(self):
        sha = self._init_repo({"src/x.py": "x\n"}, "完成阶段0.4：历史提交")
        store = self.root / "records.json"
        rc = self._run_record(
            ["record", "--task", "0.4", "--commit", sha,
             "--store", str(store), "--repo", str(self.root),
             "--impl", "src/not-tracked.py"]
        )
        self.assertEqual(rc, 1)
        data = json.loads(store.read_text(encoding="utf-8"))
        self.assertEqual(data["0.4"]["status"], STATUS_INCOMPLETE)


class TestSuiteExecution(unittest.TestCase):
    """Unit tests for run_phase0_suite fail-closed behaviour."""

    def test_suite_pass_exit_0_and_require_r_flag(self):
        from tools.stage0_gate import run_phase0_suite

        with mock.patch(
            "tools.stage0_gate.subprocess.run",
            return_value=mock.Mock(returncode=0, stdout="ok", stderr=""),
        ) as mocked:
            report = run_phase0_suite(ROOT, require_r=True)
        self.assertTrue(report.ok, [i.detail for i in report.issues])
        cmd = mocked.call_args[0][0]
        self.assertIn("--suite", cmd)
        self.assertIn("phase0_gate", cmd)
        self.assertIn("--require-r", cmd)

    def test_suite_fail_exit_nonzero(self):
        from tools.stage0_gate import run_phase0_suite

        with mock.patch(
            "tools.stage0_gate.subprocess.run",
            return_value=mock.Mock(returncode=1, stdout="boom", stderr=""),
        ):
            report = run_phase0_suite(ROOT, require_r=True)
        self.assertFalse(report.ok)
        self.assertTrue(any("suite 失败" in i.detail for i in report.issues))

    def test_suite_timeout_fails_closed(self):
        from tools.stage0_gate import run_phase0_suite

        with mock.patch(
            "tools.stage0_gate.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["x"], timeout=3600),
        ):
            report = run_phase0_suite(ROOT, require_r=True)
        self.assertFalse(report.ok)
        self.assertTrue(any("超时" in i.detail for i in report.issues))


class TestRealRepository(unittest.TestCase):
    """Integration: the actual OmniLSS repository must satisfy the gate."""

    def test_artifacts_and_completion_records_verify(self):
        report = verify_stage0_artifacts(ROOT)
        self.assertTrue(report.ok, [i.detail for i in report.issues])
        # 0.6's own completion record is written by the gate workflow after
        # the gate passes, so verify the other seven Phase 0 tasks here.
        tasks = [t for t in STAGE0_TASKS if t != "0.6"]
        records = verify_completion_records(ROOT, tasks=tasks)
        self.assertTrue(records.ok, [i.detail for i in records.issues])


if __name__ == "__main__":
    unittest.main()
