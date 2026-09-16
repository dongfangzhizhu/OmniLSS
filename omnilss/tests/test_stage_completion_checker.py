"""Phase 0 task 0.5 tests: stage completion checker & Chinese-commit integrity.

Covers `tools/stage_completion_checker.py`:
- passing completion (implementation + tests + evidence + Chinese task commit);
- missing tests / missing evidence / missing implementation;
- invalid commit id / non-Chinese subject / task-mismatched subject;
- explicit artifact path requirements and relaxed prerequisites;
- completion-record persistence (never auto-completes an unverified task);
- phase-level aggregation with integrity check;
- real git integration (GitCommitInspector) and CLI exit codes.

Property tests run >= 100 hypothesis examples each (settings(max_examples=100)).

Validates: .kiro requirements 7.1, 7.2, 7.3, 7.4; design Property 14.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage_completion_checker import (  # noqa: E402
    STATUS_COMPLETED,
    STATUS_INCOMPLETE,
    CommitInfo,
    CompletionRecordStore,
    FakeCommitInspector,
    GitCommitInspector,
    check_phase_records,
    contains_chinese,
    extract_task_ids,
    is_evidence_file,
    is_implementation_file,
    is_test_file,
    validate_task_completion,
)

try:  # pragma: no cover - exercised only when hypothesis is missing
    from hypothesis import given, settings, strategies as st
    _HYPOTHESIS_AVAILABLE = True
except ImportError:  # pragma: no cover

    def given(*args, **kwargs):  # type: ignore[misc]
        def deco(fn):
            def skipped(self, *test_args, **test_kwargs):
                self.skipTest("hypothesis not available")

            return skipped

        return deco

    def settings(*args, **kwargs):  # type: ignore[misc]
        def deco(fn):
            return fn

        return deco

    class _UnavailableStrategies:
        """Import-time stand-in so skipped property tests can be collected."""

        def composite(self, fn):
            return lambda *args, **kwargs: None

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    st = _UnavailableStrategies()  # type: ignore[assignment]
    _HYPOTHESIS_AVAILABLE = False


IMPL = "omnilss/src/omnilss/fitting.py"
TEST = "omnilss/tests/test_fitting.py"
EVID = "omnilss/tests/fixtures/r_golden/regenerated/no.json"
SUBJECT_OK = "完成Phase任务0.5：实现阶段完成检查器"
SHA_OK = "a" * 40


def _make_commit(commit_id: str, subject: str, files: list[str]) -> CommitInfo:
    return CommitInfo(commit_id=commit_id, subject=subject, changed_files=tuple(files))


def _fake_inspector(commit_id: str, subject: str, files: list[str]) -> FakeCommitInspector:
    return FakeCommitInspector({commit_id: _make_commit(commit_id, subject, files)})


class TestValidation(unittest.TestCase):
    """Unit tests for validate_task_completion failure modes."""

    def _check(self, commit_id, subject, files, task_id="0.5", **kwargs):
        inspector = _fake_inspector(commit_id, subject, files)
        return validate_task_completion(task_id, commit_id, inspector, **kwargs)

    def test_valid_completion_passes(self):
        result = self._check(SHA_OK, SUBJECT_OK, [IMPL, TEST, EVID])
        self.assertTrue(result.ok)
        for name in (
            "commit_valid",
            "subject_has_chinese",
            "subject_identifies_task",
            "contains_implementation",
            "contains_tests",
            "contains_evidence",
        ):
            self.assertTrue(result.checks[name], name)
        self.assertEqual(result.reasons, [])
        self.assertEqual(result.commit.subject, SUBJECT_OK)

    def test_missing_tests_fails(self):
        result = self._check(SHA_OK, SUBJECT_OK, [IMPL, EVID])
        self.assertFalse(result.ok)
        self.assertFalse(result.checks["contains_tests"])
        self.assertTrue(any("测试" in reason for reason in result.reasons))

    def test_missing_evidence_fails(self):
        result = self._check(SHA_OK, SUBJECT_OK, [IMPL, TEST])
        self.assertFalse(result.ok)
        self.assertFalse(result.checks["contains_evidence"])
        self.assertTrue(any("证据" in reason for reason in result.reasons))

    def test_missing_implementation_fails(self):
        result = self._check(SHA_OK, SUBJECT_OK, [TEST, EVID])
        self.assertFalse(result.ok)
        self.assertFalse(result.checks["contains_implementation"])
        self.assertTrue(any("实现" in reason for reason in result.reasons))

    def test_invalid_commit_fails(self):
        # The inspector only knows SHA_OK, so a different id cannot resolve.
        inspector = _fake_inspector(SHA_OK, SUBJECT_OK, [IMPL, TEST, EVID])
        result = validate_task_completion("0.5", "b" * 40, inspector)
        self.assertFalse(result.ok)
        self.assertFalse(result.checks["commit_valid"])
        # Remaining checks short-circuit to False; a reason is recorded.
        self.assertFalse(result.checks["subject_has_chinese"])
        self.assertTrue(result.reasons)
        self.assertIsNone(result.commit)

    def test_non_chinese_subject_fails(self):
        result = self._check(SHA_OK, "finish task 0.5 implementation", [IMPL, TEST, EVID])
        self.assertFalse(result.ok)
        self.assertFalse(result.checks["subject_has_chinese"])
        self.assertTrue(any("中文" in reason for reason in result.reasons))

    def test_task_mismatch_subject_fails(self):
        # Subject identifies 8.6, not the requested 0.5.
        result = self._check(
            SHA_OK, "完成Phase任务8.6：更新开发计划状态", [IMPL, TEST, EVID], task_id="0.5"
        )
        self.assertFalse(result.ok)
        self.assertFalse(result.checks["subject_identifies_task"])
        self.assertTrue(any("8.6" in reason for reason in result.reasons))

    def test_subject_without_task_marker_fails(self):
        result = self._check(
            SHA_OK, "完成Phase 8 收尾工作：清理死代码", [IMPL, TEST, EVID], task_id="0.5"
        )
        self.assertFalse(result.checks["subject_identifies_task"])
        self.assertTrue(any("未标识" in reason for reason in result.reasons))

    def test_explicit_artifact_paths_required(self):
        ok = self._check(
            SHA_OK, SUBJECT_OK, [IMPL, TEST, EVID], required_artifact_paths=[EVID]
        )
        self.assertTrue(ok.ok)
        self.assertTrue(ok.checks["contains_evidence"])
        missing = self._check(
            SHA_OK, SUBJECT_OK, [IMPL, TEST], required_artifact_paths=[EVID]
        )
        self.assertFalse(missing.ok)
        self.assertFalse(missing.checks["contains_evidence"])
        self.assertTrue(any("artifact" in reason for reason in missing.reasons))

    def test_optional_requirements_can_be_relaxed(self):
        result = self._check(
            SHA_OK, SUBJECT_OK, [IMPL], require_tests=False, require_evidence=False
        )
        self.assertTrue(result.ok)
        self.assertTrue(result.checks["contains_tests"])  # not required -> True
        self.assertTrue(result.checks["contains_evidence"])


class TestSubjectParsing(unittest.TestCase):
    """Unit tests for subject parsing helpers."""

    def test_contains_chinese(self):
        self.assertTrue(contains_chinese("完成Phase任务0.5"))
        self.assertTrue(contains_chinese("任务 0.5 完成"))
        self.assertFalse(contains_chinese("finish task"))
        self.assertFalse(contains_chinese(""))
        self.assertFalse(contains_chinese("0123456789"))

    def test_extract_task_ids_variants(self):
        self.assertEqual(extract_task_ids("完成Phase 8任务8.6：更新开发计划"), ["8.6"])
        self.assertEqual(extract_task_ids("任务0.5 完成"), ["0.5"])
        self.assertEqual(extract_task_ids("完成 task 1.2 实现"), ["1.2"])
        self.assertEqual(extract_task_ids("完成任务0.5与任务0.6"), ["0.5", "0.6"])
        self.assertEqual(extract_task_ids("完成任务0.5与任务0.5"), ["0.5"])

    def test_version_numbers_not_mistaken_for_task_ids(self):
        self.assertEqual(extract_task_ids("版本冻结0.4.0"), [])
        self.assertEqual(extract_task_ids("v0.4.0 release"), [])
        self.assertEqual(extract_task_ids(""), [])
        self.assertEqual(extract_task_ids("整理项目：忽略临时文件"), [])


class TestFileClassification(unittest.TestCase):
    """Unit tests for implementation / test / evidence file classification."""

    def test_implementation_file(self):
        self.assertTrue(is_implementation_file("omnilss/src/omnilss/fitting.py"))
        self.assertTrue(is_implementation_file("tools/stage_completion_checker.py"))
        self.assertFalse(is_implementation_file("omnilss/tests/test_fitting.py"))
        self.assertFalse(is_implementation_file("omnilss/src/omnilss/test_util.py"))

    def test_test_file(self):
        for path in (
            "omnilss/tests/test_fitting.py",
            "tests/test_fitting.py",
            "omnilss/tests/util_test.py",
            "omnilss/src/omnilss/tests/helper.py",
        ):
            self.assertTrue(is_test_file(path), path)
        self.assertFalse(is_test_file("omnilss/src/omnilss/fitting.py"))
        self.assertFalse(is_test_file("README.md"))

    def test_evidence_file(self):
        for path in (
            "omnilss/tests/fixtures/r_golden/regenerated/no.json",
            "omnilss/tests/fixtures/r_golden/generate_golden.R",
            "benchmarks/results/raw/r1.json",
            "docs/reports/audit.md",
            "docs/validation/comparison.md",
            "benchmarks/r_reference_results.json",
            "omnilss/tests/fixtures/r_golden/manifest.json",
        ):
            self.assertTrue(is_evidence_file(path), path)
        self.assertFalse(is_evidence_file("omnilss/src/omnilss/fitting.py"))
        self.assertFalse(is_evidence_file("README.md"))
        self.assertFalse(is_evidence_file("omnilss/tests/test_fitting.py"))


class TestCompletionRecordStore(unittest.TestCase):
    """Unit tests for persistence and no-auto-completion semantics."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store_path = Path(self._tmp.name) / "records.json"

    def test_completed_record_persists_all_fields(self):
        inspector = _fake_inspector(SHA_OK, SUBJECT_OK, [IMPL, TEST, EVID])
        result = validate_task_completion("0.5", SHA_OK, inspector)
        store = CompletionRecordStore(self.store_path)
        rec = store.record(
            result,
            verification_summary="8 项测试全绿",
            artifact_paths=[EVID],
        )
        self.assertEqual(rec.status, STATUS_COMPLETED)
        self.assertEqual(rec.commit_id, SHA_OK)
        self.assertEqual(rec.subject, SUBJECT_OK)
        self.assertEqual(rec.artifact_paths, (EVID,))
        self.assertEqual(rec.impl_paths, (IMPL,))
        self.assertEqual(rec.test_paths, (TEST,))
        self.assertEqual(rec.reasons, ())

        loaded = store.load()
        self.assertIn("0.5", loaded)
        self.assertEqual(loaded["0.5"].status, STATUS_COMPLETED)
        self.assertEqual(loaded["0.5"].verification_summary, "8 项测试全绿")
        self.assertEqual(loaded["0.5"].commit_id, SHA_OK)
        self.assertEqual(loaded["0.5"].artifact_paths, (EVID,))
        # JSON roundtrip keeps the record stable.
        raw = json.loads(self.store_path.read_text(encoding="utf-8"))
        self.assertEqual(raw["0.5"]["status"], STATUS_COMPLETED)

    def test_invalid_result_never_auto_completes(self):
        inspector = _fake_inspector(SHA_OK, SUBJECT_OK, [IMPL, EVID])  # no tests
        result = validate_task_completion("0.5", SHA_OK, inspector)
        self.assertFalse(result.ok)
        store = CompletionRecordStore(self.store_path)
        rec = store.record(result, verification_summary="尝试自动完成")
        self.assertEqual(rec.status, STATUS_INCOMPLETE)
        self.assertTrue(rec.reasons)
        self.assertEqual(store.load()["0.5"].status, STATUS_INCOMPLETE)

    def test_invalid_commit_record_keeps_incomplete(self):
        inspector = _fake_inspector(SHA_OK, SUBJECT_OK, [IMPL, TEST, EVID])
        result = validate_task_completion("0.5", "b" * 40, inspector)
        self.assertFalse(result.ok)
        store = CompletionRecordStore(self.store_path)
        rec = store.record(result)
        self.assertEqual(rec.status, STATUS_INCOMPLETE)
        self.assertTrue(any("无法解析" in r for r in rec.reasons))


class TestPhaseAggregation(unittest.TestCase):
    """Unit tests for phase-level completion checks."""

    def _record(self, task_id, status=STATUS_COMPLETED) -> dict:
        from tools.stage_completion_checker import CompletionRecord

        return {
            task_id: CompletionRecord(
                task_id=task_id,
                commit_id=SHA_OK,
                subject=SUBJECT_OK,
                status=status,
            )
        }

    def test_all_recorded_completed_passes(self):
        records = {}
        for tid in ("0.1", "0.2", "0.3", "0.4", "0.5"):
            records.update(self._record(tid))
        result = check_phase_records(records, "0")
        self.assertTrue(result.passed)
        self.assertEqual(result.incomplete_tasks, [])
        self.assertEqual(result.missing_tasks, ())

    def test_incomplete_record_fails(self):
        records = {}
        for tid in ("0.1", "0.2"):
            records.update(self._record(tid))
        records.update(self._record("0.5", STATUS_INCOMPLETE))
        result = check_phase_records(records, "0")
        self.assertFalse(result.passed)
        self.assertIn("0.5", result.incomplete_tasks)

    def test_integrity_check_requires_expected_tasks(self):
        records = {}
        for tid in ("0.1", "0.5"):
            records.update(self._record(tid))
        result = check_phase_records(
            records, "0", expected_task_ids=["0.1", "0.2", "0.3", "0.4", "0.5"]
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.missing_tasks, ("0.2", "0.3", "0.4"))
        full = check_phase_records(
            records, "0", expected_task_ids=["0.1", "0.5"]
        )
        self.assertTrue(full.passed)

    def test_other_phases_ignored(self):
        records = {}
        for tid in ("0.5", "8.6"):
            records.update(self._record(tid))
        result = check_phase_records(records, "0")
        self.assertIn("0.5", result.records)
        self.assertNotIn("8.6", result.records)
        self.assertTrue(result.passed)


class TestGitIntegration(unittest.TestCase):
    """Integration tests against real git repositories."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._root = Path(self._tmp.name)

    def _git(self, repo: Path, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def _make_repo(
        self, files: dict[str, str], subject: str, name: str = "repo"
    ) -> tuple[Path, str]:
        repo = self._root / name
        repo.mkdir()
        self.assertEqual(self._git(repo, "init", "-q").returncode, 0)
        self._git(repo, "config", "user.name", "Tester")
        self._git(repo, "config", "user.email", "tester@example.com")
        for rel, content in files.items():
            target = repo / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        self.assertEqual(self._git(repo, "add", "-A").returncode, 0)
        commit = self._git(repo, "commit", "-q", "-m", subject)
        self.assertEqual(commit.returncode, 0, commit.stderr)
        sha = self._git(repo, "rev-parse", "HEAD").stdout.strip()
        self.assertTrue(sha)
        return repo, sha

    def _good_files(self) -> dict[str, str]:
        return {
            "src/omnilss/checker.py": "def check():\n    return True\n",
            "tests/test_checker.py": "def test_check():\n    assert check()\n",
            "tests/fixtures/r_golden/regenerated/no.json": "{}\n",
        }

    def test_real_git_valid_completion(self):
        repo, sha = self._make_repo(self._good_files(), SUBJECT_OK)
        inspector = GitCommitInspector(repo)
        result = validate_task_completion("0.5", sha, inspector)
        self.assertTrue(result.ok)
        self.assertTrue(result.checks["commit_valid"])
        self.assertEqual(result.commit.subject, SUBJECT_OK)
        self.assertIn("src/omnilss/checker.py", result.commit.changed_files)
        self.assertIn("tests/test_checker.py", result.commit.changed_files)
        self.assertIn("tests/fixtures/r_golden/regenerated/no.json", result.commit.changed_files)

    def test_short_sha_and_full_sha_resolve(self):
        repo, sha = self._make_repo(self._good_files(), SUBJECT_OK)
        inspector = GitCommitInspector(repo)
        short = inspector.resolve(sha[:8])
        self.assertIsNotNone(short)
        self.assertEqual(short.commit_id, sha)
        full = inspector.resolve(sha)
        self.assertEqual(full.commit_id, sha)

    def test_invalid_commit_ids_do_not_resolve(self):
        repo, _ = self._make_repo(self._good_files(), SUBJECT_OK)
        inspector = GitCommitInspector(repo)
        self.assertIsNone(inspector.resolve("0" * 40))
        self.assertIsNone(inspector.resolve("not-a-sha"))
        self.assertIsNone(inspector.resolve(""))

    def test_non_chinese_subject_fails_against_real_git(self):
        repo, sha = self._make_repo(self._good_files(), "finish checker implementation")
        result = validate_task_completion("0.5", sha, GitCommitInspector(repo))
        self.assertFalse(result.checks["subject_has_chinese"])
        self.assertFalse(result.ok)

    def test_task_mismatch_fails_against_real_git(self):
        repo, sha = self._make_repo(self._good_files(), "完成Phase任务8.6：更新开发计划")
        result = validate_task_completion("0.5", sha, GitCommitInspector(repo))
        self.assertFalse(result.checks["subject_identifies_task"])
        self.assertFalse(result.ok)

    def test_cli_check_exit_codes(self):
        from tools.stage_completion_checker import main

        repo, sha = self._make_repo(self._good_files(), SUBJECT_OK)
        self.assertEqual(
            main(["check", "--task", "0.5", "--commit", sha, "--repo", str(repo)]), 0
        )
        self.assertEqual(
            main(["check", "--task", "0.5", "--commit", "0" * 40, "--repo", str(repo)]), 1
        )
        # Missing tests in the commit -> fail-closed.
        repo2, sha2 = self._make_repo(
            {
                "src/omnilss/checker.py": "def check():\n    return True\n",
                "tests/fixtures/r_golden/regenerated/no.json": "{}\n",
            },
            SUBJECT_OK,
            name="repo2",
        )
        self.assertEqual(
            main(["check", "--task", "0.5", "--commit", sha2, "--repo", str(repo2)]), 1
        )

    def test_cli_record_and_phase_workflow(self):
        from tools.stage_completion_checker import main

        store = self._root / "records.json"
        repo, sha = self._make_repo(self._good_files(), SUBJECT_OK)
        self.assertEqual(
            main(
                [
                    "record", "--task", "0.5", "--commit", sha,
                    "--repo", str(repo), "--store", str(store),
                    "--summary", "8 项测试全绿", "--artifact",
                    "tests/fixtures/r_golden/regenerated/no.json",
                ]
            ),
            0,
        )
        # Integrity check with the full expected task list fails (only 0.5 recorded).
        self.assertEqual(
            main(["phase", "--phase", "0", "--store", str(store),
                  "--tasks", "0.1,0.2,0.3,0.4,0.5"]),
            1,
        )
        # Without the expected list, the recorded tasks are all completed -> pass.
        self.assertEqual(main(["phase", "--phase", "0", "--store", str(store)]), 0)

    def test_cli_record_rejects_unverified_task(self):
        from tools.stage_completion_checker import main

        store = self._root / "records.json"
        repo, sha = self._make_repo(
            {"src/omnilss/checker.py": "x\n"}, SUBJECT_OK
        )  # no tests / no evidence
        self.assertEqual(
            main(["record", "--task", "0.5", "--commit", sha,
                  "--repo", str(repo), "--store", str(store)]),
            1,
        )
        data = json.loads(store.read_text(encoding="utf-8"))
        self.assertEqual(data["0.5"]["status"], STATUS_INCOMPLETE)


# ---------------------------------------------------------------------------
# Property tests (hypothesis, >= 100 examples each)
# ---------------------------------------------------------------------------
_HEX = "0123456789abcdef"
_FILE_POOL = (
    IMPL,
    TEST,
    EVID,
    "docs/reports/audit.md",
    "omnilss/tests/conftest.py",
    "benchmarks/results/raw/r1.json",
)


@st.composite
def _commit_scenario(draw):
    """Build (task_id, commit_id, subject, files) with independent facets."""
    valid = draw(st.booleans())
    has_chinese = draw(st.booleans())
    task_matches = draw(st.booleans())
    task_id = draw(st.from_regex(r"[1-9]\d*\.\d+", fullmatch=True))
    other_id = draw(
        st.from_regex(r"[1-9]\d*\.\d+", fullmatch=True).filter(lambda x: x != task_id)
    )
    if has_chinese and task_matches:
        subject = f"完成Phase任务{task_id}：实现与测试"
    elif has_chinese:
        subject = f"完成Phase任务{other_id}：实现与测试"
    else:
        subject = draw(
            st.text(
                alphabet=st.characters(min_codepoint=0x41, max_codepoint=0x7A),
                min_size=1,
                max_size=40,
            )
        )
    files = draw(st.lists(st.sampled_from(_FILE_POOL), unique=True))
    if valid:
        commit_id = draw(st.text(alphabet=_HEX, min_size=40, max_size=40))
    else:
        commit_id = draw(st.text(alphabet="0123456789", min_size=1, max_size=12))
    return task_id, commit_id, subject, tuple(files)


class TestPropertyValidation(unittest.TestCase):
    """Property: validation outcome equals the conjunction of prerequisites."""

    @settings(max_examples=100, deadline=None)
    @given(_commit_scenario())
    def test_property_validation_outcome_matches_prerequisites(self, scenario):
        if not _HYPOTHESIS_AVAILABLE:  # pragma: no cover
            self.skipTest("hypothesis not available")
        task_id, commit_id, subject, files = scenario
        inspector = FakeCommitInspector(
            {commit_id: _make_commit(commit_id, subject, list(files))}
        )
        result = validate_task_completion(task_id, commit_id, inspector)

        has_impl = any(is_implementation_file(f) for f in files)
        has_tests = any(is_test_file(f) for f in files)
        has_evidence = any(is_evidence_file(f) for f in files)
        expected = (
            valid_commit(commit_id, inspector)
            and contains_chinese(subject)
            and task_id in extract_task_ids(subject)
            and has_impl
            and has_tests
            and has_evidence
        )
        self.assertEqual(result.ok, expected)
        if result.ok:
            for name, passed in result.checks.items():
                self.assertTrue(passed, f"{name} should pass")
            self.assertEqual(result.reasons, [])
        else:
            failed = [name for name, passed in result.checks.items() if not passed]
            self.assertTrue(failed)
            self.assertTrue(result.reasons)

    @settings(max_examples=100)
    @given(st.from_regex(r"[1-9]\d*\.\d+", fullmatch=True))
    def test_property_task_marker_roundtrip(self, task_id):
        if not _HYPOTHESIS_AVAILABLE:  # pragma: no cover
            self.skipTest("hypothesis not available")
        self.assertIn(task_id, extract_task_ids(f"完成Phase任务{task_id}：实现"))

    @settings(max_examples=100)
    @given(
        st.text(
            alphabet=st.characters(min_codepoint=0x41, max_codepoint=0x7A),
            min_size=1,
            max_size=40,
        )
    )
    def test_property_ascii_subject_has_no_task_ids(self, subject):
        if not _HYPOTHESIS_AVAILABLE:  # pragma: no cover
            self.skipTest("hypothesis not available")
        self.assertEqual(extract_task_ids(subject), [])


def valid_commit(commit_id: str, inspector: FakeCommitInspector) -> bool:
    """True when the scenario's commit id resolves (mirror of the facet)."""
    return inspector.resolve(commit_id) is not None


if __name__ == "__main__":
    unittest.main()
