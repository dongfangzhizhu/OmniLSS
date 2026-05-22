from pathlib import Path


def test_changelog_contains_v0_3_0_release_section():
    repo_root = Path(__file__).resolve().parents[2]
    changelog = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    changelog_cn = (repo_root / "CHANGELOG_cn.md").read_text(encoding="utf-8")

    assert "[0.3.0]" in changelog
    assert "2026-05-17" in changelog
    assert "Validation" in changelog
    assert "Known limitations" in changelog

    assert "[0.3.0]" in changelog_cn
    assert "2026-05-17" in changelog_cn
    assert "验证" in changelog_cn
    assert "已知限制" in changelog_cn
