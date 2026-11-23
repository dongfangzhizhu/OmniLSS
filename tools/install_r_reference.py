"""Install the pinned R reference packages for Group B (D2.308/R1).

Reads docs/development/r-reference-lock.json, downloads each tarball, verifies
its SHA256, retries the optional mirror_url on failure, and prints the local
paths in lock order so the caller can hand them to Rscript install.packages().
No version or URL is hard-coded here or in CI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = REPO_ROOT / "docs" / "development" / "r-reference-lock.json"
REQUIRED_FIELDS = ("name", "version", "url", "sha256")


def load_lock(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"unsupported lock schema_version: {data.get('schema_version')!r}")
    packages = data.get("packages")
    if not packages:
        raise ValueError("lock has no packages")
    for package in packages:
        missing = [key for key in REQUIRED_FIELDS if not package.get(key)]
        if missing:
            raise ValueError(f"package entry missing {missing}: {package!r}")
        if len(str(package["sha256"])) != 64:
            raise ValueError(f"invalid sha256 for {package['name']}: {package['sha256']!r}")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "omnilss-r-reference/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with Path(dest).open("wb") as handle:
            shutil.copyfileobj(response, handle)


def fetch_package(package: dict, dest_dir: Path, download=_download) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = package["url"].rstrip("/").rsplit("/", 1)[-1]
    dest = dest_dir / filename
    errors: list[str] = []
    for url in (package["url"], package.get("mirror_url")):
        if not url:
            continue
        partial = dest.with_name(dest.name + ".part")
        try:
            download(url, partial)
            actual = sha256_file(partial)
            if actual != package["sha256"]:
                errors.append(f"{url}: sha256 {actual} != {package['sha256']}")
                partial.unlink(missing_ok=True)
                continue
            partial.replace(dest)
            return dest
        except Exception as exc:  # network/HTTP/filesystem
            errors.append(f"{url}: {exc}")
            partial.unlink(missing_ok=True)
    raise RuntimeError(
        f"cannot fetch {package['name']} {package['version']}: " + "; ".join(errors)
    )


def materialize(lock: dict, dest_dir: Path, download=_download) -> list[Path]:
    return [fetch_package(package, dest_dir, download=download) for package in lock["packages"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    dest = args.dest or Path(tempfile.mkdtemp(prefix="r-reference-"))
    try:
        paths = materialize(load_lock(args.lock), dest)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not args.quiet:
        for path in paths:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
