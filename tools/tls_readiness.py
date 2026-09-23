"""Fail-closed TLS certificate/key readiness check for deployments."""

from __future__ import annotations

import argparse
import datetime as dt
import ssl
from pathlib import Path


def validate(cert: Path, key: Path, min_days: int = 0) -> dict[str, object]:
    issues: list[str] = []
    if not cert.is_file():
        issues.append("certificate file does not exist")
    if not key.is_file():
        issues.append("private key file does not exist")
    if issues:
        return {"ok": False, "issues": issues}
    cert_bytes = cert.read_bytes()
    key_bytes = key.read_bytes()
    if b"BEGIN CERTIFICATE" not in cert_bytes:
        issues.append("certificate is not PEM encoded")
    if b"PRIVATE KEY" not in key_bytes:
        issues.append("private key is not PEM encoded")
    try:
        decoded = ssl._ssl._test_decode_cert(str(cert))
        expires = dt.datetime.strptime(decoded["notAfter"], "%b %d %H:%M:%S %Y %Z").replace(tzinfo=dt.timezone.utc)
        remaining_days = (expires - dt.datetime.now(dt.timezone.utc)).total_seconds() / 86400
        if remaining_days < min_days:
            issues.append(f"certificate expires in {remaining_days:.1f} days")
    except Exception as exc:
        expires = None
        remaining_days = None
        issues.append(f"certificate could not be decoded: {exc}")
    return {"ok": not issues, "issues": issues, "certificate": str(cert), "expires_at": expires.isoformat() if expires else None, "remaining_days": remaining_days}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cert", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--min-days", type=int, default=7)
    args = parser.parse_args()
    report = validate(args.cert, args.key, args.min_days)
    print(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
