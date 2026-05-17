"""Generate OmniLSS gRPC Python stubs from proto files.

Usage
-----
python tools/generate_grpc_stubs.py
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROTO_DIR = ROOT / "src" / "omnilss" / "api" / "grpc" / "proto"
OUT_DIR = ROOT / "src" / "omnilss" / "api" / "grpc" / "generated"
PROTOS = ["fit.proto", "predict.proto", "sample.proto"]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    init_file = OUT_DIR / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Generated protobuf modules for OmniLSS gRPC API."""\n')

    try:
        import grpc_tools.protoc  # noqa: F401
    except Exception:
        print(
            "grpcio-tools is required. Install with: pip install 'omnilss[grpc]'",
            file=sys.stderr,
        )
        return 2

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        "-I",
        str(PROTO_DIR),
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        *[str(PROTO_DIR / p) for p in PROTOS],
    ]
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
