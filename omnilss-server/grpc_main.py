"""OmniLSS gRPC server entry point.

Usage:
    python grpc_main.py [--host 0.0.0.0] [--port 50051]
"""

from __future__ import annotations

import argparse
import signal


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniLSS gRPC Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()

    from omnilss.api.grpc.server import serve

    server = serve(host=args.host, port=args.port)
    print(f"OmniLSS gRPC server listening on {args.host}:{args.port}", flush=True)

    def _shutdown(signum, frame):  # noqa: ANN001
        print("Shutting down gRPC server...", flush=True)
        server.stop(grace=5)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
