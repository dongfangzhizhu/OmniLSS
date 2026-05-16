"""gRPC boundary for OmniLSS.

Contains protobuf definitions under `proto/` and runtime server wiring in
`server.py`. Closed-source clients should call this boundary only.
"""

from .server import serve

__all__ = ["serve"]
