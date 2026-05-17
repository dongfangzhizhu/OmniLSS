"""Minimal HTTP API boundary for OmniLSS service metadata."""

from .server import create_handler, serve

__all__ = ["create_handler", "serve"]
