"""Public API adapters kept separate from core numerical infrastructure."""

from .http import serve as serve_http

__all__ = ["serve_http"]
