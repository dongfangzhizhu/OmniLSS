"""Likelihood namespace placeholder for architecture-freeze migration.

This package intentionally remains minimal for now: runtime likelihood logic is
still implemented in existing fitting/algorithm modules. New core-stable
likelihood helpers should be added here first, then imported by higher-level
modules to avoid introducing additional cross-layer coupling.
"""

__all__: list[str] = []
