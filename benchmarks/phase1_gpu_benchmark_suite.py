"""Phase 1 GPU benchmark suite scaffold.

Defines standard benchmark categories and metric collection schema.
"""

from __future__ import annotations

CATEGORIES = ("small", "medium", "large", "pathological")
METRICS = ("runtime", "memory", "compile_time", "convergence")


def suite_spec():
    return {"categories": CATEGORIES, "metrics": METRICS}


if __name__ == "__main__":
    print(suite_spec())
