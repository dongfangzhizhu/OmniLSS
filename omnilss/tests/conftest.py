from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# D2.309: pytest-xdist workers share one process-wide model registry path by
# default, so concurrent gRPC tests hit SQLite "database is locked" even with
# a busy timeout (journal-mode races and read-to-write upgrades).  Give each
# worker its own registry file before omnilss.api.grpc.server is imported.
_WORKER = os.environ.get("PYTEST_XDIST_WORKER")
if _WORKER and "OMNILSS_MODEL_DB_PATH" not in os.environ:
    os.environ["OMNILSS_MODEL_DB_PATH"] = str(
        Path(tempfile.gettempdir()) / f"omnilss-registry-{_WORKER}.sqlite3"
    )


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
