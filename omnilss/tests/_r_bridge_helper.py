"""Helper for importing RBridge in both discover and direct-run modes."""
import os
import shutil
import subprocess
from pathlib import Path


def _resolve_rscript() -> str | None:
    """Locate Rscript without requiring the current shell PATH to be configured."""
    candidates = [
        os.environ.get("RSCRIPT"),
        os.environ.get("R_SCRIPT"),
        shutil.which("Rscript"),
        r"C:\Program Files\R\R-4.6.0\bin\Rscript.exe",
        r"C:\Program Files\R\R-4.6.0\bin\x64\Rscript.exe",
        r"C:\Program Files\R\R-4.5.3\bin\Rscript.exe",
        r"C:\Program Files\R\R-4.5.3\bin\x64\Rscript.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def _r_available() -> bool:
    rscript = _resolve_rscript()
    if rscript is None:
        return False
    try:
        subprocess.run(
            [rscript, "--version"],
            capture_output=True, check=True, timeout=5,
        )
        return True
    except Exception:
        return False


def get_r_unavailable_reason() -> str:
    """Return explicit reason when R bridge tests are not runnable."""
    if R_BRIDGE_CLS is None:
        return "RBridge import unavailable"
    rscript = _resolve_rscript()
    if rscript is None:
        return "Rscript executable not found"
    if not _r_available():
        return f"Rscript exists but not runnable: {rscript}"
    return ""


def get_r_bridge_class():
    """Return RBridge class, or None if not importable."""
    try:
        from tests.rbus.r_bridge import RBridge
        return RBridge
    except ImportError:
        pass
    try:
        # When tests/ is in sys.path (direct run)
        from rbus.r_bridge import RBridge
        return RBridge
    except ImportError:
        pass
    # Last resort: absolute path
    import sys
    tests_dir = Path(__file__).parent
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    try:
        from rbus.r_bridge import RBridge
        return RBridge
    except ImportError:
        return None


R_BRIDGE_CLS = get_r_bridge_class()
R_AVAILABLE = R_BRIDGE_CLS is not None and _r_available()
R_UNAVAILABLE_REASON = get_r_unavailable_reason()
