import subprocess
import sys


def test_jax_x64_default_without_omnilss_import():
    code = "import jax; print(jax.config.read('jax_enable_x64'))"
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out in {"False", "0"}
