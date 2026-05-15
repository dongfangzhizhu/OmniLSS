"""R/Python Consistency Testing Bus.

This module provides the `RTestBus` class which communicates with a local `Rscript` 
process to invoke `gamlss` implementations.
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)

class RTestBus:
    """Invokes R scripts externally to validate JAX computational consistency."""

    def __init__(self, scripts_dir: str | Path | None = None):
        if scripts_dir is None:
            # Default to the tests/rbus/R_scripts directory
            current_dir = Path(__file__).parent
            self.scripts_dir = current_dir / "R_scripts"
        else:
            self.scripts_dir = Path(scripts_dir)
        
        # Verify Rscript is available. General CI does not install R, so callers
        # should be able to skip R-backed tests at setup time instead of failing
        # later during the first subprocess call.
        try:
            subprocess.run(["Rscript", "--version"], check=True, capture_output=True)
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            logger.warning("Rscript not found in PATH or failed to execute.")
            raise RuntimeError("Rscript not available") from exc

    def _run_script(self, script_name: str, **kwargs) -> Dict[str, Any]:
        """Runs an R script with a JSON payload of arguments via a temporary file."""
        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"R script {script_name} not found in {self.scripts_dir}")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as arg_file:
            json.dump(kwargs, arg_file)
            arg_file_path = arg_file.name

        try:
            result = subprocess.run(
                ["Rscript", str(script_path), arg_file_path],
                capture_output=True,
                text=True,
                check=True
            )
            # The R script is expected to print the JSON result to stdout.
            # R might output sum warnings, so we parse the last valid JSON block.
            output_lines = result.stdout.strip().split("\n")
            json_str = next((line for line in reversed(output_lines) if line.startswith("{") or line.startswith("[")), None)
            
            if json_str is None:
                raise ValueError(f"Could not find JSON output in R script stdout: {result.stdout}")
            
            return json.loads(json_str)

        except subprocess.CalledProcessError as e:
            logger.error(f"R script failed with exit code {e.returncode}")
            logger.error(f"Stdout:\n{e.stdout}")
            logger.error(f"Stderr:\n{e.stderr}")
            raise
        finally:
            os.unlink(arg_file_path)

    def eval_family(self, family: str, func_type: str, args: Dict[str, Any]) -> np.ndarray:
        """Evaluates a family's d, p, q functions.
        
        Args:
            family: e.g. "NO", "PO", "TF"
            func_type: "d", "p", "q"
            args: dictionary of arguments (e.g. {"x": [...], "mu": [...], "sigma": [...]})
        """
        result = self._run_script("eval_family.R", family=family, type=func_type, args=args)
        if "error" in result:
            raise RuntimeError(f"R Error: {result['error']}")
        return np.array(result["values"], dtype=np.float64)
