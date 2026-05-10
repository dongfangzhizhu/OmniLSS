"""R Bridge for consistency testing.

This module provides a bridge to call R gamlss functions from Python
for consistency testing purposes.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class RBridge:
    """Bridge to call R gamlss functions.
    
    This class manages the communication between Python and R for
    consistency testing. It uses subprocess to call R scripts and
    JSON for data serialization.
    
    Parameters
    ----------
    r_script_dir : str, optional
        Directory containing R scripts
    temp_dir : str, optional
        Directory for temporary files
    """
    
    def __init__(
        self,
        r_script_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
        rscript_path: Optional[str] = None,
    ):
        if r_script_dir is None:
            # Default to tests/rbus/R_scripts
            self.r_script_dir = Path(__file__).parent / "R_scripts"
        else:
            self.r_script_dir = Path(r_script_dir)
        
        if temp_dir is None:
            # Default to tests/rbus/temp
            self.temp_dir = Path(__file__).parent / "temp"
        else:
            self.temp_dir = Path(temp_dir)
        
        # Create temp directory if it doesn't exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.rscript_path = self._resolve_rscript(rscript_path)
        
        # Check if R is available
        self._check_r_available()

    @staticmethod
    def _resolve_rscript(explicit_path: Optional[str] = None) -> str:
        """Locate a usable Rscript executable."""
        candidates = [
            explicit_path,
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
        raise RuntimeError(
            "R is not available. Set RSCRIPT/R_SCRIPT or install R in a standard location."
        )
    
    def _check_r_available(self):
        """Check if R is available in the system."""
        try:
            subprocess.run(
                [self.rscript_path, "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            # R is available
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "R is not available. Please install R and ensure "
                "Rscript is discoverable."
            )
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other types to JSON serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def call_r_gamlss(
        self,
        data: Dict[str, Any],
        formula: str,
        family: str,
        sigma_formula: str = "~1",
        nu_formula: Optional[str] = None,
        tau_formula: Optional[str] = None,
        weights: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Call R gamlss function.
        
        Parameters
        ----------
        data : dict
            Data dictionary with variable names as keys
        formula : str
            Formula for mu parameter
        family : str
            Family name (e.g., "NO", "PO", "BI")
        sigma_formula : str, optional
            Formula for sigma parameter
        nu_formula : str, optional
            Formula for nu parameter
        tau_formula : str, optional
            Formula for tau parameter
        
        Returns
        -------
        result : dict
            Dictionary containing:
            - success : bool
            - coefficients : dict
            - fitted_values : dict
            - deviance : float
            - aic : float
            - etc.
        
        Raises
        ------
        RuntimeError
            If R script fails
        """
        # Prepare input data
        input_data = {
            "data": self._convert_to_json_serializable(data),
            "formula": formula,
            "family": family,
            "sigma_formula": sigma_formula,
        }
        
        if nu_formula is not None:
            input_data["nu_formula"] = nu_formula
        if tau_formula is not None:
            input_data["tau_formula"] = tau_formula
        if weights is not None:
            input_data["weights"] = self._convert_to_json_serializable(np.asarray(weights))
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            dir=self.temp_dir,
            delete=False
        ) as input_file:
            json.dump(input_data, input_file, indent=2)
            input_path = input_file.name
        
        output_path = input_path.replace('.json', '_output.json')
        
        try:
            # Call R script
            r_script = self.r_script_dir / "test_gamlss.R"
            
            result = subprocess.run(
                [self.rscript_path, str(r_script), input_path, output_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Read output
            with open(output_path, 'r') as f:
                output = json.load(f)
            
            if not output.get("success", False):
                raise RuntimeError(f"R script failed: {output.get('error', 'Unknown error')}")
            
            return output
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"R script failed with return code {e.returncode}\n"
                f"stdout: {e.stdout}\n"
                f"stderr: {e.stderr}"
            )
        finally:
            # Clean up temporary files
            try:
                os.unlink(input_path)
            except:
                pass
            try:
                os.unlink(output_path)
            except:
                pass
    
    def call_r_predict(self):
        """Call R predict function (to be implemented)."""
        # TODO: Implement prediction
        raise NotImplementedError("Prediction not yet implemented")


def compare_results(
    py_result: Dict[str, Any],
    r_result: Dict[str, Any],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Dict[str, bool]:
    """Compare Python and R results.
    
    Parameters
    ----------
    py_result : dict
        Python GAMLSS model results
    r_result : dict
        R GAMLSS model results
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    
    Returns
    -------
    comparison : dict
        Dictionary with comparison results for each component
    """
    comparison = {}
    
    # Compare coefficients
    for param in ["mu", "sigma", "nu", "tau"]:
        if param in py_result.get("coefficients", {}):
            py_coef = np.array(py_result["coefficients"][param])
            r_coef = np.array(r_result["coefficients"][param])
            
            try:
                np.testing.assert_allclose(py_coef, r_coef, rtol=rtol, atol=atol)
                comparison[f"coefficients_{param}"] = True
            except AssertionError:
                comparison[f"coefficients_{param}"] = False
    
    # Compare fitted values
    for param in ["mu", "sigma", "nu", "tau"]:
        if param in py_result.get("fitted_values", {}):
            py_fitted = np.array(py_result["fitted_values"][param])
            r_fitted = np.array(r_result["fitted_values"][param])
            
            try:
                np.testing.assert_allclose(py_fitted, r_fitted, rtol=rtol, atol=atol)
                comparison[f"fitted_values_{param}"] = True
            except AssertionError:
                comparison[f"fitted_values_{param}"] = False
    
    # Compare statistics
    for stat in ["deviance", "aic", "df_fit"]:
        if stat in py_result and stat in r_result:
            try:
                np.testing.assert_allclose(
                    py_result[stat],
                    r_result[stat],
                    rtol=rtol * 10,  # More lenient for statistics
                    atol=atol * 10
                )
                comparison[stat] = True
            except AssertionError:
                comparison[stat] = False
    
    return comparison
