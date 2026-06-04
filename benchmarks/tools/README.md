# Benchmarks Tools

This directory contains diagnostic and testing tools for OmniLSS development.

## Diagnostic Tools

### diagnose_smoother_differences.py

Diagnostic tool to compare Python smoother implementations (pb, ps, cs) with R gamlss.

**Usage**:
```bash
python diagnose_smoother_differences.py
```

**Output**:
- Detailed comparison of Python vs R smoothers
- Deviance differences
- Coefficient counts
- Fitted value correlations
- Error metrics (max absolute, mean absolute, RMSE)

## Test Generation Tools

### generate_all_r_tests.py

Unified test generator for R consistency tests across all family distributions.

**Usage**:
```bash
python generate_all_r_tests.py
```

**Features**:
- Generates tests for 1-4 parameter families
- Uses parameter-aware framework
- Reads parameter ranges from `family_param_ranges.py`

### generate_r_consistency_tests.py

Generate R consistency tests for specific families.

### generate_correct_r_tests.py

Generate corrected R consistency tests with proper parameter handling.

## Configuration

### family_param_ranges.py

Parameter range configuration for all family distributions.

**Format**:
```python
FAMILY_PARAM_RANGES = {
    "NO": {
        "n_params": 2,
        "param_names": ["mu", "sigma"],
        "ranges": {
            "mu": (-5.0, 5.0),
            "sigma": (0.1, 2.0)
        },
        "link_names": ["identity", "log"]
    },
    # ...
}
```

## Utility Tools

### check_missing_r_tests.py

Check which families are missing R consistency tests.

**Usage**:
```bash
python check_missing_r_tests.py
```

### inspect_family_params.py

Inspect and validate family parameter configurations.

**Usage**:
```bash
python inspect_family_params.py
```

### run_new_tests.py

Run newly generated R consistency tests.

**Usage**:
```bash
python run_new_tests.py
```

## Integration with Main Benchmarks

These tools support the main benchmark scripts:
- `../comprehensive_r_consistency_test.py` - Full R consistency testing
- `../comprehensive_performance_test.py` - Performance benchmarking

## Development Workflow

1. **Diagnose Issues**:
   ```bash
   python tools/diagnose_smoother_differences.py
   ```

2. **Generate Missing Tests**:
   ```bash
   python tools/generate_all_r_tests.py
   ```

3. **Run Full Consistency Tests**:
   ```bash
   python comprehensive_r_consistency_test.py --quick
   ```

4. **Check Coverage**:
   ```bash
   python tools/check_missing_r_tests.py
   ```

## Notes

- All tools require R and gamlss package to be installed
- Set `JAX_ENABLE_X64=true` for proper numerical precision
- Tools are designed to work from the project root directory
