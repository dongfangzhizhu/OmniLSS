"""Test Python with the same data as R."""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'omnilss/src')

from omnilss import gamlss

# Load R reference results
r_results = pd.read_csv('scripts/diagnostic/r_reference_results.csv')

print("=" * 70)
print("Testing Python with R-generated Data")
print("=" * 70)

python_results = []

for idx, r_row in r_results.iterrows():
    family_name = r_row['family']
    n = int(r_row['n'])
    formula_str = r_row['formula']
    seed = int(r_row['seed'])
    data_file = r_row['data_file']
    
    print(f"\n{'='*70}")
    print(f"Test {idx+1}/{len(r_results)}: {family_name}, n={n}, formula={formula_str}")
    print(f"{'='*70}")
    
    # Load data
    df = pd.read_csv(data_file)
    data = {
        'y': df['y'].values,
        'x1': df['x1'].values,
        'x2': df['x2'].values
    }
    
    # Fit model
    try:
        model = gamlss(
            formula=formula_str,
            sigma_formula="~1",
            family=family_name,
            data=data,
            method="RS",
            verbose=False
        )
        
        py_deviance = model.g_dev
        py_iterations = model.iter
        py_converged = model.additional_slots.get('converged', False)
        py_sigma = np.exp(model.coefficients['sigma'][0])
        py_mu_coefs = model.coefficients['mu']
        
        # Compare with R
        r_deviance = r_row['deviance']
        r_sigma = r_row['sigma']
        
        dev_diff = abs(py_deviance - r_deviance)
        sigma_diff = abs(py_sigma - r_sigma)
        sigma_rel_diff = sigma_diff / r_sigma * 100
        
        status = "✓ PASS" if dev_diff < 0.01 else "✗ FAIL"
        
        print(f"Python:")
        print(f"  Deviance: {py_deviance:.6f}")
        print(f"  Iterations: {py_iterations}")
        print(f"  Sigma: {py_sigma:.6f}")
        print(f"  Mu coefficients: {py_mu_coefs}")
        print(f"\nR:")
        print(f"  Deviance: {r_deviance:.6f}")
        print(f"  Iterations: {r_row['iterations']}")
        print(f"  Sigma: {r_sigma:.6f}")
        print(f"\nDifference:")
        print(f"  Deviance diff: {dev_diff:.6f} {status}")
        print(f"  Sigma diff: {sigma_diff:.6f} ({sigma_rel_diff:.2f}%)")
        
        python_results.append({
            'family': family_name,
            'n': n,
            'formula': formula_str,
            'seed': seed,
            'py_deviance': py_deviance,
            'r_deviance': r_deviance,
            'dev_diff': dev_diff,
            'py_sigma': py_sigma,
            'r_sigma': r_sigma,
            'sigma_diff': sigma_diff,
            'sigma_rel_diff': sigma_rel_diff,
            'py_iterations': py_iterations,
            'r_iterations': r_row['iterations'],
            'status': 'PASS' if dev_diff < 0.01 else 'FAIL'
        })
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        python_results.append({
            'family': family_name,
            'n': n,
            'formula': formula_str,
            'seed': seed,
            'py_deviance': None,
            'r_deviance': r_row['deviance'],
            'dev_diff': None,
            'py_sigma': None,
            'r_sigma': r_row['sigma'],
            'sigma_diff': None,
            'sigma_rel_diff': None,
            'py_iterations': None,
            'r_iterations': r_row['iterations'],
            'status': 'ERROR'
        })

# Save results
results_df = pd.DataFrame(python_results)
results_df.to_csv('scripts/diagnostic/python_vs_r_comparison.csv', index=False)
print(f"\n✓ Results saved to scripts/diagnostic/python_vs_r_comparison.csv")

# Summary
print(f"\n{'='*70}")
print("Summary")
print(f"{'='*70}")

for family in ['NBI', 'ZAGA']:
    family_results = [r for r in python_results if r['family'] == family]
    
    if len(family_results) > 0:
        pass_count = sum(1 for r in family_results if r['status'] == 'PASS')
        fail_count = sum(1 for r in family_results if r['status'] == 'FAIL')
        error_count = sum(1 for r in family_results if r['status'] == 'ERROR')
        
        print(f"\n{family}:")
        print(f"  Total tests: {len(family_results)}")
        print(f"  Passed: {pass_count}")
        print(f"  Failed: {fail_count}")
        print(f"  Errors: {error_count}")
        
        if pass_count + fail_count > 0:
            valid_results = [r for r in family_results if r['dev_diff'] is not None]
            if len(valid_results) > 0:
                avg_dev_diff = np.mean([r['dev_diff'] for r in valid_results])
                max_dev_diff = np.max([r['dev_diff'] for r in valid_results])
                avg_sigma_rel_diff = np.mean([r['sigma_rel_diff'] for r in valid_results])
                
                print(f"  Avg deviance diff: {avg_dev_diff:.6f}")
                print(f"  Max deviance diff: {max_dev_diff:.6f}")
                print(f"  Avg sigma rel diff: {avg_sigma_rel_diff:.2f}%")

print(f"\n{'='*70}")
print("Detailed Results")
print(f"{'='*70}")
print(f"{'Family':<8} {'n':<6} {'Formula':<15} {'Dev Diff':<12} {'Sigma %':<10} {'Status':<8}")
print("-" * 70)

for r in python_results:
    sigma_pct = f"{r['sigma_rel_diff']:.2f}%" if r['sigma_rel_diff'] is not None else "N/A"
    dev_diff_str = f"{r['dev_diff']:.6f}" if r['dev_diff'] is not None else "N/A"
    print(f"{r['family']:<8} {r['n']:<6} {r['formula']:<15} {dev_diff_str:<12} {sigma_pct:<10} {r['status']:<8}")

print("=" * 70)
