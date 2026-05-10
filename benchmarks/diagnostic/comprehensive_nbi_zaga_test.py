"""Comprehensive test for NBI and ZAGA distributions after RS algorithm fix."""

import numpy as np
import pandas as pd
import sys
import time
sys.path.insert(0, 'omnilss/src')

from omnilss import gamlss

def test_distribution(family_name, n_values, formulas, seed=42):
    """Test a distribution with different sample sizes and formulas."""
    
    results = []
    
    for n in n_values:
        for formula_pair in formulas:
            mu_formula = formula_pair['mu']
            sigma_formula = formula_pair.get('sigma', '~1')
            
            print(f"\n{'='*70}")
            print(f"Testing {family_name}: n={n}, formula={mu_formula}")
            print(f"{'='*70}")
            
            # Generate data
            np.random.seed(seed)
            x1 = np.random.normal(0, 1, n)
            x2 = np.random.normal(0, 1, n)
            
            if family_name == "NBI":
                # Generate NBI data
                mu_true = np.exp(1 + 0.5 * x1)
                sigma_true = 0.5
                y = np.random.negative_binomial(
                    n=1/sigma_true**2, 
                    p=1/(1 + mu_true * sigma_true**2), 
                    size=n
                )
            elif family_name == "ZAGA":
                # Generate ZAGA data (zero-adjusted gamma)
                # First generate zero/non-zero indicator
                nu_true = 0.3  # probability of zero
                is_zero = np.random.binomial(1, nu_true, n)
                
                # For non-zero values, generate from gamma
                mu_true = np.exp(1 + 0.5 * x1)
                sigma_true = 0.5
                shape = 1 / sigma_true**2
                scale = mu_true * sigma_true**2
                
                y = np.zeros(n)
                y[is_zero == 0] = np.random.gamma(shape, scale[is_zero == 0])
            
            data = {
                'y': y,
                'x1': x1,
                'x2': x2
            }
            
            # Fit model
            start_time = time.time()
            try:
                model = gamlss(
                    formula=mu_formula,
                    sigma_formula=sigma_formula,
                    family=family_name,
                    data=data,
                    method="RS",
                    verbose=False
                )
                
                fit_time = time.time() - start_time
                
                result = {
                    'family': family_name,
                    'n': n,
                    'formula': mu_formula,
                    'deviance': model.g_dev,
                    'iterations': model.iter,
                    'converged': model.additional_slots.get('converged', False),
                    'fit_time': fit_time,
                    'mu_coef': model.coefficients['mu'],
                    'sigma': np.exp(model.coefficients['sigma'][0]) if 'sigma' in model.coefficients else None,
                    'status': 'SUCCESS'
                }
                
                print(f"✓ Converged in {model.iter} iterations")
                print(f"  Deviance: {model.g_dev:.6f}")
                print(f"  Fit time: {fit_time:.3f}s")
                if result['sigma'] is not None:
                    print(f"  Sigma: {result['sigma']:.6f}")
                
            except Exception as e:
                result = {
                    'family': family_name,
                    'n': n,
                    'formula': mu_formula,
                    'deviance': None,
                    'iterations': None,
                    'converged': False,
                    'fit_time': None,
                    'mu_coef': None,
                    'sigma': None,
                    'status': f'ERROR: {str(e)}'
                }
                print(f"✗ Error: {str(e)}")
            
            results.append(result)
    
    return results

def compare_with_r(python_results, r_results_file):
    """Compare Python results with R results."""
    
    try:
        r_df = pd.read_csv(r_results_file)
        
        print(f"\n{'='*70}")
        print("Comparison with R Results")
        print(f"{'='*70}")
        print(f"{'Family':<10} {'n':<6} {'Formula':<15} {'Dev Diff':<12} {'Status':<10}")
        print("-" * 70)
        
        for py_result in python_results:
            # Find matching R result
            r_match = r_df[
                (r_df['family'] == py_result['family']) &
                (r_df['n'] == py_result['n']) &
                (r_df['formula'] == py_result['formula'])
            ]
            
            if len(r_match) > 0 and py_result['deviance'] is not None:
                r_dev = r_match.iloc[0]['deviance']
                dev_diff = abs(py_result['deviance'] - r_dev)
                
                # Check if difference is acceptable (< 0.01)
                status = "✓ PASS" if dev_diff < 0.01 else "✗ FAIL"
                
                print(f"{py_result['family']:<10} {py_result['n']:<6} {py_result['formula']:<15} {dev_diff:<12.6f} {status:<10}")
            else:
                print(f"{py_result['family']:<10} {py_result['n']:<6} {py_result['formula']:<15} {'N/A':<12} {'NO R DATA':<10}")
        
        print("=" * 70)
        
    except FileNotFoundError:
        print(f"\nWarning: R results file not found: {r_results_file}")
        print("Skipping comparison with R.")

def main():
    print("=" * 70)
    print("Comprehensive NBI and ZAGA Test After RS Algorithm Fix")
    print("=" * 70)
    
    # Test configurations
    n_values = [100, 500, 5000]
    formulas = [
        {'mu': 'y ~ 1'},
        {'mu': 'y ~ x1'},
        {'mu': 'y ~ x1 + x2'}
    ]
    
    all_results = []
    
    # Test NBI
    print("\n" + "=" * 70)
    print("Testing NBI Distribution")
    print("=" * 70)
    nbi_results = test_distribution("NBI", n_values, formulas, seed=42)
    all_results.extend(nbi_results)
    
    # Test ZAGA
    print("\n" + "=" * 70)
    print("Testing ZAGA Distribution")
    print("=" * 70)
    zaga_results = test_distribution("ZAGA", n_values, formulas, seed=42)
    all_results.extend(zaga_results)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('scripts/diagnostic/python_nbi_zaga_results.csv', index=False)
    print(f"\n✓ Results saved to scripts/diagnostic/python_nbi_zaga_results.csv")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    for family in ['NBI', 'ZAGA']:
        family_results = [r for r in all_results if r['family'] == family]
        success_count = sum(1 for r in family_results if r['status'] == 'SUCCESS')
        total_count = len(family_results)
        
        print(f"\n{family}:")
        print(f"  Success: {success_count}/{total_count}")
        
        if success_count > 0:
            avg_iterations = np.mean([r['iterations'] for r in family_results if r['iterations'] is not None])
            avg_time = np.mean([r['fit_time'] for r in family_results if r['fit_time'] is not None])
            print(f"  Avg iterations: {avg_iterations:.1f}")
            print(f"  Avg fit time: {avg_time:.3f}s")
    
    # Try to compare with R results if available
    compare_with_r(all_results, 'scripts/diagnostic/r_nbi_zaga_results.csv')
    
    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)

if __name__ == "__main__":
    main()
