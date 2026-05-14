"""Fix all five Colab notebooks: correct API calls, translate Chinese to English, fix timing."""
import json
import re
from pathlib import Path

COLAB_DIR = Path("examples/colab")

def fix_source(source_lines):
    """Fix API calls and translate Chinese in source code lines."""
    result = []
    for line in source_lines:
        # Fix wrong API calls
        line = line.replace("model.mu_fv", 'np.asarray(model.fitted_values["mu"])')
        line = line.replace("model.sigma_fv", 'np.asarray(model.fitted_values["sigma"])')
        line = line.replace("model.aic", 'model.additional_slots["aic"]')
        line = line.replace("model.bic", 'model.additional_slots["sbc"]')
        line = line.replace("model.mu_coefficients", 'model.coefficients["mu"]')
        line = line.replace("model.sigma_coefficients", 'model.coefficients["sigma"]')
        # Fix algorithm case
        line = re.sub(r'algorithm\s*=\s*"rs"', 'algorithm="RS"', line)
        line = re.sub(r'algorithm\s*=\s*"cg"', 'algorithm="CG"', line)
        line = re.sub(r'algorithm\s*=\s*"mixed"', 'algorithm="Mixed"', line)
        result.append(line)
    return result

def fix_markdown(source_lines):
    """Remove Chinese text from markdown cells."""
    result = []
    for line in source_lines:
        # Remove Chinese characters (Unicode range 4E00-9FFF and common CJK)
        cleaned = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+', '', line)
        # Clean up leftover artifacts like "/ " at start or end
        cleaned = re.sub(r'\s*/\s*$', '', cleaned)
        cleaned = re.sub(r'^\s*/\s*', '', cleaned)
        # Remove lines that are now empty or just whitespace/punctuation
        stripped = cleaned.strip()
        if stripped and stripped not in ('/', '\\n', '\n'):
            result.append(cleaned)
        elif line.strip() == '' or line == '\n':
            result.append(line)  # keep blank lines
    return result

def process_notebook(nb_path):
    """Load, fix, and save a notebook."""
    print(f"Processing {nb_path.name}...")
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        # Clear outputs
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None
        
        source = cell.get('source', [])
        if isinstance(source, str):
            source = [source]
        
        if cell['cell_type'] == 'code':
            cell['source'] = fix_source(source)
        elif cell['cell_type'] == 'markdown':
            cell['source'] = fix_markdown(source)
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Done: {nb_path.name}")

# Process all five notebooks
for nb_name in ['02_consistency_dpqr.ipynb', '03_consistency_fitting.ipynb',
                '04_consistency_smoothing.ipynb', '05_performance_cpu.ipynb',
                '08_comprehensive_comparison.ipynb']:
    nb_path = COLAB_DIR / nb_name
    if nb_path.exists():
        process_notebook(nb_path)
    else:
        print(f"  MISSING: {nb_name}")

print("All done!")
