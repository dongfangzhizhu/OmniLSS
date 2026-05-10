# Contributing to OmniLSS

Thank you for your interest in contributing to OmniLSS! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

---

## 📜 Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- UV package manager (recommended) or pip
- Git
- Basic knowledge of JAX and GAMLSS

### Areas for Contribution

We welcome contributions in the following areas:

- 🐛 **Bug fixes**: Fix issues reported in GitHub Issues
- ✨ **New features**: Implement new distributions, algorithms, or functionality
- 📝 **Documentation**: Improve docs, tutorials, or examples
- 🧪 **Tests**: Add or improve test coverage
- 🎨 **Examples**: Create new example scripts or notebooks
- 🌍 **Translations**: Translate documentation to other languages
- 🔧 **Performance**: Optimize existing code

---

## 💻 Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/omnilss.git
cd omnilss
```

### 2. Set Up Environment

#### Using UV (Recommended)

```bash
# UV will automatically create and manage the virtual environment
cd omnilss
uv pip install -e ".[dev]"
```

#### Using pip

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
# Run tests to verify setup
cd omnilss
python -m pytest tests/ -v

# Check that you can import omnilss
python -c "import omnilss; print(omnilss.__version__)"
```

### 4. Set Up Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

---

## 🤝 How to Contribute

### Reporting Bugs

Before creating a bug report:
1. Check if the bug has already been reported in [Issues](https://github.com/your-org/omnilss/issues)
2. Verify the bug exists in the latest version
3. Collect relevant information (Python version, JAX version, error messages)

When creating a bug report, include:
- Clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Code snippet demonstrating the issue
- Environment details (OS, Python version, package versions)

### Suggesting Features

Before suggesting a feature:
1. Check if it has already been suggested
2. Consider if it fits the project scope
3. Think about how it would benefit users

When suggesting a feature, include:
- Clear description of the feature
- Use cases and examples
- Potential implementation approach
- Any relevant references (papers, R GAMLSS implementation)

### Contributing Code

1. **Find or create an issue**: Discuss your proposed changes first
2. **Create a branch**: Use a descriptive name (e.g., `fix-normal-distribution`, `add-beta-family`)
3. **Make changes**: Follow coding standards and write tests
4. **Test thoroughly**: Ensure all tests pass
5. **Update documentation**: Document new features or changes
6. **Submit a pull request**: Follow the PR template

---

## 📏 Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings (except when single quotes avoid escaping)
- **Imports**: Organized in three groups (stdlib, third-party, local)

### Code Formatting

We use the following tools:

```bash
# Format code with black
black omnilss/src/omnilss/

# Check code style with ruff
ruff check omnilss/src/omnilss/

# Type checking with mypy (optional)
mypy omnilss/src/omnilss/
```

### Naming Conventions

- **Functions/methods**: `snake_case` (e.g., `compute_deviance`)
- **Classes**: `PascalCase` (e.g., `GAMLSSModel`, `NormalFamily`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TOLERANCE`)
- **Private members**: Prefix with `_` (e.g., `_internal_helper`)

### Documentation Strings

Use Google-style docstrings:

```python
def fit_model(formula: str, data: dict, family: str = "NO") -> GAMLSSModel:
    """Fit a GAMLSS model to data.
    
    Args:
        formula: Model formula in R-style syntax (e.g., "y ~ x1 + x2")
        data: Dictionary containing the data
        family: Distribution family name (default: "NO" for Normal)
    
    Returns:
        Fitted GAMLSSModel object
    
    Raises:
        ValueError: If formula is invalid or data is missing required variables
    
    Examples:
        >>> data = {"y": [1, 2, 3], "x": [0, 1, 2]}
        >>> model = fit_model("y ~ x", data)
        >>> print(model.g_dev)
    """
    pass
```

### Type Hints

Use type hints for function signatures:

```python
from typing import Optional, Dict, List
import jax.numpy as jnp

def predict(
    model: GAMLSSModel,
    newdata: Dict[str, jnp.ndarray],
    type: str = "response"
) -> jnp.ndarray:
    """Predict from fitted model."""
    pass
```

---

## 🧪 Testing Guidelines

### Test Structure

Tests are organized in `omnilss/tests/`:

```
tests/
├── test_distributions.py      # Distribution family tests
├── test_algorithms.py          # Algorithm tests
├── test_smoothers.py           # Smoother tests
├── test_r_consistency.py       # R GAMLSS consistency tests
└── test_integration.py         # Integration tests
```

### Writing Tests

Use pytest for testing:

```python
import pytest
import jax.numpy as jnp
from omnilss import gamlss, NO

def test_normal_distribution_fit():
    """Test fitting a simple Normal distribution model."""
    # Arrange
    data = {
        "y": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    }
    
    # Act
    model = gamlss("y ~ x", family=NO(), data=data)
    
    # Assert
    assert model.converged
    assert model.g_dev > 0
    assert len(model.mu) == 5

def test_invalid_formula_raises_error():
    """Test that invalid formula raises ValueError."""
    data = {"y": jnp.array([1, 2, 3])}
    
    with pytest.raises(ValueError, match="Invalid formula"):
        gamlss("y ~ nonexistent_var", family=NO(), data=data)
```

### Test Coverage

- Aim for **>95% code coverage**
- Test both success and failure cases
- Test edge cases and boundary conditions
- Include numerical accuracy tests (compare with R GAMLSS)

### Running Tests

```bash
# Run all tests
cd omnilss
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_distributions.py -v

# Run with coverage
python -m pytest tests/ --cov=omnilss --cov-report=html

# Run only fast tests (skip slow performance tests)
python -m pytest tests/ -m "not slow"
```

### R Consistency Tests

When adding new distributions or algorithms, include R consistency tests:

```python
def test_normal_distribution_r_consistency():
    """Test that Normal distribution matches R GAMLSS results."""
    # This requires R and gamlss package installed
    # Use the rbus framework for Python-R communication
    pass
```

---

## 📚 Documentation

### Documentation Structure

- **User documentation**: `docs/` directory
  - `docs/tutorials/`: Step-by-step tutorials
  - `docs/api/`: API reference
  - `docs/algorithm_guide.md`: Algorithm usage guide
- **Code documentation**: Docstrings in source code
- **Examples**: `examples/` directory

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add mathematical formulas when relevant (use LaTeX)
- Link to related documentation
- Keep documentation up-to-date with code changes

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation (if using Sphinx)
cd docs
make html

# View documentation
open _build/html/index.html
```

---

## 🔄 Pull Request Process

### Before Submitting

1. ✅ **Code quality**: Ensure code follows style guidelines
2. ✅ **Tests**: All tests pass, new tests added for new features
3. ✅ **Documentation**: Updated for new features or changes
4. ✅ **Commits**: Clear, descriptive commit messages
5. ✅ **Branch**: Up-to-date with main branch

### PR Checklist

When submitting a PR, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass (`pytest tests/`)
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] No merge conflicts with main branch
- [ ] PR description clearly explains changes
- [ ] Linked to relevant issue(s)

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Related Issues
Fixes #123

## Testing
Describe testing performed

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guidelines
```

### Review Process

1. **Automated checks**: CI/CD runs tests and checks
2. **Code review**: Maintainers review code
3. **Feedback**: Address review comments
4. **Approval**: At least one maintainer approval required
5. **Merge**: Maintainer merges PR

---

## 👥 Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Discord** (coming soon): Real-time chat
- **Mailing List** (coming soon): Announcements and updates

### Getting Help

- Check [documentation](docs/README.md)
- Search [existing issues](https://github.com/your-org/omnilss/issues)
- Ask in [GitHub Discussions](https://github.com/your-org/omnilss/discussions)
- Read [tutorials](docs/tutorials/)

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project README

---

## 📝 License

By contributing to OmniLSS, you agree that your contributions will be licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

---

## 🙏 Thank You!

Thank you for contributing to OmniLSS! Your contributions help make this project better for everyone.

**Questions?** Feel free to ask in [GitHub Discussions](https://github.com/your-org/omnilss/discussions) or open an issue.

---

**Last Updated**: 2026-05-02
