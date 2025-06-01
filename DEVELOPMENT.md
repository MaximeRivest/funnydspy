# FunnyDSPy Development Guide

This guide covers how to develop, test, and publish the FunnyDSPy package.

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/funnydspy.git
cd funnydspy

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest tests/ --cov=funnydspy --cov-report=html

# Run specific test file
python -m pytest tests/test_basic.py

# Run with verbose output
python -m pytest -v
```

### Test Structure

```
tests/
├── __init__.py
├── test_basic.py          # Basic functionality tests
├── test_dataclasses.py    # Dataclass-specific tests
├── test_namedtuples.py    # NamedTuple-specific tests
├── test_optimization.py   # DSPy optimization tests
└── test_edge_cases.py     # Edge cases and error handling
```

### Manual Testing

```bash
# Test installation
pip install dist/funnydspy-*.whl

# Test import
python -c "import funnydspy; print(funnydspy.__version__)"

# Run example scripts
python examples/basic_usage.py
python examples/optimization_example.py
```

## 📦 Building and Publishing

### Version Management

1. Update version in `pyproject.toml`
2. Update version in `funnydspy/__init__.py`
3. Update CHANGELOG.md
4. Commit changes

### Building

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build package
python -m build

# Check package
python -m twine check dist/*
```

### Code Quality

```bash
# Format code
black funnydspy/ tests/

# Sort imports
isort funnydspy/ tests/

# Lint code
flake8 funnydspy/ tests/

# Type checking (optional)
mypy funnydspy/
```

## 📁 Project Structure

```
funnydspy/
├── funnydspy/           # Main package
│   ├── __init__.py      # Main module with decorators
│   └── utils.py         # Utility functions (if needed)
├── tests/               # Test suite
├── examples/            # Example scripts
├── docs/                # Documentation
├── pyproject.toml       # Project configuration
├── README.md            # Main documentation
├── DEVELOPMENT.md       # This file
├── LICENSE              # MIT License
└── CHANGELOG.md         # Version history
```

## 🐛 Debugging

### Common Issues

- **Import errors**: Check if package is installed in development mode
- **DSPy compatibility**: Ensure DSPy version matches requirements
- **Type conversion**: Check `_from_text` function for edge cases

### Debug Tools

```bash
# Check package location
- Use `python -c "import funnydspy; print(funnydspy.__file__)"` to check package location
# Check dependencies
- Use `python -m pip show funnydspy` for package details
```

## 🚀 Release Process

1. **Prepare Release**
   - Update version numbers
   - Update documentation
   - Run full test suite
   - Update CHANGELOG.md

2. **Build and Test**
   - Build package locally
   - Test installation
   - Run integration tests

3. **Publish**
   - Tag release in Git
   - Push to GitHub
   - Publish to PyPI
   - Update documentation

## 🤝 Contributing Guidelines

- Follow PEP 8 style guidelines
- Write tests for new features
- Update documentation
- Use meaningful commit messages
- Create pull requests for changes 