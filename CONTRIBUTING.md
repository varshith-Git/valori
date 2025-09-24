# Contributing to Vectara

Thank you for your interest in contributing to Vectara! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to team@vectara.com.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/varshith-Git/valori.git
   cd valori
   ```
3. **Set up the development environment** (see below)
4. **Create a branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, virtualenv, or conda)

### Setup Steps

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Verify setup**:
   ```bash
   python -m pytest tests/ -v
   ```

## Contributing Process

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix existing issues
- **New features**: Add new functionality
- **Documentation**: Improve docs, examples, or comments
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Refactoring**: Improve code structure without changing functionality

### Workflow

1. **Check existing issues** and pull requests to avoid duplication
2. **Create an issue** for significant changes to discuss the approach
3. **Fork and clone** the repository
4. **Create a feature branch** from `main`
5. **Make your changes** following our coding standards
6. **Add tests** for new functionality
7. **Update documentation** as needed
8. **Run the test suite** and ensure all tests pass
9. **Submit a pull request**

## Coding Standards

### Python Style

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Import order**: Use `isort` for consistent import sorting
- **Type hints**: Use type hints for all function signatures
- **Docstrings**: Use Google-style docstrings

### Code Formatting

We use automated tools for code formatting:

```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Check formatting
black --check src/ tests/
```

### Code Quality Tools

```bash
# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/

# Security check with bandit
bandit -r src/
```

## Testing

### Test Structure

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Performance tests**: Benchmark critical operations

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src/vectordb --cov-report=html

# Run specific test file
python -m pytest tests/test_storage.py

# Run tests in parallel
python -m pytest tests/ -n auto

# Run only fast tests
python -m pytest tests/ -m "not slow"
```

### Writing Tests

- **Test naming**: Use descriptive test names
- **Test isolation**: Each test should be independent
- **Mocking**: Use mocks for external dependencies
- **Fixtures**: Use pytest fixtures for common setup
- **Assertions**: Use specific assertions with helpful messages

Example test:

```python
def test_memory_storage_store_vector():
    """Test storing a vector in memory storage."""
    storage = MemoryStorage({})
    storage.initialize()
    
    vector = np.array([1.0, 2.0, 3.0])
    vector_id = "test_vector"
    metadata = {"category": "test"}
    
    success = storage.store_vector(vector_id, vector, metadata)
    assert success
    
    retrieved_vector, retrieved_metadata = storage.retrieve_vector(vector_id)
    assert np.array_equal(retrieved_vector, vector)
    assert retrieved_metadata == metadata
```

## Documentation

### Code Documentation

- **Docstrings**: All public functions and classes need docstrings
- **Comments**: Explain complex logic, not obvious code
- **Type hints**: Use type hints for better IDE support

### API Documentation

- **Sphinx**: We use Sphinx for API documentation
- **Examples**: Include usage examples in docstrings
- **Parameter descriptions**: Document all parameters and return values

### User Documentation

- **README**: Keep README.md updated
- **Examples**: Maintain examples in `examples/` directory
- **Tutorials**: Add tutorials for complex features

## Pull Request Process

### Before Submitting

1. **Run the full test suite**:
   ```bash
   python -m pytest tests/
   ```

2. **Check code quality**:
   ```bash
   black --check src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

3. **Update documentation** if needed

4. **Add tests** for new functionality

### Pull Request Template

When submitting a PR, include:

- **Description**: What changes were made and why
- **Related issues**: Link to any related issues
- **Testing**: How the changes were tested
- **Breaking changes**: Note any breaking changes
- **Documentation**: Note any documentation updates needed

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Approval** from at least one maintainer
4. **Merge** by maintainers

## Issue Reporting

### Bug Reports

When reporting bugs, include:

- **Vectara version**: `python -c "import vectordb; print(vectordb.__version__)"`
- **Python version**: `python --version`
- **Operating system**: OS and version
- **Steps to reproduce**: Minimal code example
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full traceback if applicable

### Feature Requests

For feature requests, include:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other ways to solve the problem
- **Additional context**: Any other relevant information

## Development Guidelines

### Architecture

- **Modular design**: Keep components loosely coupled
- **Interface-based**: Use abstract base classes for extensibility
- **Configuration-driven**: Use configuration dictionaries
- **Error handling**: Provide meaningful error messages

### Performance

- **Benchmarking**: Benchmark critical paths
- **Memory usage**: Be mindful of memory consumption
- **Scalability**: Consider large dataset scenarios
- **Optimization**: Profile before optimizing

### Security

- **Input validation**: Validate all inputs
- **Error handling**: Don't expose sensitive information
- **Dependencies**: Keep dependencies up to date
- **Security scanning**: Run security tools regularly

## Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: team@vectara.com for private matters
- **Documentation**: Check the docs first

## Recognition

Contributors will be recognized in:

- **Contributors list** in README.md
- **Release notes** for significant contributions
- **Documentation** for code contributions

## License

By contributing to Vectara, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Vectara! 🚀
