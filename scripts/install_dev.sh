#!/bin/bash

# Development environment setup script for Vectara
# This script sets up a development environment with all necessary dependencies

set -e  # Exit on any error

echo "Setting up Vectara development environment..."
echo "============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "Python version: $PYTHON_VERSION ✓"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
echo "Installing development dependencies..."
pip install -r requirements-dev.txt

# Install the package in development mode
echo "Installing Vectara in development mode..."
pip install -e .

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
else
    echo "Warning: pre-commit not found. Install it with: pip install pre-commit"
fi

# Run tests to verify installation
echo "Running tests to verify installation..."
python -m pytest tests/ -v --tb=short

echo ""
echo "Development environment setup complete! ✓"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  python -m pytest tests/"
echo ""
echo "To run linting:"
echo "  python -m flake8 src/"
echo "  python -m black src/ tests/"
echo ""
echo "To build documentation:"
echo "  cd docs && make html"
