#!/bin/bash

# Test runner script for Vectara
# This script runs all tests with various configurations

set -e  # Exit on any error

echo "Running Vectara tests..."
echo "======================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run install_dev.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Default test configuration
TEST_ARGS=""
COVERAGE=false
VERBOSE=false
PARALLEL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            TEST_ARGS="$TEST_ARGS -v"
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose    Run tests in verbose mode"
            echo "  -c, --coverage   Generate coverage report"
            echo "  -p, --parallel   Run tests in parallel"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Run all tests"
            echo "  $0 -v                 # Run tests with verbose output"
            echo "  $0 -c                 # Run tests with coverage"
            echo "  $0 -v -c              # Run tests with verbose output and coverage"
            echo "  $0 tests/test_storage.py  # Run specific test file"
            exit 0
            ;;
        *)
            # Assume it's a test file or directory
            TEST_ARGS="$TEST_ARGS $1"
            shift
            ;;
    esac
done

# Add default test arguments if none specified
if [ -z "$TEST_ARGS" ]; then
    TEST_ARGS="tests/"
fi

# Add coverage if requested
if [ "$COVERAGE" = true ]; then
    TEST_ARGS="--cov=src/vectordb --cov-report=html --cov-report=term-missing $TEST_ARGS"
fi

# Add parallel execution if requested
if [ "$PARALLEL" = true ]; then
    TEST_ARGS="-n auto $TEST_ARGS"
fi

# Add pytest configuration
TEST_ARGS="$TEST_ARGS --tb=short --strict-markers"

echo "Test configuration:"
echo "  Args: $TEST_ARGS"
echo "  Coverage: $COVERAGE"
echo "  Verbose: $VERBOSE"
echo "  Parallel: $PARALLEL"
echo ""

# Run tests
echo "Starting test execution..."
python -m pytest $TEST_ARGS

# Check if tests passed
if [ $? -eq 0 ]; then
    echo ""
    echo "All tests passed! ✓"
    
    if [ "$COVERAGE" = true ]; then
        echo ""
        echo "Coverage report generated in htmlcov/index.html"
    fi
else
    echo ""
    echo "Some tests failed! ✗"
    exit 1
fi
