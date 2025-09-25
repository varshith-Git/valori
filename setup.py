#!/usr/bin/env python3
"""
Setup script for Valori vector database.

This file is provided for compatibility with older build tools.
For new projects, use pyproject.toml instead.
"""

from setuptools import setup, find_packages

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="valori",
    version="0.1.1",
    author="Varshith",
    author_email="varshith.gudur17@gmail.com",
    description="A high-performance vector database library for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/varshith-Git/valori",
    project_urls={
        "Bug Tracker": "https://github.com/varshith-Git/valori/issues",
        "Documentation": "https://github.com/varshith-Git/valori",
        "Source Code": "https://github.com/varshith-Git/valori",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pytest-xdist>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
            "pre-commit>=2.0",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "safety>=1.10",
            "bandit>=1.7",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "sphinx-autodoc-typehints>=1.12",
        ],
        "benchmark": [
            "matplotlib>=3.3",
            "seaborn>=0.11",
            "pandas>=1.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "valori-benchmark=scripts.benchmark:main",
        ],
    },
    include_package_data=True,
    package_data={
        "valori": ["py.typed"],
    },
    zip_safe=False,
)
