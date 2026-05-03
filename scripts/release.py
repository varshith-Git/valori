#!/usr/bin/env python3
"""Release helper: clean old dists, build wheel, and upload to PyPI.

Usage examples:
  # interactive (will prompt before deleting old artifacts and uploading)
  python3 scripts/release.py

  # non-interactive: confirm deletions and upload to PyPI using env token
  PYPI_API_TOKEN="pypi-..." python3 scripts/release.py --yes

  # upload to TestPyPI using its token
  TEST_PYPI_API_TOKEN="pypi-..." python3 scripts/release.py --yes --test

Security notes:
  - Prefer passing tokens via env variables (PYPI_API_TOKEN or TEST_PYPI_API_TOKEN).
  - If no token provided, the script will rely on ~/.pypirc or repo .pypirc (if present).
  - Do NOT hard-code tokens in the repo.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


def parse_pyproject(pyproject_path: Path) -> Tuple[Optional[str], Optional[str]]:
    text = pyproject_path.read_text(encoding="utf-8")
    name_m = re.search(r"^\s*name\s*=\s*\"(?P<name>[^\"]+)\"", text, re.M)
    version_m = re.search(r"^\s*version\s*=\s*\"(?P<version>[^\"]+)\"", text, re.M)
    if name_m and version_m:
        return name_m.group("name"), version_m.group("version")
    return None, None


def parse_setup_cfg(setup_cfg_path: Path) -> Tuple[Optional[str], Optional[str]]:
    text = setup_cfg_path.read_text(encoding="utf-8")
    meta = re.search(r"\[metadata\](.*?)(\n\[|\Z)", text, re.S)
    if meta:
        section = meta.group(1)
        name_m = re.search(r"^\s*name\s*=\s*(?P<name>\S+)", section, re.M)
        version_m = re.search(r"^\s*version\s*=\s*(?P<version>\S+)", section, re.M)
        if name_m and version_m:
            return name_m.group("name").strip(), version_m.group("version").strip()
    return None, None


def get_version_from_init(root: Path) -> Optional[str]:
    """Try to read version from src/valori/__init__.py."""
    init_path = root / "src" / "valori" / "__init__.py"
    if not init_path.exists():
        return None
    text = init_path.read_text(encoding="utf-8")
    version_m = re.search(r"^\s*__version__\s*=\s*\"(?P<version>[^\"]+)\"", text, re.M)
    if version_m:
        return version_m.group("version")
    return None


def bump_version(root: Path, part: str = "patch") -> str:
    """Increment the version number in src/valori/__init__.py."""
    init_path = root / "src" / "valori" / "__init__.py"
    if not init_path.exists():
        raise FileNotFoundError(f"Could not find {init_path}")

    current_version = get_version_from_init(root)
    if not current_version:
        raise ValueError("Could not determine current version")

    # Split version into parts
    parts = current_version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {current_version}")

    major, minor, patch = map(int, parts)

    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    else:  # patch
        patch += 1

    new_version = f"{major}.{minor}.{patch}"

    # Update the file
    text = init_path.read_text(encoding="utf-8")
    new_text = re.sub(
        r"^(__version__\s*=\s*\")[^\"]+(\")$",
        rf"\g<1>{new_version}\g<2>",
        text,
        flags=re.M,
    )
    init_path.write_text(new_text, encoding="utf-8")

    print(f"Bumped version from {current_version} to {new_version}")
    return new_version


def get_name_version(root: Path) -> Tuple[str, str]:
    pyproj = root / "pyproject.toml"
    setup_cfg = root / "setup.cfg"
    name = version = None
    if pyproj.exists():
        name, version = parse_pyproject(pyproj)

    # Fallback for dynamic versioning
    if not version:
        version = get_version_from_init(root)

    if (not name or not version) and setup_cfg.exists():
        n, v = parse_setup_cfg(setup_cfg)
        name = name or n
        version = version or v

    if not name or not version:
        raise SystemExit(
            "ERROR: unable to determine package name/version from pyproject.toml or setup.cfg"
        )
    return name, version


def clean_old_dists(
    root: Path, name: str, version: str, dry_run: bool = False
) -> list[Path]:
    dist = root / "dist"
    removed = []
    if not dist.exists():
        return removed

    keep_token = f"{name}-{version}"
    for p in dist.iterdir():
        if not p.is_file():
            continue
        if not p.name.endswith((".whl", ".tar.gz", ".zip", ".tar.bz2")):
            continue
        if p.name.startswith(f"{name}-") and keep_token not in p.name:
            removed.append(p)

    if not removed:
        print("No older distribution files found to remove.")
        return removed

    print("Old distribution files detected:")
    for p in removed:
        print(" - ", p.name)

    if dry_run:
        print("Dry-run mode: not deleting files.")
        return removed

    for p in removed:
        try:
            p.unlink()
            print(f"Removed {p.name}")
        except Exception as e:
            print(f"Failed to remove {p}: {e}", file=sys.stderr)
    return removed


def run_build(root: Path) -> None:
    print("Building wheel + sdist with 'python -m build'...")
    cmd = [sys.executable, "-m", "build", "--sdist", "--wheel"]
    try:
        subprocess.check_call(cmd, cwd=root)
    except subprocess.CalledProcessError as e:
        # Try to install build and retry once
        print(
            "'build' module not available or failed — attempting to install/upgrade 'build' and retrying..."
        )
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "build",
                    "wheel",
                    "setuptools",
                ],
                cwd=root,
            )
        except subprocess.CalledProcessError:
            print(
                "Failed to install build/wheel via pip. Please install 'build' (pip install --upgrade build) and retry.",
                file=sys.stderr,
            )
            raise

        # retry build
        subprocess.check_call(cmd, cwd=root)


def upload_dist(
    root: Path, repository: Optional[str], use_token: bool, token_env_name: str
) -> None:
    print("Uploading distributions with twine...")

    env = os.environ.copy()

    token = os.environ.get(token_env_name)
    if use_token and token:
        env["TWINE_USERNAME"] = "__token__"
        env["TWINE_PASSWORD"] = token
        print(f"Using token from env var {token_env_name} (TWINE_USERNAME=__token__).")

    # twine upload with glob; use shell to allow wildcard expansion while keeping secrets in env
    if repository:
        repo_arg = f"--repository-url {repository}"
    else:
        repo_arg = ""
    full_cmd = f'"{sys.executable}" -m twine upload {repo_arg} "{root / "dist" / "*"}"'
    subprocess.check_call(full_cmd, cwd=root, env=env, shell=True)


def ensure_venv(root: Path) -> None:
    """Ensure the script is running inside the project's virtual environment."""
    venv_path = root / "venv"
    if not venv_path.exists():
        return

    venv_python = venv_path / "bin" / "python3"
    if not venv_python.exists():
        venv_python = venv_path / "bin" / "python"

    if venv_python.exists() and os.path.abspath(sys.executable) != os.path.abspath(
        str(venv_python)
    ):
        # We are not in the venv, so re-execute the script using the venv's python
        print(f"Auto-activating virtual environment: {venv_path}")
        os.environ["VIRTUAL_ENV"] = str(venv_path.absolute())
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Clean old dists, build wheel, and upload to PyPI."
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=True,
        help="auto-confirm deletions and upload (default: True)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be done without deleting or uploading",
    )
    parser.add_argument(
        "--test", action="store_true", help="upload to TestPyPI instead of PyPI"
    )
    parser.add_argument(
        "--skip-clean", action="store_true", help="skip cleaning old dist files"
    )
    parser.add_argument(
        "--bump",
        choices=["major", "minor", "patch"],
        nargs="?",
        const="patch",
        help="automatically bump the version (default: patch)",
    )
    args = parser.parse_args(argv)

    root = Path.cwd()
    ensure_venv(root)

    if args.bump:
        bump_version(root, args.bump)

    try:
        name, version = get_name_version(root)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2

    print(f"Package: {name}, version: {version}")

    if not args.skip_clean:
        removed = clean_old_dists(
            root, name, version, dry_run=args.dry_run
        )

    if args.dry_run:
        print("Dry-run complete. Exiting.")
        return 0

    run_build(root)

    if args.test:
        repository = "https://test.pypi.org/legacy/"
        token_env_name = "TEST_PYPI_API_TOKEN"
    else:
        repository = None
        token_env_name = "PYPI_API_TOKEN"

    token_provided = bool(os.environ.get(token_env_name))
    pypirc_present = (Path.home() / ".pypirc").exists() or (root / ".pypirc").exists()

    if not token_provided and not pypirc_present:
        print(
            "No PyPI token found in environment and no ~/.pypirc or .pypirc in project.\nPlease set PYPI_API_TOKEN or create ~/.pypirc, or run with --test and TEST_PYPI_API_TOKEN."
        )
        return 1

    try:
        upload_dist(
            root, repository, use_token=token_provided, token_env_name=token_env_name
        )
    except subprocess.CalledProcessError as e:
        print(f"Upload failed: {e}", file=sys.stderr)
        return e.returncode

    print("Upload complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
