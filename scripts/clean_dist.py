#!/usr/bin/env python3
"""Clean older distribution artifacts from the dist/ directory.

This script reads `pyproject.toml` (or falls back to `setup.cfg`) to determine
the current package name and version, then removes any files in `dist/`
matching ``{name}-*`` that do NOT contain the current version. It only
removes common distribution file extensions (.whl, .tar.gz, .zip, .tar.bz2).

Safe defaults: the script prints what it will remove and only deletes files
that look like package distributions for the same package name but different
version. It will exit with 0 on success.
"""
from pathlib import Path
import re
import sys


def parse_pyproject(pyproject_path: Path):
    text = pyproject_path.read_text(encoding="utf-8")
    name_m = re.search(r"^\s*name\s*=\s*\"(?P<name>[^\"]+)\"", text, re.M)
    version_m = re.search(r"^\s*version\s*=\s*\"(?P<version>[^\"]+)\"", text, re.M)
    if name_m and version_m:
        return name_m.group("name"), version_m.group("version")
    return None, None


def parse_setup_cfg(setup_cfg_path: Path):
    text = setup_cfg_path.read_text(encoding="utf-8")
    # naive parse: look for [metadata] then name/version lines
    meta = re.search(r"\[metadata\](.*?)(\n\[|\Z)", text, re.S)
    if meta:
        section = meta.group(1)
        name_m = re.search(r"^\s*name\s*=\s*(?P<name>\S+)", section, re.M)
        version_m = re.search(r"^\s*version\s*=\s*(?P<version>\S+)", section, re.M)
        if name_m and version_m:
            return name_m.group("name").strip(), version_m.group("version").strip()
    return None, None


def main():
    root = Path.cwd()
    pyproj = root / "pyproject.toml"
    setup_cfg = root / "setup.cfg"

    name = None
    version = None
    if pyproj.exists():
        name, version = parse_pyproject(pyproj)

    if not name or not version:
        if setup_cfg.exists():
            name, version = parse_setup_cfg(setup_cfg)

    if not name or not version:
        print("Unable to determine package name/version from pyproject.toml or setup.cfg", file=sys.stderr)
        sys.exit(1)

    dist = root / "dist"
    if not dist.exists() or not dist.is_dir():
        print("No dist/ directory found, nothing to clean.")
        return

    keep_token = f"{name}-{version}"
    removed = []
    for p in dist.iterdir():
        if not p.is_file():
            continue
        # Only target common distribution file extensions
        if not p.name.endswith(('.whl', '.tar.gz', '.zip', '.tar.bz2')):
            continue
        if p.name.startswith(f"{name}-") and keep_token not in p.name:
            removed.append(p)

    if not removed:
        print("No older distribution files found to remove.")
        return

    print("Removing the following old distribution files:")
    for p in removed:
        print(f" - {p.name}")

    # Confirm removal (non-interactive) and delete
    for p in removed:
        try:
            p.unlink()
        except Exception as e:
            print(f"Failed to remove {p}: {e}", file=sys.stderr)

    print("Clean complete.")


if __name__ == "__main__":
    main()
