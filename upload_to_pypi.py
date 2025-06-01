#!/usr/bin/env python3
"""
Script to upload funnydspy to PyPI.
Handles both test and production uploads with proper error handling.

Before running this script:
1. Create an account on PyPI (https://pypi.org/account/register/)
2. Create an API token at https://pypi.org/manage/account/token/
3. Configure your credentials:
   - Either set environment variables: TWINE_USERNAME=__token__ TWINE_PASSWORD=<your-token>
   - Or use: python -m twine configure

To upload to PyPI:
    python upload_to_pypi.py

To upload to TestPyPI first (recommended):
    python upload_to_pypi.py --test
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        sys.exit(1)

def check_dist_files():
    """Check if distribution files exist."""
    dist_path = Path("dist")
    if not dist_path.exists():
        print("❌ No dist/ directory found. Run 'python -m build' first.")
        sys.exit(1)
    
    files = list(dist_path.glob("*.whl")) + list(dist_path.glob("*.tar.gz"))
    if not files:
        print("❌ No distribution files found in dist/. Run 'python -m build' first.")
        sys.exit(1)
    
    print(f"📦 Found {len(files)} distribution files:")
    for file in files:
        print(f"   - {file.name}")
    return files

def check_package_online(test=False):
    """Check if package exists online."""
    if test:
        url = "https://test.pypi.org/project/funnydspy/"
        print(f"🔍 Checking TestPyPI: {url}")
    else:
        url = "https://pypi.org/project/funnydspy/"
        print(f"🔍 Checking PyPI: {url}")

def main():
    parser = argparse.ArgumentParser(description="Upload funnydspy to PyPI")
    parser.add_argument("--test", action="store_true", help="Upload to TestPyPI instead of PyPI")
    parser.add_argument("--skip-build", action="store_true", help="Skip building and go straight to upload")
    args = parser.parse_args()

    if not args.skip_build:
        # Clean and build
        run_command("rm -rf dist/ build/ *.egg-info/", "Cleaning previous builds")
        run_command("python -m build", "Building package")
    
    # Check distribution files
    dist_files = check_dist_files()
    
    # Validate package
    run_command("python -m twine check dist/*", "Validating package")
    
    # Upload
    if args.test:
        print("🚀 Uploading to TestPyPI...")
        run_command("python -m twine upload --repository testpypi dist/*", "Uploading to TestPyPI")
        print("\n✅ Upload to TestPyPI successful!")
        print("📥 To install from TestPyPI:")
        print("    pip install --index-url https://test.pypi.org/simple/ funnydspy")
    else:
        print("🚀 Uploading to PyPI...")
        run_command("python -m twine upload dist/*", "Uploading to PyPI")
        print("\n✅ Upload to PyPI successful!")
        print("📥 To install:")
        print("    pip install funnydspy")
    
    # Check online
    check_package_online(test=args.test)

if __name__ == "__main__":
    print("🔧 FunnyDSPy PyPI Upload Tool")
    print("=" * 40)
    main() 