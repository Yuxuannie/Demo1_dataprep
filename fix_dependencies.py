#!/usr/bin/env python3
"""
Dependency Fix Script for Timing Data Selection Agent
Attempts to resolve common package installation issues
"""

import sys
import subprocess
import os

def run_command(cmd, description):
    """Run a command and show results."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("✅ Success")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
        else:
            print("❌ Failed")
            if result.stderr.strip():
                print(f"Error: {result.stderr.strip()}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main fix routine."""
    print("🔧 DEPENDENCY FIX SCRIPT")
    print("Attempting to resolve package installation issues...")
    print(f"Using Python: {sys.executable}")

    # List of critical packages
    packages = [
        'streamlit',
        'pandas',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'langchain',
        'python-dotenv'
    ]

    print(f"\n📦 Will install/upgrade: {', '.join(packages)}")

    # Method 1: Upgrade pip first
    success = run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        "Upgrading pip to latest version"
    )

    # Method 2: Install packages with force reinstall
    if success:
        success = run_command(
            [sys.executable, "-m", "pip", "install", "--force-reinstall"] + packages,
            "Force reinstalling all packages"
        )

    # Method 3: If force reinstall fails, try user install
    if not success:
        print("\n🔄 Trying user-local installation...")
        success = run_command(
            [sys.executable, "-m", "pip", "install", "--user", "--force-reinstall"] + packages,
            "Installing to user directory"
        )

    # Method 4: Clear cache and reinstall
    if not success:
        print("\n🔄 Clearing pip cache and reinstalling...")
        run_command(
            [sys.executable, "-m", "pip", "cache", "purge"],
            "Clearing pip cache"
        )
        success = run_command(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir"] + packages,
            "Installing without cache"
        )

    # Verification
    print(f"\n{'='*50}")
    print("🔍 VERIFICATION")
    print('='*50)

    package_imports = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'scikit-learn': 'sklearn',
        'langchain': 'langchain',
        'python-dotenv': 'dotenv'
    }

    all_good = True
    for pkg_name, import_name in package_imports.items():
        try:
            __import__(import_name)
            print(f"✅ {pkg_name}: Working")
        except ImportError as e:
            print(f"❌ {pkg_name}: Still missing - {e}")
            all_good = False

    print(f"\n{'='*50}")
    if all_good:
        print("🎉 SUCCESS! All packages are now available.")
        print("You can now run: python3 run_streamlit.py")
    else:
        print("⚠️  Some packages are still missing.")
        print("Try running: python3 diagnose_env.py for more detailed analysis")

        print("\nManual installation options:")
        print("1. conda install scikit-learn streamlit pandas numpy matplotlib")
        print("2. Create virtual environment:")
        print("   python3 -m venv venv")
        print("   source venv/bin/activate")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()