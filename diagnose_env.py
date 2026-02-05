#!/usr/bin/env python3
"""
Environment Diagnostic Script for Timing Data Selection Agent
Helps identify Python environment and package installation issues
"""

import sys
import subprocess
import os

def print_separator(title):
    """Print section separator."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)

def check_python_info():
    """Check Python version and location."""
    print_separator("PYTHON ENVIRONMENT INFO")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths
    print(f"Current working directory: {os.getcwd()}")

def check_streamlit_python():
    """Check which Python Streamlit is using."""
    print_separator("STREAMLIT PYTHON INFO")
    try:
        result = subprocess.run([sys.executable, "-c", "import streamlit; print('Streamlit available in current Python')"],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("[OK] Streamlit available in current Python")
        else:
            print("[ERROR] Streamlit NOT available in current Python")
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"[ERROR] Error checking Streamlit: {e}")

    # Check streamlit command directly
    try:
        result = subprocess.run(["streamlit", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"[OK] Streamlit command available: {result.stdout.strip()}")

            # Check which python streamlit uses
            result = subprocess.run(["which", "streamlit"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"Streamlit location: {result.stdout.strip()}")
        else:
            print("[ERROR] Streamlit command not found")
    except Exception as e:
        print(f"[ERROR] Error checking streamlit command: {e}")

def check_package_availability():
    """Check availability of required packages."""
    print_separator("PACKAGE AVAILABILITY CHECK")

    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'matplotlib',
        'sklearn',  # This is how scikit-learn is imported
        'langchain',
        'dotenv'  # This is how python-dotenv is imported
    ]

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"[OK] {package}: Available")
        except ImportError as e:
            print(f"[ERROR] {package}: Missing - {e}")

def check_pip_info():
    """Check pip installation info."""
    print_separator("PIP INSTALLATION INFO")

    try:
        # Check pip version and location
        result = subprocess.run([sys.executable, "-m", "pip", "--version"],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"Pip info: {result.stdout.strip()}")
        else:
            print(f"[ERROR] Error getting pip info: {result.stderr}")
    except Exception as e:
        print(f"[ERROR] Error checking pip: {e}")

    # Check where packages are installed
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "show", "scikit-learn"],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"\nScikit-learn installation info:")
            print(result.stdout)
        else:
            print("[ERROR] Scikit-learn not found via pip show")
    except Exception as e:
        print(f"[ERROR] Error checking scikit-learn installation: {e}")

def suggest_fixes():
    """Suggest potential fixes."""
    print_separator("SUGGESTED FIXES")

    print("If packages are missing, try these solutions:")
    print()
    print("1. FORCE REINSTALL with current Python:")
    print(f"   {sys.executable} -m pip install --force-reinstall scikit-learn")
    print()
    print("2. INSTALL TO USER DIRECTORY:")
    print(f"   {sys.executable} -m pip install --user scikit-learn")
    print()
    print("3. UPGRADE PIP first:")
    print(f"   {sys.executable} -m pip install --upgrade pip")
    print(f"   {sys.executable} -m pip install scikit-learn")
    print()
    print("4. USE CONDA (if available):")
    print("   conda install scikit-learn")
    print()
    print("5. VIRTUAL ENVIRONMENT (recommended):")
    print("   python3 -m venv venv")
    print("   source venv/bin/activate  # On Mac/Linux")
    print("   pip install -r requirements.txt")
    print()
    print("6. RUN STREAMLIT with specific Python:")
    print(f"   {sys.executable} -m streamlit run app_ui.py")

def main():
    """Run all diagnostic checks."""
    print("🔍 TIMING AGENT ENVIRONMENT DIAGNOSTICS")
    print("This will help identify package installation issues")

    check_python_info()
    check_streamlit_python()
    check_package_availability()
    check_pip_info()
    suggest_fixes()

    print(f"\n{'=' * 60}")
    print("Info SUMMARY")
    print('=' * 60)
    print("Share this output if you need further help!")

if __name__ == "__main__":
    main()