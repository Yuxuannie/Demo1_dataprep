#!/usr/bin/env python3
"""
Setup script for Demo1 Data Preparation Agent
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Installation failed: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported."""
    print("\nTesting imports...")

    required_modules = [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("sklearn.decomposition", "PCA"),
        ("langchain.prompts", "ChatPromptTemplate"),
        ("langchain.llms.base", "LLM"),
    ]

    failed = []
    for module, item in required_modules:
        try:
            __import__(module, fromlist=[item])
            print(f"  ✓ {module}.{item}")
        except ImportError as e:
            print(f"  ✗ {module}.{item}: {e}")
            failed.append(f"{module}.{item}")

    if failed:
        print(f"\nFailed imports: {', '.join(failed)}")
        return False

    print("\n✓ All imports successful!")
    return True

def main():
    """Main setup function."""
    print("=" * 60)
    print("DEMO1 DATA PREPARATION AGENT SETUP")
    print("=" * 60)

    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("✗ requirements.txt not found!")
        sys.exit(1)

    # Install requirements
    if not install_requirements():
        print("\nSetup failed. Please check your Python environment.")
        sys.exit(1)

    # Test imports
    if not test_imports():
        print("\nSome imports failed. Please check the error messages above.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("You can now run: python3 chatbot.py")
    print("=" * 60)

if __name__ == "__main__":
    main()