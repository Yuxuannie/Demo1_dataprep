"""
Streamlit App Launcher for Timing-Aware Data Selection Agent
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    # Map package names to their import names
    package_imports = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'scikit-learn': 'sklearn',  # Important: scikit-learn imports as 'sklearn'
        'langchain': 'langchain',
        'python-dotenv': 'dotenv'
    }

    missing_packages = []
    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
            print(f"[OK] {package_name}: Available")
        except ImportError as e:
            missing_packages.append(package_name)
            print(f"[ERROR] {package_name}: Missing ({e})")

    if missing_packages:
        print(f"\nTips INSTALLATION SOLUTIONS:")
        print(f"1. Try: {sys.executable} -m pip install {' '.join(missing_packages)}")
        print(f"2. Force reinstall: {sys.executable} -m pip install --force-reinstall {' '.join(missing_packages)}")
        print(f"3. User install: {sys.executable} -m pip install --user {' '.join(missing_packages)}")
        print(f"4. Run diagnostics: python3 diagnose_env.py")
        return False

    print("[OK] All core dependencies satisfied")

    # Check optional dependencies
    optional_packages = ['seaborn']
    missing_optional = []
    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)

    if missing_optional:
        print(f"[INFO]  Optional packages not installed: {', '.join(missing_optional)}")
        print("   (Visualization will use matplotlib fallbacks)")

    return True

def main():
    """Launch Streamlit app."""
    print("Actions Launching Timing-Aware Data Selection Agent Web UI...")
    print(f"Using Python: {sys.executable}")

    # Check dependencies
    if not check_dependencies():
        print("\n[ERROR] Cannot start - missing dependencies")
        print("\n🔍 For detailed diagnostics, run: python3 diagnose_env.py")
        sys.exit(1)

    # Check if Ollama is running
    print("\nConnection Checking Ollama connection...")
    try:
        from agent.timing_llm_config import test_ollama_connection
        if test_ollama_connection():
            print("[OK] Ollama connected")
        else:
            print("[WARNING]  Ollama not connected. Start with: ollama serve")
    except Exception as e:
        print(f"[WARNING]  Could not test Ollama: {e}")

    # Launch Streamlit
    app_path = Path(__file__).parent / "app_ui.py"

    print(f"\n🌐 Starting Streamlit at: http://localhost:8501")
    print(f"App file: {app_path}")
    print("Press Ctrl+C to stop the server\n")

    try:
        # Use explicit python executable to ensure same environment
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--theme.base=light",
            "--theme.primaryColor=#1f4e79",
            "--theme.backgroundColor=#ffffff"
        ]
        print(f"Running command: {' '.join(cmd[:3])} {app_path}")
        subprocess.run(cmd)
    except FileNotFoundError as e:
        print(f"[ERROR] Error: {e}")
        print("Streamlit may not be installed. Try:")
        print(f"{sys.executable} -m pip install streamlit")
    except KeyboardInterrupt:
        print("\n👋 Streamlit server stopped")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        print("\n🔍 Run diagnostics: python3 diagnose_env.py")

if __name__ == "__main__":
    main()