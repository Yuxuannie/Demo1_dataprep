"""
Streamlit App Launcher for Timing-Aware Data Selection Agent
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'langchain'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        return False

    print("✅ All dependencies satisfied")
    return True

def main():
    """Launch Streamlit app."""
    print("🚀 Launching Timing-Aware Data Selection Agent Web UI...")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check if Ollama is running
    print("\n📡 Checking Ollama connection...")
    try:
        from agent.timing_llm_config import test_ollama_connection
        if test_ollama_connection():
            print("✅ Ollama connected")
        else:
            print("⚠️  Ollama not connected. Start with: ollama serve")
    except Exception as e:
        print(f"⚠️  Could not test Ollama: {e}")

    # Launch Streamlit
    app_path = Path(__file__).parent / "app_ui.py"

    print(f"\n🌐 Starting Streamlit at: http://localhost:8501")
    print("Press Ctrl+C to stop the server\n")

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--theme.base=light",
            "--theme.primaryColor=#1f4e79",
            "--theme.backgroundColor=#ffffff"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit server stopped")

if __name__ == "__main__":
    main()