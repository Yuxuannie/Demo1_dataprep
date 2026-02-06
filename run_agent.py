#!/usr/bin/env python3
"""
Simple launcher for Timing Agent UI
Run: python run_agent.py
"""

import subprocess
import sys
import os

def run_timing_agent():
    """Launch the Timing Agent Streamlit UI."""

    print("=" * 60)
    print("TIMING DATA SELECTION AGENT")
    print("=" * 60)

    # Check if app_ui.py exists
    if not os.path.exists("app_ui.py"):
        print("ERROR: app_ui.py not found in current directory")
        sys.exit(1)

    print("Starting Streamlit UI...")
    print("Access the agent at: http://localhost:8501")
    print("Press Ctrl+C to stop")
    print("-" * 60)

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app_ui.py",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\nTiming Agent stopped.")
    except Exception as e:
        print(f"ERROR: Failed to start Streamlit: {e}")
        print("Try: pip install streamlit")

if __name__ == "__main__":
    run_timing_agent()