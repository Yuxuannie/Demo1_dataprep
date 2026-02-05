#!/usr/bin/env python3
"""
Fix Special Symbol Issues in Streamlit App
Replaces Unicode symbols with plain text equivalents
"""

import re

def fix_symbols_in_file(file_path):
    """Replace special symbols with plain text."""

    # Symbol replacement mapping
    symbol_replacements = {
        '👤': '[USER]',
        '🤖': '[AGENT]',
        '💬': 'Chat',
        '⚙️': 'Setup',
        '📁': 'Data',
        '🚀': 'Actions',
        '💡': 'Tips',
        '📊': 'Dashboard',
        '🧠': 'Reasoning',
        '❌': '[ERROR]',
        '✅': '[OK]',
        '💾': 'Save',
        'ℹ️': '[INFO]',
        '📥': 'Download',
        '📄': 'Report',
        '🗑️': 'Clear',
        '🔄': 'Refresh',
        '📋': 'Info',
        '❗': '[!]',
        '⚠️': '[WARNING]',
        '🟢': '[CONNECTED]',
        '🟡': '[PENDING]',
        '📡': 'Connection',
        '🎯': 'Target',
        '📈': 'Analysis',
        '💻': 'System'
    }

    print(f"Processing {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Replace each symbol
    for symbol, replacement in symbol_replacements.items():
        if symbol in content:
            count = content.count(symbol)
            content = content.replace(symbol, replacement)
            print(f"  Replaced {count} instances of '{symbol}' with '{replacement}'")

    # Write back if changes were made
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Updated {file_path}")
    else:
        print(f"  No symbols found in {file_path}")

def main():
    """Fix symbols in all relevant files."""
    files_to_fix = [
        'app_ui.py',
        'run_streamlit.py',
        'diagnose_env.py',
        'fix_dependencies.py'
    ]

    print("🔧 Fixing special symbol display issues...")

    for file_path in files_to_fix:
        try:
            fix_symbols_in_file(file_path)
        except FileNotFoundError:
            print(f"  Skipping {file_path} (not found)")
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")

    print("\n✓ Symbol fixing complete!")

if __name__ == "__main__":
    main()