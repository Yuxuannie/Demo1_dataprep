# Troubleshooting Guide

## 🚨 Common Issue: "Missing packages scikit-learn"

Even after successful pip installation, you might see import errors. This usually means pip installed packages to a different Python environment than Streamlit is using.

### 🔧 Quick Fixes

#### Method 1: Use the Fix Script
```bash
python3 fix_dependencies.py
```

#### Method 2: Diagnostic First
```bash
python3 diagnose_env.py
```
This will show you exactly which Python environments are being used.

#### Method 3: Force Reinstall with Correct Python
```bash
# Find which Python Streamlit uses
which python3
which streamlit

# Install with that specific Python
/usr/bin/python3 -m pip install --force-reinstall scikit-learn
# OR
python3 -m pip install --force-reinstall scikit-learn
```

#### Method 4: Virtual Environment (Recommended)
```bash
# Create clean environment
python3 -m venv timing_agent_env
source timing_agent_env/bin/activate  # Mac/Linux
# timing_agent_env\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt

# Run app
python run_streamlit.py
```

### 🔍 Understanding the Issue

**The Problem:**
- `pip install scikit-learn` succeeds
- `import sklearn` fails in Streamlit
- This means pip installed to Python A, but Streamlit uses Python B

**Common Causes:**
1. Multiple Python versions installed (system Python, Homebrew Python, conda Python)
2. Virtual environments not activated
3. Different PATH settings between terminal and Streamlit

### 📋 Verification Commands

Check what's happening:
```bash
# Check Python locations
which python
which python3
python3 -c "import sys; print(sys.executable)"

# Check if packages are really installed
python3 -c "import sklearn; print('sklearn works!')"
python3 -m pip show scikit-learn

# Check Streamlit's Python
streamlit --version
which streamlit
```

### ⚡ Quick Test

After any fix attempt, test quickly:
```bash
python3 -c "
import streamlit
import sklearn
import pandas
import numpy
import matplotlib
print('✅ All packages working!')
"
```

### 🆘 Still Not Working?

1. **Try conda instead of pip:**
   ```bash
   conda install scikit-learn streamlit pandas numpy matplotlib
   ```

2. **Check for permission issues:**
   ```bash
   python3 -m pip install --user scikit-learn
   ```

3. **Clear all Python caches:**
   ```bash
   python3 -m pip cache purge
   python3 -c "import sys; sys.exit(0)"
   ```

### 🎯 Environment-Specific Solutions

#### macOS with Homebrew
```bash
# Use Homebrew Python specifically
/opt/homebrew/bin/python3 -m pip install scikit-learn
/opt/homebrew/bin/python3 run_streamlit.py
```

#### macOS with System Python
```bash
# Use system Python specifically
/usr/bin/python3 -m pip install --user scikit-learn
/usr/bin/python3 run_streamlit.py
```

#### Using pyenv
```bash
pyenv versions  # See available versions
pyenv global 3.9.7  # Set global version
python -m pip install scikit-learn
```

### ✅ Success Indicators

You'll know it's working when:
1. `python3 diagnose_env.py` shows all packages as "Available"
2. `python3 fix_dependencies.py` shows "SUCCESS! All packages are now available"
3. `python3 run_streamlit.py` starts without dependency errors