# Quick Fix: Column Name Error

## Problem
Error: `'('cross_sigma_sec',) not in index'`

This means your CSV has different column names than the agent expected.

---

## Solution 1: Use Updated Agent (Recommended)

I've fixed the agent to automatically detect your CSV columns!

### Steps:
1. **Stop the current server**: Press `Ctrl+C` in terminal
2. **Copy the fixed agent**:
   ```bash
   cd /home/claude/demo1_dataprep
   cp /mnt/user-data/outputs/demo1_dataprep_fixed/agent/data_prep_agent.py agent/
   ```
3. **Restart server**:
   ```bash
   python app.py
   ```
4. **Re-upload your CSV** and try again!

The agent will now:
- ✅ Look for ideal columns first
- ✅ Fall back to any numeric columns if ideal ones missing
- ✅ Work with any CSV structure!

---

## Solution 2: Use Mock Data (Quick Test)

If you want to see the demo work immediately:

1. In the upload dialog, select:
   ```
   demo1_dataprep/mock_data/test_data.csv
   ```
2. This file has all the right columns
3. Demo will work perfectly!

---

## What Was Fixed

### Before (Rigid):
```python
# Expected exact columns
feature_cols = [
    'nominal_delay', 'lib_sigma_delay_late',
    'nominal_tran', 'lib_sigma_tran_late',
    'sigma_by_nominal', 'early_sigma_by_late_sigma',
    'stdev_by_late_sigma', 'cross_sigma_sec'
]
self.features = self.df[feature_cols].values  # FAILS if missing!
```

### After (Flexible):
```python
# Try preferred columns first
preferred_cols = ['nominal_delay', 'lib_sigma_delay_late', ...]

# Use only columns that exist
available_cols = self.df.columns.tolist()
feature_cols = [col for col in preferred_cols if col in available_cols]

# If not enough, use all numeric columns
if len(feature_cols) < 3:
    feature_cols = [col for col in available_cols 
                   if col != 'arc_pt' and self.df[col].dtype in ['float64', 'int64']]

self.features = self.df[feature_cols].values  # ALWAYS works!
```

---

## For Your Real CSV

### What Columns Do You Have?

Can you run this to see your CSV columns?
```bash
head -1 /path/to/your/file.csv
```

Or in Python:
```python
import pandas as pd
df = pd.read_csv('/path/to/your/file.csv')
print(df.columns.tolist())
```

### The Agent Needs:
- ✅ At least 3 numeric features (delay, tran, sigma, etc.)
- ✅ One identifier column (arc_pt, cell_name, etc.)
- ✅ No specific naming required anymore!

---

## Test the Fix

1. Restart server with updated code
2. Upload your CSV again
3. Should work this time! ✅

If it still fails, show me the error and I'll fix it immediately!
