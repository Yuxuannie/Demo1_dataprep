# Configuration Guide: Addresses & Endpoints

This guide explains the TWO different addresses in Demo 1 and how to configure them.

---

## 🌐 Address 1: Flask Server (Web UI)

**What it is**: Where YOUR DEMO WEB APPLICATION runs  
**Who accesses it**: Users in their web browser  
**Default**: `http://localhost:5000`

### How to Change Flask Server Address/Port:

#### Option 1: Edit app.py directly
```python
# In app.py (bottom)
if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0',  # Accept connections from any IP
        port=5000        # Change this to your preferred port
    )
```

#### Option 2: Use environment variables
```bash
export FLASK_RUN_HOST=0.0.0.0
export FLASK_RUN_PORT=8080
flask run
```

#### Option 3: Command line arguments
```bash
python app.py --host 0.0.0.0 --port 8080
```

### Common Configurations:

**Local development only** (most secure):
```python
app.run(host='127.0.0.1', port=5000)
# Access: http://localhost:5000
```

**Allow remote access** (for demo to others):
```python
app.run(host='0.0.0.0', port=5000)
# Access: http://YOUR_IP:5000 from any machine
```

**Use different port** (if 5000 is taken):
```python
app.run(host='0.0.0.0', port=8080)
# Access: http://localhost:8080
```

---

## 🤖 Address 2: LLM API Endpoint (Qwen Server)

**What it is**: Where THE AGENT CALLS for natural language reasoning  
**Who accesses it**: The agent code (backend)  
**Default**: Not configured (uses template-based reasoning)

### How to Configure Qwen LLM:

#### Step 1: Copy config example
```bash
cp config_example.py config.py
```

#### Step 2: Edit config.py
```python
QWEN_CONFIG = {
    'enabled': True,  # Enable LLM
    'api_url': 'http://your-qwen-server:8000/v1/chat/completions',  # YOUR QWEN SERVER
    'api_key': 'your-api-key-here',  # YOUR API KEY
    'model': 'qwen-72b',
    'timeout': 30
}
```

#### Step 3: Restart Flask server
```bash
python app.py
# Should see: "✓ LLM Configuration loaded: Enabled"
```

### Example Qwen Configurations:

**1. Local Qwen Server** (running on same machine):
```python
QWEN_CONFIG = {
    'enabled': True,
    'api_url': 'http://localhost:8000/v1/chat/completions',
    'api_key': 'local-dev-key',
    'model': 'qwen-72b'
}
```

**2. Internal TSMC Server** (hypothetical):
```python
QWEN_CONFIG = {
    'enabled': True,
    'api_url': 'http://qwen-inference.tsmc.com:8080/v1/chat/completions',
    'api_key': 'your-tsmc-api-key',
    'model': 'qwen-72b-instruct'
}
```

**3. Cloud Service** (hypothetical):
```python
QWEN_CONFIG = {
    'enabled': True,
    'api_url': 'https://api.qwen.cloud/v1/chat/completions',
    'api_key': 'sk-xxxxxxxxxxxxxxxxxxxxxxxx',
    'model': 'qwen-72b'
}
```

**4. Environment Variables** (most secure):
```python
import os

QWEN_CONFIG = {
    'enabled': True,
    'api_url': os.getenv('QWEN_API_URL'),
    'api_key': os.getenv('QWEN_API_KEY'),
    'model': 'qwen-72b'
}
```

Then run:
```bash
export QWEN_API_URL="http://your-qwen-server:8000/v1/chat/completions"
export QWEN_API_KEY="your-api-key"
python app.py
```

---

## 🔍 How to Verify Configuration

### 1. Check Flask Server:
```bash
python app.py
```

Should see:
```
🚀 Demo 1: Data Preparation Agent
================================================================================
Server starting at: http://localhost:5000
✓ LLM Configuration loaded: Enabled   (or "Disabled" if no config)
```

### 2. Access Web UI:
Open browser: `http://localhost:5000`  
Should see the demo interface.

### 3. Check LLM Connection (if configured):
Upload a CSV and run agent. Check console output:
```
✓ Using LLM-enhanced reasoning: http://your-qwen-server:8000/...
```

vs (if disabled):
```
✓ Using template-based reasoning
```

---

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  User's Browser                                             │
│  http://localhost:5000  ← YOU ACCESS THIS                  │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ HTTP requests (upload, run_agent, etc.)
               │
┌──────────────▼──────────────────────────────────────────────┐
│  Flask Server (app.py)                                      │
│  Runs at: localhost:5000                                    │
│  Handles: UI, file upload, API endpoints                    │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ Calls agent
               │
┌──────────────▼──────────────────────────────────────────────┐
│  Agent (data_prep_agent.py)                                 │
│  Performs: OBSERVE → THINK → DECIDE → ACT                   │
│  Uses: Template reasoning OR LLM reasoning                  │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ HTTP requests (if LLM enabled)
               │
┌──────────────▼──────────────────────────────────────────────┐
│  Qwen LLM Server                                            │
│  URL: http://your-qwen-server:8000 ← YOU CONFIGURE THIS    │
│  Provides: Natural language reasoning                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Common Issues & Solutions

### Issue 1: "Address already in use" (Port 5000 taken)
**Solution**: Change Flask port
```python
# In app.py
app.run(host='0.0.0.0', port=8080)  # Use 8080 instead
```

### Issue 2: "Connection refused" when accessing from another machine
**Solution**: Use `host='0.0.0.0'` (not `127.0.0.1`)
```python
app.run(host='0.0.0.0', port=5000)  # Allows remote access
```

### Issue 3: "LLM API error: Connection timeout"
**Solution**: Check Qwen server is running and accessible
```bash
# Test Qwen connection
curl http://your-qwen-server:8000/v1/models
```

### Issue 4: "Module 'config' not found"
**Solution**: This is normal! Agent falls back to templates
```bash
# To enable LLM, create config.py:
cp config_example.py config.py
# Then edit config.py with your Qwen details
```

---

## 🎯 Quick Configuration Checklist

For **Web Demo Only** (no LLM):
- [ ] Keep default: `app.run(host='0.0.0.0', port=5000)`
- [ ] Access at: `http://localhost:5000`
- [ ] Agent uses template-based reasoning
- [✓] Ready to demo!

For **LLM-Enhanced Demo**:
- [ ] Get Qwen API URL and key from your IT team
- [ ] Copy `config_example.py` → `config.py`
- [ ] Set `enabled=True` and update `api_url`, `api_key`
- [ ] Restart Flask: `python app.py`
- [ ] Verify: Should see "LLM Configuration loaded: Enabled"
- [✓] Ready for LLM-powered reasoning!

---

## 💡 Recommendations

### For Initial Demo (This Week):
✅ **Use template-based reasoning** (no LLM needed)
- Faster, more predictable
- No external dependencies
- Still shows intelligent reasoning

### For Production (Later):
✅ **Add Qwen LLM integration**
- Natural language explanations
- Better adaptability
- More impressive for management

---

## 📞 Need Help?

**Q**: "I want to demo on my laptop to management in conference room"  
**A**: Keep default `localhost:5000`, bring your laptop, no changes needed

**Q**: "I want colleagues to access demo from their machines"  
**A**: Use `host='0.0.0.0'`, share your IP: `http://YOUR_IP:5000`

**Q**: "I have Qwen running locally on port 8000"  
**A**: Set `api_url='http://localhost:8000/v1/chat/completions'` in config.py

**Q**: "Port 5000 is taken, how to change?"  
**A**: Change `port=5000` to `port=8080` in app.py

**Q**: "Where is the Qwen server?"  
**A**: Ask your IT team for internal Qwen inference server details
