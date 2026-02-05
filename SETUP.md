# Setup Instructions

## Quick Setup

Run the setup script to install all dependencies:

```bash
python3 setup.py
```

This will:
1. Install all required packages from `requirements.txt`
2. Test that imports work correctly
3. Verify the environment is ready

## Manual Setup

If you prefer to install manually:

```bash
# Install dependencies
pip install -r requirements.txt

# Test the installation
python3 test_import.py
```

## Required Dependencies

- **Data Processing**: pandas, numpy, scikit-learn, scipy
- **LangChain**: langchain, langchain-core, langchain-community
- **Other**: requests, python-dotenv

## Environment Variables

Create a `.env` file with your Ollama configuration:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:32b
LLM_TEMPERATURE=0.3
LLM_TOP_P=0.9
LLM_NUM_PREDICT=1500
```

## Running the Agent

After setup:

```bash
python3 chatbot.py
```

## Troubleshooting

### "Cannot import DataSelectionAgent" Error

This is usually caused by missing dependencies, not a circular import. Run:

```bash
python3 test_import.py
```

This will show you exactly which packages are missing.

### Common Issues

1. **sklearn not found**: Install with `pip install scikit-learn`
2. **langchain errors**: Update with `pip install --upgrade langchain`
3. **Ollama connection**: Ensure Ollama server is running and accessible