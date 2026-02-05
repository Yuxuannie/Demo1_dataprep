"""
Simplified LLM Configuration for Offline Local Ollama
No API keys needed - uses environment variables only
"""

import os
from langchain_community.llms import Ollama
from langchain.llms.base import LLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_ollama_llm() -> LLM:
    """
    Initialize Ollama LLM using environment variables.
    
    Required environment variables:
    - OLLAMA_BASE_URL: http://localhost:11434 or http://your-server:11434
    - OLLAMA_MODEL: qwen:32b (or other model you have pulled)
    
    Example .env:
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_MODEL=qwen:32b
    LLM_TEMPERATURE=0.3
    LLM_TOP_P=0.9
    LLM_NUM_PREDICT=1500
    """
    
    # Get configuration from environment variables
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    model = os.getenv('OLLAMA_MODEL', 'qwen:32b')
    temperature = float(os.getenv('LLM_TEMPERATURE', '0.3'))
    top_p = float(os.getenv('LLM_TOP_P', '0.9'))
    top_k = int(os.getenv('LLM_TOP_K', '40'))
    num_predict = int(os.getenv('LLM_NUM_PREDICT', '1500'))
    
    logger.info(f"🦙 Ollama Configuration:")
    logger.info(f"   Base URL: {base_url}")
    logger.info(f"   Model: {model}")
    logger.info(f"   Temperature: {temperature}")
    
    # Initialize Ollama LLM
    llm = Ollama(
        base_url=base_url,
        model=model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_predict=num_predict,
        verbose=True
    )
    
    logger.info(f"✓ Ollama LLM initialized successfully")
    
    return llm


def test_ollama_connection():
    """Test if Ollama server is accessible."""
    import requests
    
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            logger.info(f"✓ Ollama connected! Found {len(models)} models")
            return True
        else:
            logger.error(f"✗ Ollama returned status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"✗ Cannot connect to Ollama at {base_url}")
        logger.error(f"   Error: {e}")
        return False
