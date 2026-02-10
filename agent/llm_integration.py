"""
LLM Integration for Real Intelligence Agent
Supports Ollama and other LLM providers
"""

import requests
import json
from typing import Optional, Dict, Any


class OllamaLLM:
    """Ollama LLM integration for real intelligence agent"""

    def __init__(self, base_url: str = "http://f15dtpai1:11434", model: str = "qwen2.5_coder_32B"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.headers = {"Content-Type": "application/json"}

    def invoke(self, prompt: str) -> str:
        """Invoke the LLM with a prompt"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                headers=self.headers,
                data=json.dumps(payload),
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"LLM Error: HTTP {response.status_code}"

        except Exception as e:
            return f"LLM Error: {str(e)}"


def test_ollama_connection(base_url: str = "http://f15dtpai1:11434", model: str = "qwen2.5_coder_32B") -> bool:
    """Test if Ollama is available and model is accessible"""
    try:
        # Test basic connectivity
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code != 200:
            print(f"[LLM] Ollama server not responding at {base_url}")
            return False

        # Check if the model is available
        models = response.json().get('models', [])
        model_names = [m.get('name', '').split(':')[0] for m in models]

        if model not in model_names:
            print(f"[LLM] Model '{model}' not found. Available models: {model_names}")
            return False

        # Test a simple generation
        test_llm = OllamaLLM(base_url, model)
        test_response = test_llm.invoke("Hello")

        if "Error:" in test_response:
            print(f"[LLM] Test generation failed: {test_response}")
            return False

        print(f"[LLM] Ollama connection successful. Model: {model}")
        return True

    except Exception as e:
        print(f"[LLM] Connection test failed: {e}")
        return False


def initialize_timing_llm(base_url: str = "http://f15dtpai1:11434", model: str = "qwen2.5_coder_32B") -> Optional[OllamaLLM]:
    """Initialize LLM for timing data analysis"""
    if test_ollama_connection(base_url, model):
        return OllamaLLM(base_url, model)
    return None


# For compatibility with different LLM input formats
class LLMInputAdapter:
    """Adapter to handle different LLM input formats"""

    def __init__(self, llm):
        self.llm = llm

    def invoke(self, input_data):
        """Handle both string and dict inputs"""
        if isinstance(input_data, str):
            prompt = input_data
        elif isinstance(input_data, dict):
            prompt = input_data.get('input', str(input_data))
        else:
            prompt = str(input_data)

        return self.llm.invoke(prompt)