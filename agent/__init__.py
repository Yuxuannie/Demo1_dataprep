"""
Data Selection Agent Package

Contains:
- DataSelectionAgent: Main agent with OBSERVE-THINK-DECIDE-ACT reasoning
- LLM Configuration: Ollama setup for local LLM inference
"""

from .data_selection_agent import DataSelectionAgent
from .llm_config import initialize_ollama_llm, test_ollama_connection

__all__ = [
    'DataSelectionAgent',
    'initialize_ollama_llm',
    'test_ollama_connection'
]