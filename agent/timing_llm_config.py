"""
LLM Configuration for Timing Domain Analysis
Optimized parameters for senior timing engineer level reasoning
"""

import os
from agent.llm_config import initialize_ollama_llm, test_ollama_connection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_timing_llm():
    """
    Initialize LLM with timing domain optimized parameters.

    This builds on your existing working LLM configuration
    and adds enhanced parameters for timing domain expertise.
    """

    # Determine parameters based on available prompt mode
    try:
        from agentic_timing_prompts import AGENTIC_LLM_PARAMETERS
        agentic_params = AGENTIC_LLM_PARAMETERS
        timing_params = {
            'LLM_TEMPERATURE': str(agentic_params.get('temperature', 0.25)),
            'LLM_TOP_P': str(agentic_params.get('top_p', 0.90)),
            'LLM_TOP_K': str(agentic_params.get('top_k', 40)),
            'LLM_NUM_PREDICT': str(agentic_params.get('num_predict', 2500)),
            'LLM_REPEAT_PENALTY': str(agentic_params.get('repeat_penalty', 1.20))
        }
        mode_info = "AGENTIC MODE (autonomous exploration)"
    except ImportError:
        timing_params = {
            'LLM_TEMPERATURE': '0.2',     # Lower for consistent technical reasoning
            'LLM_TOP_P': '0.9',          # Focused sampling for domain expertise
            'LLM_TOP_K': '40',           # Reduced for technical precision
            'LLM_NUM_PREDICT': '2500',   # More tokens for detailed explanations
            'LLM_REPEAT_PENALTY': '1.1'  # Avoid repetitive generic phrases
        }
        mode_info = "STANDARD MODE (structured reasoning)"

    # Apply parameters only if not already set
    applied_count = 0
    for param, value in timing_params.items():
        if not os.getenv(param):
            os.environ[param] = value
            applied_count += 1

    logger.info(f"Timing Domain LLM Configuration - {mode_info}:")
    logger.info(f"   Base URL: {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}")
    logger.info(f"   Model: {os.getenv('OLLAMA_MODEL', 'qwen:32b')}")
    logger.info(f"   Temperature: {os.getenv('LLM_TEMPERATURE')} (exploration vs consistency)")
    logger.info(f"   Top-P: {os.getenv('LLM_TOP_P')} (creative vs focused)")
    logger.info(f"   Max Tokens: {os.getenv('LLM_NUM_PREDICT')} (extended reasoning)")
    if applied_count > 0:
        logger.info(f"   Applied {applied_count} parameters for {mode_info}")

    try:
        llm = initialize_ollama_llm()
        logger.info("Timing domain LLM initialized successfully")
        return llm

    except Exception as e:
        logger.error(f"Timing LLM initialization failed: {e}")
        logger.error("Please check your Ollama setup and model availability")
        raise


def validate_timing_llm():
    """Validate LLM works for timing domain tasks."""
    logger.info("Validating timing domain LLM configuration...")

    if not test_ollama_connection():
        logger.error("Ollama connection test failed")
        return False

    try:
        llm = initialize_timing_llm()

        test_prompt = """You are a senior timing engineer. In 1 sentence, explain why uncertainty
        sampling is better than random sampling for Monte Carlo timing characterization."""

        response = llm.invoke(test_prompt)
        response_text = response if isinstance(response, str) else response.content

        if len(response_text) > 30 and any(word in response_text.lower()
                                         for word in ['uncertainty', 'sampling', 'timing']):
            logger.info("Timing domain LLM validation successful")
            return True
        else:
            logger.warning("LLM responds but may need prompt tuning")
            return False

    except Exception as e:
        logger.error(f"Timing domain LLM validation failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Timing Domain LLM Configuration Test")
    print("=" * 60)

    if validate_timing_llm():
        print("\nTiming domain LLM configuration successful!")
        print("Ready for timing-aware data selection agent.")
    else:
        print("\nConfiguration test failed.")
        print("Check Ollama setup and model availability.")