"""
Enhanced LLM Configuration - Compatible Version
Builds on your working agent/llm_config.py with enhanced parameters
"""

import os
import logging
from agent.llm_config import initialize_ollama_llm, test_ollama_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_enhanced_ollama_llm():
    """
    Initialize enhanced LLM using your existing working configuration.

    This version:
    - Uses your proven working LLM setup
    - Adds enhanced parameters via environment variables
    - Maintains full compatibility
    - Focuses on timing domain optimization
    """

    # Set enhanced parameters via environment if not already set
    enhanced_params = {
        'LLM_TEMPERATURE': '0.2',     # Lower for consistency
        'LLM_TOP_P': '0.9',          # Focused sampling
        'LLM_TOP_K': '40',           # Precision tuned
        'LLM_NUM_PREDICT': '2500',   # More tokens for detailed explanations
        'LLM_REPEAT_PENALTY': '1.1'  # Avoid repetitive phrases
    }

    # Apply enhanced parameters without overriding existing ones
    for param, value in enhanced_params.items():
        if not os.getenv(param):
            os.environ[param] = value
            logger.info(f"🎯 Enhanced parameter: {param}={value}")

    logger.info(f"🦙 Enhanced Ollama Configuration for Timing Domain:")
    logger.info(f"   Base URL: {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}")
    logger.info(f"   Model: {os.getenv('OLLAMA_MODEL', 'qwen:32b')}")
    logger.info(f"   Temperature: {os.getenv('LLM_TEMPERATURE')} (optimized for consistency)")
    logger.info(f"   Top-P: {os.getenv('LLM_TOP_P')} (focused sampling)")
    logger.info(f"   Top-K: {os.getenv('LLM_TOP_K')} (precision tuned)")
    logger.info(f"   Max Tokens: {os.getenv('LLM_NUM_PREDICT')} (detailed explanations)")
    logger.info(f"   Repeat Penalty: {os.getenv('LLM_REPEAT_PENALTY')} (avoid generic phrases)")

    try:
        # Use your proven working LLM configuration
        llm = initialize_ollama_llm()
        logger.info("✓ Enhanced timing domain LLM initialized successfully")
        logger.info("✓ Using your existing working configuration + enhanced parameters")
        return llm

    except Exception as e:
        logger.error(f"✗ Enhanced LLM initialization failed: {e}")
        logger.error("Please check your Ollama setup and model availability")
        raise


def validate_enhanced_configuration():
    """
    Validate that enhanced configuration works with timing domain prompts.
    """
    logger.info("🔍 Validating enhanced configuration for timing domain...")

    if not test_ollama_connection():
        logger.error("✗ Ollama connection test failed")
        return False

    try:
        llm = initialize_enhanced_ollama_llm()

        # Test with a simple timing domain prompt
        test_prompt = """You are a senior timing engineer. In 1 sentence, explain why uncertainty
        sampling is better than random sampling for Monte Carlo timing characterization."""

        response = llm.invoke(test_prompt)
        response_text = response if isinstance(response, str) else response.content

        # Basic validation
        if len(response_text) > 30 and any(word in response_text.lower()
                                         for word in ['uncertainty', 'sampling', 'timing']):
            logger.info("✓ Enhanced configuration validation successful")
            logger.info(f"✓ Test response length: {len(response_text)} characters")
            return True
        else:
            logger.warning("⚠️ Configuration works but response quality needs improvement")
            return False

    except Exception as e:
        logger.error(f"✗ Enhanced configuration validation failed: {e}")
        return False


def get_enhanced_parameters_summary():
    """
    Get summary of enhanced parameters for timing domain.
    """
    return {
        'optimization_goal': 'Senior timing engineer level reasoning',
        'key_improvements': [
            'Lower temperature (0.2) for consistent technical responses',
            'Focused top-p (0.9) for domain-appropriate vocabulary',
            'More tokens (2500) for detailed explanations',
            'Repeat penalty (1.1) to avoid generic ML phrases'
        ],
        'compatibility': 'Uses existing working LLM configuration',
        'domain_focus': 'Timing signoff, Monte Carlo sampling, active learning'
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced LLM Configuration - Compatibility Test")
    print("=" * 60)

    # Test enhanced configuration
    try:
        success = validate_enhanced_configuration()

        if success:
            print("\n✓ Enhanced configuration ready for timing domain agent!")
            print("✓ Compatible with your existing setup")

            summary = get_enhanced_parameters_summary()
            print(f"\nOptimization Goal: {summary['optimization_goal']}")
            print("\nKey Improvements:")
            for improvement in summary['key_improvements']:
                print(f"  • {improvement}")

        else:
            print("\n⚠️ Configuration needs adjustment")
            print("Check your Ollama setup and model availability")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("Please ensure Ollama is running and your model is available")