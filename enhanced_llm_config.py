"""
Enhanced LLM Configuration for Timing Domain Expertise
Optimized parameters for consistent senior engineer level reasoning
"""

import os
from langchain_community.llms import Ollama
from langchain.llms.base import LLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_enhanced_ollama_llm() -> LLM:
    """
    Initialize Ollama LLM with enhanced parameters for timing domain expertise.

    Optimized for:
    - Consistent technical reasoning
    - Reduced generic responses
    - Better domain terminology usage
    - More detailed explanations

    Environment variables (enhanced defaults):
    - OLLAMA_BASE_URL: http://localhost:11434
    - OLLAMA_MODEL: qwen2.5:32b-instruct (recommended for technical domains)
    - LLM_TEMPERATURE: 0.1-0.3 (lower for consistency)
    - LLM_TOP_P: 0.85-0.95 (focused sampling)
    - LLM_TOP_K: 30-50 (reduced for technical precision)
    - LLM_NUM_PREDICT: 2000-3000 (more tokens for detailed explanations)
    """

    # Enhanced configuration for timing domain
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    model = os.getenv('OLLAMA_MODEL', 'qwen2.5:32b-instruct')

    # Optimized parameters for technical consistency
    temperature = float(os.getenv('LLM_TEMPERATURE', '0.2'))  # Lower for consistency
    top_p = float(os.getenv('LLM_TOP_P', '0.9'))             # Focused sampling
    top_k = int(os.getenv('LLM_TOP_K', '40'))                # Reduced for precision
    num_predict = int(os.getenv('LLM_NUM_PREDICT', '2500'))  # More tokens for detail
    repeat_penalty = float(os.getenv('LLM_REPEAT_PENALTY', '1.1'))  # Avoid repetition

    # Advanced parameters for enhanced reasoning
    presence_penalty = float(os.getenv('LLM_PRESENCE_PENALTY', '0.1'))
    frequency_penalty = float(os.getenv('LLM_FREQUENCY_PENALTY', '0.1'))

    logger.info(f"🦙 Enhanced Ollama Configuration for Timing Domain:")
    logger.info(f"   Base URL: {base_url}")
    logger.info(f"   Model: {model}")
    logger.info(f"   Temperature: {temperature} (optimized for consistency)")
    logger.info(f"   Top-P: {top_p} (focused sampling)")
    logger.info(f"   Top-K: {top_k} (precision tuned)")
    logger.info(f"   Max Tokens: {num_predict} (detailed explanations)")
    logger.info(f"   Repeat Penalty: {repeat_penalty}")

    # Initialize Enhanced Ollama LLM
    llm = Ollama(
        base_url=base_url,
        model=model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_predict=num_predict,
        repeat_penalty=repeat_penalty,
        verbose=True,
        # Additional parameters for enhanced reasoning
        options={
            'num_thread': os.cpu_count(),  # Use all CPU cores
            'num_gpu': 1,                  # Use GPU if available
            'low_vram': False,             # Use full VRAM for better performance
            'f16_kv': True,                # Use fp16 for key-value cache (faster)
            'use_mlock': True,             # Lock model in memory
            'use_mmap': True,              # Memory map model file
            'numa': False                  # Disable NUMA for better performance
        }
    )

    logger.info(f"✓ Enhanced Ollama LLM initialized for timing domain expertise")

    return llm


def get_timing_domain_system_parameters():
    """
    Get recommended system parameters for timing domain expertise.

    Returns:
        dict: Optimized parameter configuration
    """
    return {
        'model_selection': {
            'recommended_models': [
                'qwen2.5:32b-instruct',      # Best for technical domains
                'qwen2.5:14b-instruct',      # Good alternative if memory limited
                'mixtral:8x7b-instruct',     # Alternative architecture
                'codellama:34b-instruct'     # Strong coding/technical reasoning
            ],
            'minimum_parameters': '14B',
            'optimal_parameters': '32B+'
        },

        'generation_parameters': {
            'temperature': {
                'value': 0.2,
                'range': '0.1-0.3',
                'rationale': 'Lower temperature for consistent technical reasoning'
            },
            'top_p': {
                'value': 0.9,
                'range': '0.85-0.95',
                'rationale': 'Focused nucleus sampling for domain expertise'
            },
            'top_k': {
                'value': 40,
                'range': '30-50',
                'rationale': 'Reduced top-k for technical precision'
            },
            'max_tokens': {
                'value': 2500,
                'range': '2000-3000',
                'rationale': 'More tokens for detailed technical explanations'
            },
            'repeat_penalty': {
                'value': 1.1,
                'range': '1.05-1.15',
                'rationale': 'Prevent repetitive generic phrases'
            }
        },

        'prompt_engineering': {
            'system_prompt_length': '800-1200 tokens',
            'context_window_usage': '60-80%',
            'few_shot_examples': 'Recommended for domain concepts',
            'structured_output': 'Use for consistency'
        },

        'quality_targets': {
            'specific_numbers_cited': '3+ per response',
            'domain_concepts_used': '5+ timing terms',
            'active_learning_explained': 'Required for THINK/ACT stages',
            'business_impact_mentioned': 'Required for strategic reasoning',
            'expertise_level': 'SENIOR_ENGINEER minimum'
        }
    }


def validate_model_capabilities(llm: LLM) -> Dict[str, bool]:
    """
    Validate if the LLM model can handle timing domain tasks.

    Args:
        llm: Initialized LLM instance

    Returns:
        dict: Capability assessment
    """
    test_prompt = """You are a senior timing engineer. Explain in 2 sentences why uncertainty sampling
    (selecting samples far from cluster centroids) is better than random sampling for Monte Carlo
    timing characterization. Include specific technical reasoning."""

    try:
        response = llm.invoke(test_prompt)
        response_text = response if isinstance(response, str) else response.content

        # Basic capability checks
        capabilities = {
            'responds_to_prompts': len(response_text) > 50,
            'uses_technical_terms': any(term in response_text.lower()
                                      for term in ['uncertainty', 'sampling', 'monte carlo', 'characterization']),
            'provides_reasoning': 'because' in response_text.lower() or 'since' in response_text.lower(),
            'sufficient_length': len(response_text) > 100,
            'domain_awareness': any(term in response_text.lower()
                                  for term in ['timing', 'model', 'accuracy', 'robustness'])
        }

        logger.info(f"Model capability assessment:")
        for capability, passed in capabilities.items():
            status = "✓" if passed else "✗"
            logger.info(f"  {status} {capability.replace('_', ' ').title()}")

        return capabilities

    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return {
            'responds_to_prompts': False,
            'uses_technical_terms': False,
            'provides_reasoning': False,
            'sufficient_length': False,
            'domain_awareness': False
        }


def optimize_for_timing_domain():
    """
    Apply timing domain specific optimizations.

    Returns:
        dict: Optimization recommendations
    """
    return {
        'environment_setup': {
            'OLLAMA_FLASH_ATTENTION': 'true',
            'OLLAMA_NUM_PARALLEL': '4',
            'OLLAMA_MAX_LOADED_MODELS': '1',
            'OLLAMA_KEEP_ALIVE': '10m'
        },

        'memory_optimization': {
            'recommended_ram': '32GB+ for 32B models',
            'gpu_memory': '24GB+ VRAM recommended',
            'swap_usage': 'Minimize for consistent performance'
        },

        'prompt_caching': {
            'cache_system_prompt': True,
            'cache_common_patterns': True,
            'invalidate_on_parameter_change': True
        },

        'monitoring': {
            'track_response_quality': True,
            'monitor_consistency': True,
            'validate_domain_expertise': True,
            'measure_token_efficiency': True
        }
    }


def create_enhanced_env_template():
    """
    Create .env template for enhanced timing domain configuration.

    Returns:
        str: Environment file template
    """
    return """# Enhanced Ollama Configuration for Timing Domain AI
# Copy this to .env and adjust values as needed

# === OLLAMA SERVER SETTINGS ===
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:32b-instruct

# === ENHANCED GENERATION PARAMETERS ===
# Temperature: Lower for consistent technical reasoning
LLM_TEMPERATURE=0.2

# Top-P: Focused nucleus sampling for domain expertise
LLM_TOP_P=0.9

# Top-K: Reduced for technical precision
LLM_TOP_K=40

# Max Tokens: More tokens for detailed explanations
LLM_NUM_PREDICT=2500

# Repeat Penalty: Prevent generic phrases
LLM_REPEAT_PENALTY=1.1

# === PERFORMANCE OPTIMIZATION ===
OLLAMA_FLASH_ATTENTION=true
OLLAMA_NUM_PARALLEL=4
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_KEEP_ALIVE=10m

# === QUALITY TARGETS ===
# Minimum expertise level to achieve
TARGET_EXPERTISE_LEVEL=SENIOR_ENGINEER

# Quality thresholds
MIN_SPECIFIC_NUMBERS=3
MIN_DOMAIN_CONCEPTS=5
REQUIRE_ACTIVE_LEARNING=true
REQUIRE_BUSINESS_IMPACT=true

# === MONITORING ===
ENABLE_REASONING_VALIDATION=true
ENABLE_QUALITY_METRICS=true
LOG_LEVEL=INFO
"""


if __name__ == "__main__":
    # Test enhanced configuration
    print("=== Enhanced LLM Configuration Test ===")

    try:
        llm = initialize_enhanced_ollama_llm()
        print("✓ Enhanced LLM initialized successfully")

        capabilities = validate_model_capabilities(llm)
        all_passed = all(capabilities.values())

        if all_passed:
            print("✓ All capability checks passed - Ready for timing domain tasks")
        else:
            print("⚠️ Some capability checks failed - Consider model upgrade")

        # Display optimization recommendations
        optimizations = optimize_for_timing_domain()
        print("\n=== Optimization Recommendations ===")
        for category, settings in optimizations.items():
            print(f"{category.replace('_', ' ').title()}:")
            for key, value in settings.items():
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")

    # Generate .env template
    print("\n=== Environment Template ===")
    template = create_enhanced_env_template()
    with open('enhanced_env_template.txt', 'w') as f:
        f.write(template)
    print("✓ Enhanced .env template saved to 'enhanced_env_template.txt'")