#!/usr/bin/env python3
"""
Test script for the integrated agentic timing agent
Tests intent classification and conversation handling
"""

import sys
import os

def test_intent_classification():
    """Test intent classification with various inputs."""
    print("Testing Intent Classification...")

    try:
        from agent.timing_data_selection_agent import TimingDataSelectionAgent, UserIntent
        from unittest.mock import Mock

        # Mock LLM for testing
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="test response"))

        agent = TimingDataSelectionAgent(llm=mock_llm, verbose=False)

        # Test cases
        test_cases = [
            ("why not k-means?", UserIntent.QUESTION_ABOUT_RESULTS),
            ("Select 5% of timing data", UserIntent.EXECUTE_SAMPLING),
            ("Change to 8%", UserIntent.MODIFY_PARAMETERS),
            ("show dashboard", UserIntent.REQUEST_VISUALIZATION),
            ("How does clustering work?", UserIntent.EXPLAIN_METHODOLOGY),
            ("help", UserIntent.GENERAL_HELP),
            ("why did you choose 5%?", UserIntent.QUESTION_ABOUT_RESULTS),
        ]

        success_count = 0
        for test_input, expected_intent in test_cases:
            try:
                actual_intent, params = agent.classify_user_intent(test_input)
                if actual_intent == expected_intent:
                    print(f"✓ '{test_input}' -> {actual_intent.value}")
                    success_count += 1
                else:
                    print(f"✗ '{test_input}' -> Expected: {expected_intent.value}, Got: {actual_intent.value}")
            except Exception as e:
                print(f"✗ '{test_input}' -> Error: {e}")

        print(f"\nIntent Classification: {success_count}/{len(test_cases)} tests passed")
        return success_count == len(test_cases)

    except Exception as e:
        print(f"Error setting up intent classification test: {e}")
        return False

def test_enhanced_prompts():
    """Test that enhanced prompts are properly formatted."""
    print("\nTesting Enhanced Prompts...")

    try:
        from agent.timing_data_selection_agent import (
            AGENTIC_EXPLORE_PROMPT,
            AGENTIC_STRATEGY_PROMPT,
            AGENTIC_EXECUTE_PROMPT,
            AGENTIC_LLM_PARAMETERS,
            VALIDATION_BOUNDARIES
        )

        # Test prompt parameter placeholders
        test_params = {
            'total_samples': 10000,
            'target_count': 500,
            'target_percentage': 5.0,
            'n_features': 12,
            'n_cell_types': 8,
            'calculated_stats': 'test stats',
            'correlation_details': 'test correlations',
            'sigma_analysis': 'test sigma'
        }

        success_count = 0

        # Test AGENTIC_EXPLORE_PROMPT
        try:
            formatted = AGENTIC_EXPLORE_PROMPT.format(**test_params)
            if "10000" in formatted and "500" in formatted:
                print("✓ AGENTIC_EXPLORE_PROMPT formatting works")
                success_count += 1
            else:
                print("✗ AGENTIC_EXPLORE_PROMPT formatting failed")
        except Exception as e:
            print(f"✗ AGENTIC_EXPLORE_PROMPT error: {e}")

        # Test AGENTIC_STRATEGY_PROMPT
        try:
            strategy_params = {
                'exploration_findings': 'test findings',
                'target_count': 500
            }
            formatted = AGENTIC_STRATEGY_PROMPT.format(**strategy_params)
            if "500" in formatted:
                print("✓ AGENTIC_STRATEGY_PROMPT formatting works")
                success_count += 1
            else:
                print("✗ AGENTIC_STRATEGY_PROMPT formatting failed")
        except Exception as e:
            print(f"✗ AGENTIC_STRATEGY_PROMPT error: {e}")

        # Test enhanced parameters
        if 'temperature' in AGENTIC_LLM_PARAMETERS and 'minimum_cell_type_coverage' in VALIDATION_BOUNDARIES:
            print("✓ Enhanced parameters and boundaries loaded")
            success_count += 1
        else:
            print("✗ Enhanced parameters missing")

        print(f"Enhanced Prompts: {success_count}/3 tests passed")
        return success_count == 3

    except Exception as e:
        print(f"Error testing enhanced prompts: {e}")
        return False

def test_conversation_handling():
    """Test conversation handling logic without LLM dependencies."""
    print("\nTesting Conversation Handling...")

    try:
        from agent.timing_data_selection_agent import TimingDataSelectionAgent, UserIntent
        from unittest.mock import Mock

        # Mock LLM
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="Mocked LLM response explaining timing methodology"))

        agent = TimingDataSelectionAgent(llm=mock_llm, verbose=False)

        # Test intent classification for conversation vs execution
        conversation_queries = [
            "why not k-means?",
            "explain your reasoning",
            "how does clustering work?"
        ]

        execution_queries = [
            "select 5% of data",
            "run analysis",
            "change to 8%"
        ]

        conversation_intents = {UserIntent.QUESTION_ABOUT_RESULTS, UserIntent.EXPLAIN_METHODOLOGY}
        execution_intents = {UserIntent.EXECUTE_SAMPLING, UserIntent.MODIFY_PARAMETERS}

        success_count = 0

        # Test conversation queries
        for query in conversation_queries:
            intent, params = agent.classify_user_intent(query)
            if intent in conversation_intents:
                print(f"✓ '{query}' correctly classified as conversational: {intent.value}")
                success_count += 1
            else:
                print(f"✗ '{query}' misclassified as: {intent.value}")

        # Test execution queries
        for query in execution_queries:
            intent, params = agent.classify_user_intent(query)
            if intent in execution_intents:
                print(f"✓ '{query}' correctly classified as execution: {intent.value}")
                success_count += 1
            else:
                print(f"✗ '{query}' misclassified as: {intent.value}")

        total_tests = len(conversation_queries) + len(execution_queries)
        print(f"Conversation Logic: {success_count}/{total_tests} classifications correct")
        return success_count >= total_tests * 0.8  # 80% success rate

    except Exception as e:
        print(f"Error testing conversation handling: {e}")
        return False

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("AGENTIC TIMING AGENT - INTEGRATION TESTS")
    print("=" * 60)

    tests = [
        test_intent_classification,
        test_enhanced_prompts,
        test_conversation_handling
    ]

    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1

    print(f"\n{'=' * 60}")
    print(f"INTEGRATION TEST RESULTS: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("✓ All integration tests PASSED - System ready!")
    else:
        print(f"✗ {len(tests) - passed} tests FAILED - Review issues above")

    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)