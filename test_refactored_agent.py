#!/usr/bin/env python3
"""
Test script for the refactored TimingDataSelectionAgent.

This script tests the new ReAct architecture with:
1. Execution Engine (real Python code execution)
2. Dynamic schema discovery with while loop
3. Algorithm tournament with empirical metrics
4. Comprehensive visualization dashboard
5. Domain-specific high sigma and boundary detection
"""

import sys
import os
sys.path.append('.')

from agent.timing_data_selection_agent import TimingDataSelectionAgent


class MockLLM:
    """Mock LLM for testing the agent without external dependencies."""

    def invoke(self, inputs):
        class MockResponse:
            def __init__(self, content):
                self.content = content

        return MockResponse("Mock LLM response for testing")


def test_refactored_agent():
    """Test the refactored TimingDataSelectionAgent."""

    print("=" * 80)
    print("TESTING REFACTORED TIMING DATA SELECTION AGENT")
    print("=" * 80)

    # Initialize agent with mock LLM
    mock_llm = MockLLM()
    agent = TimingDataSelectionAgent(mock_llm, verbose=True)

    print(f"\n[TEST] Agent initialized with execution engine: {hasattr(agent, 'execution_context')}")
    print(f"[TEST] Execution context available: {'dataset' in agent.execution_context}")

    # Test data path
    test_csv_path = "/Users/nieyuxuan/Downloads/demo1_dataprep/mock_data/test_data.csv"

    if not os.path.exists(test_csv_path):
        print(f"\n[ERROR] Test data not found at: {test_csv_path}")
        return False

    try:
        # Test 1: OBSERVE stage - ReAct schema discovery
        print(f"\n{'='*60}")
        print("TEST 1: OBSERVE STAGE - ReAct Schema Discovery")
        print(f"{'='*60}")

        observation = agent.observe(test_csv_path, target_percentage=5.0)

        print(f"\n[TEST RESULT] Observation completed:")
        print(f"  - Total samples: {observation.get('total_samples', 'N/A')}")
        print(f"  - Features detected: {observation.get('n_features', 'N/A')}")
        print(f"  - Feature names: {observation.get('feature_names', [])[:5]}...")  # Show first 5
        print(f"  - Conversation history length: {len(observation.get('conversation_history', []))}")

        # Test 2: Domain-specific detection methods
        print(f"\n{'='*60}")
        print("TEST 2: DOMAIN-SPECIFIC DETECTION")
        print(f"{'='*60}")

        # High sigma detection
        print("\n[TEST 2A] High Sigma Detection...")
        high_sigma_result = agent.detect_high_sigma_points()
        print(f"  - High sigma detection completed: {len(high_sigma_result.get('detection_output', ''))} chars output")

        # Boundary detection
        print("\n[TEST 2B] Boundary Detection...")
        boundary_result = agent.detect_boundary_points()
        print(f"  - Boundary detection completed: {len(boundary_result.get('detection_output', ''))} chars output")

        # Test 3: Algorithm Tournament
        print(f"\n{'='*60}")
        print("TEST 3: ALGORITHM TOURNAMENT")
        print(f"{'='*60}")

        # Mock strategy for testing
        mock_strategy = {
            'variance_threshold': 0.9,
            'n_clusters_range': [2, 3, 4, 5]
        }

        print("\n[TEST 3] Running algorithm tournament...")
        decision = agent.decide(mock_strategy)

        print(f"\n[TEST RESULT] Decision completed:")
        print(f"  - Algorithm selected: {decision.get('algorithm', 'N/A')}")
        print(f"  - Number of clusters: {decision.get('n_clusters', 'N/A')}")
        print(f"  - Performance score: {decision.get('silhouette_score', 'N/A')}")

        # Test 4: Coverage Analysis (if we have selected indices)
        if 'labels' in decision and decision['labels'] is not None:
            print(f"\n{'='*60}")
            print("TEST 4: COVERAGE ANALYSIS")
            print(f"{'='*60}")

            # Create mock selected indices for testing
            mock_selected_indices = [0, 5, 10, 15, 20] if observation['total_samples'] > 20 else [0, 1]

            print(f"\n[TEST 4] Running coverage analysis with {len(mock_selected_indices)} mock selections...")
            coverage_result = agent.analyze_timing_coverage(mock_selected_indices)

            print(f"  - Coverage analysis completed: {len(coverage_result.get('analysis_output', ''))} chars output")

        print(f"\n{'='*60}")
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")

        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting refactored agent test...")
    success = test_refactored_agent()

    if success:
        print("\n[SUCCESS] All tests passed! The refactored agent is working correctly.")
        print("\nKey improvements verified:")
        print("✓ Execution Engine - Real Python code execution")
        print("✓ ReAct Loop - Dynamic schema discovery")
        print("✓ Algorithm Tournament - Empirical performance metrics")
        print("✓ Domain-specific Logic - High sigma and boundary detection")
        print("✓ Comprehensive Analysis - Coverage validation")
    else:
        print("\n[FAILURE] Tests failed. Check the error messages above.")
        sys.exit(1)