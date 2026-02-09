"""
Test script for the Autonomous Timing Data Selection Agent

This script tests the autonomous agent's capabilities:
1. Autonomous exploration
2. Hypothesis generation
3. Parallel experimentation
4. Self-validation
5. Strategy synthesis
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path

# Mock LLM for testing
class MockLLM:
    """Simple mock LLM for testing autonomous agent without external dependencies."""

    def invoke(self, inputs):
        # Return structured responses for different prompt types
        class MockResponse:
            def __init__(self, content):
                self.content = content

        # Simple responses to test the autonomous pipeline
        return MockResponse("""
I need to understand the dataset structure and patterns.

```python
print("Dataset columns:", dataset.columns.tolist())
print("Data shape:", dataset.shape)
print("Data types:")
print(dataset.dtypes)
print("\\nFirst 3 rows:")
print(dataset.head(3))
print("\\nBasic statistics:")
print(dataset.describe())
```
""")


def create_test_data(n_samples=1000):
    """Create synthetic timing data for testing."""
    np.random.seed(42)

    # Generate synthetic timing features
    data = {
        'delay': np.random.gamma(2, 2, n_samples),
        'sigma': np.random.exponential(0.5, n_samples),
        'slew': np.random.uniform(0.1, 10.0, n_samples),
        'load': np.random.uniform(0.01, 5.0, n_samples),
        'voltage': np.random.choice([0.8, 1.0, 1.2], n_samples),
        'temperature': np.random.choice([-40, 25, 125], n_samples)
    }

    # Add some cell identifier
    cell_types = ['INV', 'NAND', 'NOR', 'BUF', 'XOR']
    data['cell_arc_pt'] = [f"{np.random.choice(cell_types)}#A#Y_{i//100}_{i%100}"
                          for i in range(n_samples)]

    df = pd.DataFrame(data)
    return df


async def test_autonomous_agent():
    """Test the autonomous agent capabilities."""

    print("=" * 80)
    print("[TEST] AUTONOMOUS AGENT TESTING")
    print("=" * 80)

    # Create test data
    print("\n[DATA] Creating test dataset...")
    test_data = create_test_data(500)  # Smaller dataset for faster testing

    # Save to temporary CSV
    test_csv = '/tmp/claude/test_timing_data.csv'
    Path(test_csv).parent.mkdir(exist_ok=True)
    test_data.to_csv(test_csv, index=False)

    print(f"[SUCCESS] Test data created: {len(test_data)} samples")
    print(f"[FILE] Saved to: {test_csv}")

    # Initialize autonomous agent
    print("\n[AGENT] Initializing Autonomous Agent...")
    try:
        from agent.timing_data_selection_agent import TimingDataSelectionAgent

        # Create mock LLM
        mock_llm = MockLLM()

        # Initialize agent
        agent = TimingDataSelectionAgent(mock_llm, verbose=True)

        print("[SUCCESS] Agent initialized successfully")

        # Test autonomous sampling pipeline
        print(f"\n[START] Testing Autonomous Sampling Pipeline...")
        print(f"Target: 5% of {len(test_data)} samples = ~{len(test_data) * 0.05:.0f} samples")

        results = await agent.autonomous_sample_selection(test_csv, target_percentage=5.0)

        # Analyze results
        print(f"\n[DATA] AUTONOMOUS PIPELINE RESULTS:")
        print(f"[SUCCESS] Selected samples: {len(results.get('selected_indices', []))}")
        print(f"[TARGET] Final algorithm: {results.get('final_strategy', {}).get('algorithm', 'unknown')}")
        print(f"[PROGRESS] Quality score: {results.get('quality_assessment', {}).get('overall_quality', 0):.3f}")

        # Pipeline statistics
        stats = results.get('autonomous_pipeline_stats', {})
        print(f"\n[DATA] PIPELINE STATISTICS:")
        print(f"   Exploration iterations: {stats.get('total_exploration_iterations', 0)}")
        print(f"   Experiments run: {stats.get('experiments_run', 0)}")
        print(f"   Validations passed: {stats.get('validations_passed', 0)}")
        print(f"   Pipeline quality: {stats.get('pipeline_quality', 0):.3f}")

        # Test specific autonomous capabilities
        print(f"\n[THINK] TESTING AUTONOMOUS CAPABILITIES:")

        # Test exploration engine directly
        exploration_results = await agent.exploration_engine.autonomous_explore(test_data, 5.0)
        print(f"[SUCCESS] Exploration engine: {exploration_results.get('total_iterations', 0)} iterations")

        # Test hypothesis generation
        hypotheses = await agent._generate_autonomous_hypotheses(exploration_results)
        print(f"[SUCCESS] Hypothesis generation: {len(hypotheses)} hypotheses generated")

        # Test experiment execution
        experiment_results = await agent.experiment_executor.execute_experiments_parallel(hypotheses[:2])  # Test with 2
        print(f"[SUCCESS] Parallel experiments: {len(experiment_results)} experiments completed")

        print(f"\n[DONE] ALL AUTONOMOUS TESTS PASSED!")
        return True

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        print(f"Stack trace:")
        traceback.print_exc()
        return False


def test_autonomous_components():
    """Test individual autonomous components."""

    print("\n🔧 TESTING INDIVIDUAL COMPONENTS:")

    try:
        from agent.autonomous_prompts import (
            get_autonomous_exploration_prompt,
            get_autonomous_hypothesis_prompt,
            autonomous_prompt_generator
        )

        # Test prompt generation
        context = {
            'iteration': 1,
            'knowledge_gaps': ['What are the key features?'],
            'discoveries': {'schema': 'basic discovery'},
            'data_size': 500
        }

        exploration_prompt = get_autonomous_exploration_prompt(context)
        print(f"[SUCCESS] Exploration prompt generated: {len(exploration_prompt)} chars")

        hypothesis_prompt = get_autonomous_hypothesis_prompt(
            {'statistical_analysis': 'test discovery'},
            {'exploration_quality': 0.8}
        )
        print(f"[SUCCESS] Hypothesis prompt generated: {len(hypothesis_prompt)} chars")

        print(f"[SUCCESS] All prompt components working")

        return True

    except Exception as e:
        print(f"[ERROR] Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""

    print("[START] Starting Autonomous Agent Test Suite...")

    # Test components first
    component_success = test_autonomous_components()

    if component_success:
        # Test full autonomous agent
        agent_success = await test_autonomous_agent()

        if agent_success:
            print(f"\n[DONE] ALL TESTS SUCCESSFUL!")
            print(f"[AGENT] Autonomous agent is ready for intelligent sampling!")
        else:
            print(f"\n[ERROR] Agent tests failed")
    else:
        print(f"\n[ERROR] Component tests failed")


if __name__ == "__main__":
    # Run tests
    asyncio.run(main())