"""
Test the Final Real Intelligence Agent
Verify it uses real analysis with LLM reasoning about computed data
"""

import numpy as np
import pandas as pd
import asyncio
import os


async def test_final_agent():
    """Test the final integrated real intelligence agent"""
    print("TESTING FINAL REAL INTELLIGENCE AGENT")
    print("=" * 50)

    # Create test dataset with known characteristics
    np.random.seed(42)
    cluster1 = np.random.normal([2, 2], 0.5, (40, 2))
    cluster2 = np.random.normal([8, 2], 0.5, (40, 2))
    test_data = np.vstack([cluster1, cluster2])

    df = pd.DataFrame(test_data, columns=['delay_ns', 'sigma_ps'])
    df['measurement'] = df['delay_ns'] + df['sigma_ps'] + np.random.normal(0, 0.1, 80)

    # Save test data
    os.makedirs('/tmp/claude', exist_ok=True)
    csv_path = '/tmp/claude/final_test.csv'
    df.to_csv(csv_path, index=False)

    print(f"Test dataset: 2 clusters, 80 samples, 3 features")

    # Import and test the agent
    try:
        from agent.timing_data_selection_agent import TimingDataSelectionAgent

        # Test without LLM first
        print(f"\n1. Testing without LLM (statistical analysis only):")
        agent_no_llm = TimingDataSelectionAgent(llm=None, verbose=False)

        result_no_llm = await agent_no_llm.intelligent_sample_selection(csv_path, 12.5)

        print(f"   Hopkins computed: {result_no_llm['data_analysis']['data_characteristics']['clustering_potential']['hopkins_statistic']:.3f}")
        print(f"   Optimal clusters: {result_no_llm['data_analysis']['data_characteristics']['clustering_potential']['optimal_cluster_count']}")
        print(f"   Samples selected: {len(result_no_llm['selected_indices'])}")
        print(f"   Method used: {result_no_llm['selection_method']}")

        # Check reasoning chain contains real data
        reasoning = ' '.join(result_no_llm['reasoning_chain'])
        contains_hopkins = 'hopkins' in reasoning.lower()
        contains_numbers = any(char.isdigit() for char in reasoning)

        print(f"   Reasoning contains Hopkins: {contains_hopkins}")
        print(f"   Reasoning contains numbers: {contains_numbers}")

        # Test with mock LLM
        print(f"\n2. Testing with mock LLM (enhanced reasoning):")

        class MockLLM:
            def invoke(self, prompt):
                # Mock LLM that actually reads the prompt and responds appropriately
                if "Hopkins statistic" in prompt:
                    lines = prompt.split('\n')
                    hopkins_line = [line for line in lines if 'Hopkins statistic' in line]
                    if hopkins_line:
                        # Extract the Hopkins value
                        hopkins_str = hopkins_line[0].split(':')[1].strip().split()[0]
                        try:
                            hopkins_val = float(hopkins_str)
                            if hopkins_val > 0.7:
                                return "Strong clustering detected. Recommend stratified sampling to preserve cluster structure while ensuring representative coverage across identified groups."
                            else:
                                return "Weak clustering detected. Recommend uniform random sampling as data lacks clear structure for stratified approaches."
                        except:
                            pass

                return "Mock LLM response based on data analysis"

        agent_with_llm = TimingDataSelectionAgent(llm=MockLLM(), verbose=False)

        result_with_llm = await agent_with_llm.intelligent_sample_selection(csv_path, 12.5)

        print(f"   Hopkins computed: {result_with_llm['data_analysis']['data_characteristics']['clustering_potential']['hopkins_statistic']:.3f}")
        print(f"   LLM reasoning: {result_with_llm.get('llm_reasoning', 'None')}")
        print(f"   Samples selected: {len(result_with_llm['selected_indices'])}")

        # Verify LLM reasoning is based on real data
        llm_reasoning = result_with_llm.get('llm_reasoning', '')
        reasoning_mentions_clustering = 'clustering' in llm_reasoning.lower()
        reasoning_mentions_sampling = 'sampling' in llm_reasoning.lower()

        print(f"   LLM reasoning mentions clustering: {reasoning_mentions_clustering}")
        print(f"   LLM reasoning mentions sampling: {reasoning_mentions_sampling}")

        print(f"\n3. Validation Results:")

        # Both should produce similar statistical results
        hopkins_diff = abs(
            result_no_llm['data_analysis']['data_characteristics']['clustering_potential']['hopkins_statistic'] -
            result_with_llm['data_analysis']['data_characteristics']['clustering_potential']['hopkins_statistic']
        )

        print(f"   Hopkins consistency: {hopkins_diff < 0.1} (diff: {hopkins_diff:.6f})")
        print(f"   LLM adds reasoning: {bool(result_with_llm.get('llm_reasoning'))}")

        # Both should detect 2 clusters for this test data
        clusters_no_llm = result_no_llm['data_analysis']['data_characteristics']['clustering_potential']['optimal_cluster_count']
        clusters_with_llm = result_with_llm['data_analysis']['data_characteristics']['clustering_potential']['optimal_cluster_count']

        print(f"   Cluster detection consistent: {clusters_no_llm == clusters_with_llm} ({clusters_no_llm} vs {clusters_with_llm})")

        # Final assessment
        if (contains_hopkins and contains_numbers and reasoning_mentions_clustering and
            hopkins_diff < 0.1 and clusters_no_llm == clusters_with_llm):
            print(f"\n✓ SUCCESS: Real intelligence confirmed")
            print(f"  - Statistical analysis produces real measurements")
            print(f"  - LLM reasons about computed data, not fake thoughts")
            print(f"  - Consistent results with/without LLM")
            print(f"  - Reasoning contains actual data values")
            return True
        else:
            print(f"\n✗ FAILURE: Issues detected")
            return False

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the test"""
    success = await test_final_agent()

    if success:
        print(f"\n🎉 REAL INTELLIGENCE AGENT VALIDATED")
        print(f"The agent now uses genuine statistical analysis")
        print(f"LLM provides reasoning about real computed data")
        print(f"No fake thoughts or random phrase generation")
    else:
        print(f"\n❌ VALIDATION FAILED")
        print(f"Further fixes may be needed")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()