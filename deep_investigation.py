"""
Deep Investigation: Is the Agent Actually Running Real Analysis?
This will catch any remaining fake intelligence that appears real but isn't
"""

import numpy as np
import pandas as pd
import asyncio
import time
import inspect
from agent.real_timing_agent import RealTimingDataSelectionAgent
from agent.real_intelligence_engine import DataCharacteristicsAnalyzer


def investigate_actual_execution():
    """Investigate if analysis methods are actually executing real computations"""
    print("🔍 DEEP INVESTIGATION: ACTUAL EXECUTION VERIFICATION")
    print("=" * 70)

    # Create test data where we can verify computations
    np.random.seed(12345)
    test_data = np.random.normal([0, 0], 1, (100, 2))
    df = pd.DataFrame(test_data, columns=['x', 'y'])
    df['timing'] = df['x'] + df['y'] + np.random.normal(0, 0.1, 100)

    print(f"📊 Test Dataset Created: {len(df)} samples, {len(df.columns)} features")

    analyzer = DataCharacteristicsAnalyzer()

    # Test 1: Check if Hopkins statistic is actually computed
    print(f"\n🧮 TEST 1: Hopkins Statistic Computation")
    print("-" * 50)

    characteristics1 = analyzer.analyze_dataset(df)
    hopkins1 = characteristics1['clustering_potential']['hopkins_statistic']

    # Run again - should be identical for same data
    characteristics2 = analyzer.analyze_dataset(df)
    hopkins2 = characteristics2['clustering_potential']['hopkins_statistic']

    print(f"Run 1 Hopkins: {hopkins1}")
    print(f"Run 2 Hopkins: {hopkins2}")
    print(f"Difference: {abs(hopkins1 - hopkins2)}")

    if abs(hopkins1 - hopkins2) < 1e-10:
        print(f"✅ Hopkins computation appears deterministic (good)")
    else:
        print(f"⚠ Hopkins varies between runs - may be using random seed internally")

    # Test 2: Manual Hopkins calculation to verify
    print(f"\n🧮 TEST 2: Manual Hopkins Verification")
    print("-" * 50)

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors

        X = StandardScaler().fit_transform(df.select_dtypes(include=[np.number]))
        n_samples = min(50, len(X)//10)

        # Manual Hopkins calculation
        sample_indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[sample_indices]

        # Random points in data space
        random_points = np.random.uniform(X.min(), X.max(), (n_samples, X.shape[1]))

        # Distances to nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=2).fit(X)
        u_distances = nbrs.kneighbors(random_points)[0][:, 1]
        w_distances = nbrs.kneighbors(X_sample)[0][:, 1]

        manual_hopkins = u_distances.sum() / (u_distances.sum() + w_distances.sum())

        print(f"Agent Hopkins: {hopkins1:.6f}")
        print(f"Manual Hopkins: {manual_hopkins:.6f}")
        print(f"Difference: {abs(hopkins1 - manual_hopkins):.6f}")

        if abs(hopkins1 - manual_hopkins) < 0.2:  # Allow some variation due to randomness
            print(f"✅ Hopkins calculation appears legitimate")
        else:
            print(f"❌ Hopkins calculation may be fake or using different method")

    except Exception as e:
        print(f"❌ Manual Hopkins calculation failed: {e}")

    # Test 3: Check correlation computation
    print(f"\n🧮 TEST 3: Correlation Matrix Verification")
    print("-" * 50)

    corr_analysis = characteristics1['correlation_analysis']
    agent_corr_matrix = corr_analysis['correlation_matrix']

    # Manual correlation calculation
    manual_corr = df.select_dtypes(include=[np.number]).corr().to_dict()

    print(f"Agent correlation x↔y: {agent_corr_matrix.get('x', {}).get('y', 'N/A')}")
    print(f"Manual correlation x↔y: {manual_corr.get('x', {}).get('y', 'N/A')}")

    if 'x' in agent_corr_matrix and 'y' in agent_corr_matrix['x']:
        diff = abs(agent_corr_matrix['x']['y'] - manual_corr['x']['y'])
        print(f"Difference: {diff:.6f}")
        if diff < 1e-10:
            print(f"✅ Correlation calculation appears correct")
        else:
            print(f"❌ Correlation calculation may be incorrect")
    else:
        print(f"❌ Correlation matrix structure is wrong")


async def investigate_pipeline_execution():
    """Check if the full pipeline actually executes different analyses"""
    print(f"\n🔍 PIPELINE EXECUTION INVESTIGATION")
    print("=" * 70)

    # Create CSV file
    import os
    os.makedirs('/tmp/claude', exist_ok=True)

    # Dataset with very specific characteristics
    np.random.seed(777)
    cluster1 = np.random.normal([10, 10], 0.5, (30, 2))
    cluster2 = np.random.normal([20, 10], 0.5, (30, 2))
    data = np.vstack([cluster1, cluster2])
    df = pd.DataFrame(data, columns=['delay_ps', 'sigma_ps'])

    csv_path = '/tmp/claude/investigation.csv'
    df.to_csv(csv_path, index=False)

    print(f"📊 Test Dataset: 2 clusters at [10,10] and [20,10]")

    agent = RealTimingDataSelectionAgent(verbose=True)

    # Run analysis and capture all output
    print(f"\n🏃 Running Full Pipeline...")
    start_time = time.time()

    result = await agent.intelligent_sample_selection(csv_path, 10.0)
    execution_time = time.time() - start_time

    print(f"⏱ Total execution time: {execution_time:.2f} seconds")

    # Analyze the results
    print(f"\n📋 RESULT ANALYSIS:")
    print("-" * 30)

    data_analysis = result.get('data_analysis', {})
    characteristics = data_analysis.get('data_characteristics', {})

    if characteristics:
        # Check basic stats
        basic_stats = characteristics.get('basic_stats', {})
        print(f"Reported samples: {basic_stats.get('n_samples', 'N/A')}")
        print(f"Actual samples: {len(df)}")
        print(f"Reported features: {basic_stats.get('n_features', 'N/A')}")
        print(f"Actual features: {len(df.columns)}")

        # Check clustering results
        clustering = characteristics.get('clustering_potential', {})
        print(f"Reported Hopkins: {clustering.get('hopkins_statistic', 'N/A')}")
        print(f"Reported clusters: {clustering.get('optimal_cluster_count', 'N/A')}")

        # Check if it correctly identified 2 clusters
        optimal_k = clustering.get('optimal_cluster_count', 0)
        if optimal_k == 2:
            print(f"✅ Correctly identified 2 clusters")
        else:
            print(f"⚠ Identified {optimal_k} clusters, expected 2")

    else:
        print(f"❌ No data characteristics found in results")

    # Check reasoning chain for specific mentions
    reasoning_chain = result.get('reasoning_chain', [])
    print(f"\n📝 REASONING ANALYSIS:")
    print("-" * 30)

    for i, reason in enumerate(reasoning_chain):
        print(f"{i+1}. {reason}")

    # Check for actual computed values in reasoning
    reasoning_text = ' '.join(reasoning_chain)
    contains_delay_ps = 'delay_ps' in reasoning_text
    contains_sigma_ps = 'sigma_ps' in reasoning_text
    contains_specific_numbers = any(str(round(val, 3)) in reasoning_text for val in [10.0, 20.0])

    print(f"\nReasoning specificity check:")
    print(f"  Mentions actual feature names: {contains_delay_ps or contains_sigma_ps}")
    print(f"  Mentions specific values: {contains_specific_numbers}")

    return result


def investigate_iteration_behavior():
    """Check if iterations actually differ or just repeat"""
    print(f"\n🔄 ITERATION BEHAVIOR INVESTIGATION")
    print("=" * 70)

    analyzer = DataCharacteristicsAnalyzer()

    # Create different datasets
    datasets = {}

    # Dataset 1: High clustering
    np.random.seed(100)
    high_cluster = np.vstack([
        np.random.normal([0, 0], 0.5, (50, 2)),
        np.random.normal([10, 10], 0.5, (50, 2))
    ])
    datasets['high_clustering'] = pd.DataFrame(high_cluster, columns=['a', 'b'])

    # Dataset 2: Low clustering
    np.random.seed(200)
    low_cluster = np.random.uniform(-5, 5, (100, 2))
    datasets['low_clustering'] = pd.DataFrame(low_cluster, columns=['c', 'd'])

    print(f"Testing if agent produces different results for different data...")

    results = {}
    for name, df in datasets.items():
        characteristics = analyzer.analyze_dataset(df)
        hopkins = characteristics['clustering_potential']['hopkins_statistic']
        optimal_k = characteristics['clustering_potential']['optimal_cluster_count']
        outlier_ratio = characteristics['outlier_analysis']['global_outlier_ratio']

        results[name] = {
            'hopkins': hopkins,
            'optimal_k': optimal_k,
            'outlier_ratio': outlier_ratio
        }

        print(f"\n{name}:")
        print(f"  Hopkins: {hopkins:.6f}")
        print(f"  Optimal k: {optimal_k}")
        print(f"  Outlier ratio: {outlier_ratio:.6f}")

    # Check if results are actually different
    high_hopkins = results['high_clustering']['hopkins']
    low_hopkins = results['low_clustering']['hopkins']
    hopkins_diff = abs(high_hopkins - low_hopkins)

    print(f"\nDifference Analysis:")
    print(f"  Hopkins difference: {hopkins_diff:.6f}")

    if hopkins_diff > 0.1:
        print(f"✅ Agent produces different Hopkins values for different data")
    else:
        print(f"❌ Hopkins values too similar - may be fake or broken")

    high_k = results['high_clustering']['optimal_k']
    low_k = results['low_clustering']['optimal_k']

    if high_k != low_k:
        print(f"✅ Agent produces different cluster counts ({high_k} vs {low_k})")
    else:
        print(f"⚠ Same cluster count for different data patterns")


def check_for_hardcoded_responses():
    """Look for hardcoded or templated responses"""
    print(f"\n🎭 HARDCODED RESPONSE INVESTIGATION")
    print("=" * 70)

    from agent.real_intelligence_engine import IntelligentReasoningEngine

    reasoning_engine = IntelligentReasoningEngine()

    # Create very different datasets
    test_datasets = []

    # Test 1: All zeros
    df1 = pd.DataFrame(np.zeros((50, 3)), columns=['x', 'y', 'z'])
    test_datasets.append(('zeros', df1))

    # Test 2: Linear pattern
    x = np.linspace(0, 10, 50)
    df2 = pd.DataFrame({'a': x, 'b': x*2, 'c': x*3})
    test_datasets.append(('linear', df2))

    # Test 3: Random
    df3 = pd.DataFrame(np.random.random((50, 2)), columns=['p', 'q'])
    test_datasets.append(('random', df3))

    responses = []
    for name, df in test_datasets:
        analyzer = DataCharacteristicsAnalyzer()
        characteristics = analyzer.analyze_dataset(df)
        insight = reasoning_engine.generate_evidence_based_insight(characteristics)
        responses.append((name, insight))
        print(f"\n{name} dataset insight:")
        print(f"  '{insight}'")

    # Check if responses are actually different
    print(f"\nResponse Uniqueness Check:")
    unique_responses = set(resp[1] for resp in responses)
    if len(unique_responses) == len(responses):
        print(f"✅ All responses are unique ({len(unique_responses)}/{len(responses)})")
    else:
        print(f"❌ Some responses are identical ({len(unique_responses)}/{len(responses)} unique)")

    # Check for templated patterns
    common_phrases = ['dataset contains', 'samples with', 'features']
    for phrase in common_phrases:
        count = sum(1 for _, resp in responses if phrase in resp.lower())
        print(f"  '{phrase}' appears in {count}/{len(responses)} responses")


async def main():
    """Run all investigations"""
    print("🕵️ DEEP INVESTIGATION: VERIFYING REAL vs FAKE INTELLIGENCE")
    print("Checking if the agent actually executes real analysis or just simulates it")
    print("")

    investigate_actual_execution()
    await investigate_pipeline_execution()
    investigate_iteration_behavior()
    check_for_hardcoded_responses()

    print(f"\n" + "=" * 70)
    print(f"🎯 INVESTIGATION COMPLETE")
    print(f"Review the results above to determine if intelligence is genuine or simulated")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()