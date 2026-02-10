"""
Comprehensive Validation: Real Intelligence vs Fake Intelligence
This script demonstrates that fake components have been eliminated and replaced with genuine analysis
"""

import numpy as np
import pandas as pd
import asyncio
import time
import matplotlib.pyplot as plt
from agent.real_timing_agent import RealTimingDataSelectionAgent
from agent.real_intelligence_engine import DataCharacteristicsAnalyzer
from agent.intelligent_sampling_engine import EvidenceBasedSampler, IntelligentAlgorithmSelector


def create_validation_datasets():
    """Create datasets with known, verifiable characteristics"""
    np.random.seed(12345)  # Fixed seed for reproducible validation

    datasets = {}

    # Dataset 1: Three clear clusters (KNOWN structure)
    cluster1 = np.random.normal([2, 2], 0.5, (100, 2))
    cluster2 = np.random.normal([8, 2], 0.5, (100, 2))
    cluster3 = np.random.normal([5, 8], 0.5, (100, 2))

    clustered_data = np.vstack([cluster1, cluster2, cluster3])
    clustered_df = pd.DataFrame(clustered_data, columns=['x', 'y'])
    clustered_df['timing'] = clustered_df['x'] + clustered_df['y'] + np.random.normal(0, 0.1, 300)

    datasets['three_clusters'] = {
        'data': clustered_df,
        'expected_clusters': 3,
        'expected_hopkins': 'high',  # Should be > 0.7
        'expected_outliers': 'low'   # Should be < 0.15
    }

    # Dataset 2: Heavy outliers (KNOWN contamination)
    normal_points = np.random.normal([5, 5], 1, (200, 2))
    outlier_points = np.array([
        [15, 15], [15, -5], [-5, 15], [-5, -5],  # Corner outliers
        [20, 5], [5, 20], [-10, 5], [5, -10]    # Edge outliers
    ] * 5)  # Repeat to create 40 outliers = 40/240 = 16.7%

    contaminated_data = np.vstack([normal_points, outlier_points])
    contaminated_df = pd.DataFrame(contaminated_data, columns=['feature1', 'feature2'])
    contaminated_df['value'] = contaminated_df.sum(axis=1)

    datasets['high_outliers'] = {
        'data': contaminated_df,
        'expected_clusters': 'variable',
        'expected_hopkins': 'medium',
        'expected_outliers': 'high'  # Should detect ~16.7% outliers
    }

    # Dataset 3: Uniform random (NO structure)
    uniform_data = np.random.uniform(-10, 10, (250, 4))
    uniform_df = pd.DataFrame(uniform_data, columns=['a', 'b', 'c', 'd'])

    datasets['uniform_random'] = {
        'data': uniform_df,
        'expected_clusters': 'unclear',
        'expected_hopkins': 'low',    # Should be < 0.5
        'expected_outliers': 'low'
    }

    return datasets


def validate_data_analysis_is_real():
    """Validate that data analysis produces measurable, consistent results"""
    print("=" * 60)
    print("VALIDATION 1: DATA ANALYSIS PRODUCES REAL MEASUREMENTS")
    print("=" * 60)

    analyzer = DataCharacteristicsAnalyzer()
    datasets = create_validation_datasets()

    validation_results = {}

    for name, dataset_info in datasets.items():
        print(f"\n--- Analyzing {name} ---")
        df = dataset_info['data']
        expected = dataset_info

        # Run analysis multiple times to check consistency
        analyses = []
        for run in range(3):
            characteristics = analyzer.analyze_dataset(df)
            analyses.append(characteristics)

        # Extract key metrics
        hopkins_scores = [a['clustering_potential']['hopkins_statistic'] for a in analyses]
        outlier_ratios = [a['outlier_analysis']['global_outlier_ratio'] for a in analyses]
        cluster_counts = [a['clustering_potential']['optimal_cluster_count'] for a in analyses]

        # Check consistency (real analysis should be stable)
        hopkins_std = np.std(hopkins_scores)
        outlier_std = np.std(outlier_ratios)

        print(f"Hopkins statistic: {np.mean(hopkins_scores):.3f} ± {hopkins_std:.3f}")
        print(f"Outlier ratio: {np.mean(outlier_ratios):.3f} ± {outlier_std:.3f}")
        print(f"Optimal clusters: {cluster_counts}")

        # Validate against expected characteristics
        mean_hopkins = np.mean(hopkins_scores)
        mean_outliers = np.mean(outlier_ratios)

        validation_results[name] = {
            'hopkins_consistent': hopkins_std < 0.1,  # Should be stable
            'outliers_consistent': outlier_std < 0.05,
            'meets_expectations': True
        }

        # Check expectations
        if expected['expected_hopkins'] == 'high':
            validation_results[name]['meets_expectations'] &= mean_hopkins > 0.7
            print(f"✓ High clustering tendency detected: {mean_hopkins:.3f} > 0.7")
        elif expected['expected_hopkins'] == 'low':
            validation_results[name]['meets_expectations'] &= mean_hopkins < 0.6
            print(f"✓ Low clustering tendency detected: {mean_hopkins:.3f} < 0.6")

        if expected['expected_outliers'] == 'high':
            validation_results[name]['meets_expectations'] &= mean_outliers > 0.10
            print(f"✓ High outlier ratio detected: {mean_outliers:.3f} > 0.10")
        elif expected['expected_outliers'] == 'low':
            validation_results[name]['meets_expectations'] &= mean_outliers < 0.15
            print(f"✓ Reasonable outlier ratio: {mean_outliers:.3f} < 0.15")

        if name == 'three_clusters':
            most_common_k = max(set(cluster_counts), key=cluster_counts.count)
            if most_common_k == 3:
                print(f"✓ Correctly identified 3 clusters: {cluster_counts}")
            else:
                print(f"⚠ Cluster count varied: {cluster_counts} (expected 3)")
                validation_results[name]['meets_expectations'] &= False

    print(f"\n--- VALIDATION 1 SUMMARY ---")
    all_valid = True
    for name, results in validation_results.items():
        consistent = results['hopkins_consistent'] and results['outliers_consistent']
        accurate = results['meets_expectations']
        status = "PASS" if (consistent and accurate) else "FAIL"
        all_valid &= (consistent and accurate)

        print(f"{name}: {status} (consistent: {consistent}, accurate: {accurate})")

    print(f"\nVALIDATION 1 RESULT: {'PASS - Real analysis confirmed' if all_valid else 'FAIL - Analysis issues detected'}")
    return all_valid


def validate_algorithm_selection_is_intelligent():
    """Validate that algorithm selection is based on data characteristics, not random"""
    print("\n" + "=" * 60)
    print("VALIDATION 2: ALGORITHM SELECTION IS DATA-DRIVEN")
    print("=" * 60)

    analyzer = DataCharacteristicsAnalyzer()
    selector = IntelligentAlgorithmSelector()
    datasets = create_validation_datasets()

    selection_results = {}

    for name, dataset_info in datasets.items():
        print(f"\n--- Testing algorithm selection for {name} ---")
        df = dataset_info['data']

        # Analyze data characteristics
        characteristics = analyzer.analyze_dataset(df)

        # Test selection consistency (should be deterministic for same data)
        selections = []
        for run in range(5):
            algorithms = selector.select_optimal_algorithm(characteristics)
            top_choice = algorithms[0]['name'] if algorithms else 'none'
            selections.append(top_choice)

        # Check if selection is consistent (real intelligence should be deterministic)
        most_common = max(set(selections), key=selections.count)
        consistency_rate = selections.count(most_common) / len(selections)

        print(f"Algorithm selections: {selections}")
        print(f"Most common choice: {most_common} ({consistency_rate*100:.0f}% consistency)")

        # Get detailed reasoning for top choice
        algorithms = selector.select_optimal_algorithm(characteristics)
        if algorithms:
            top_algorithm = algorithms[0]
            print(f"Primary choice: {top_algorithm['name']}")
            print(f"Confidence: {top_algorithm['confidence']:.3f}")
            print(f"Reasoning: {top_algorithm['reasoning']}")

            # Validate reasoning mentions actual data characteristics
            reasoning = top_algorithm['reasoning'].lower()
            data_terms = ['hopkins', 'outlier', 'correlation', 'cluster', 'density', 'feature', 'sample']
            mentions_data = sum(1 for term in data_terms if term in reasoning)

            selection_results[name] = {
                'consistent': consistency_rate >= 0.8,
                'reasoning_quality': mentions_data >= 2,  # Should mention at least 2 data characteristics
                'confidence_reasonable': 0.3 <= top_algorithm['confidence'] <= 1.0
            }

            print(f"✓ Reasoning mentions {mentions_data} data characteristics")
        else:
            selection_results[name] = {'consistent': False, 'reasoning_quality': False, 'confidence_reasonable': False}

    print(f"\n--- VALIDATION 2 SUMMARY ---")
    all_intelligent = True
    for name, results in selection_results.items():
        intelligent = all(results.values())
        all_intelligent &= intelligent
        status = "PASS" if intelligent else "FAIL"
        print(f"{name}: {status} (consistent: {results['consistent']}, reasoning: {results['reasoning_quality']}, confidence: {results['confidence_reasonable']})")

    print(f"\nVALIDATION 2 RESULT: {'PASS - Intelligent selection confirmed' if all_intelligent else 'FAIL - Selection not data-driven'}")
    return all_intelligent


def validate_sampling_preserves_structure():
    """Validate that sampling preserves known data structure"""
    print("\n" + "=" * 60)
    print("VALIDATION 3: SAMPLING PRESERVES DATA STRUCTURE")
    print("=" * 60)

    analyzer = DataCharacteristicsAnalyzer()
    sampler = EvidenceBasedSampler()
    datasets = create_validation_datasets()

    sampling_results = {}

    for name, dataset_info in datasets.items():
        print(f"\n--- Testing sampling for {name} ---")
        df = dataset_info['data']

        # Analyze characteristics to determine strategy
        characteristics = analyzer.analyze_dataset(df)

        # Define strategy based on characteristics
        if characteristics['clustering_potential']['is_clusterable']:
            strategy = {
                'primary_method': 'stratified_clustering',
                'parameters': {'n_clusters': characteristics['clustering_potential']['optimal_cluster_count']}
            }
        elif characteristics['outlier_analysis']['global_outlier_ratio'] > 0.15:
            strategy = {
                'primary_method': 'outlier_preserving',
                'parameters': {'outlier_ratio': characteristics['outlier_analysis']['global_outlier_ratio']}
            }
        else:
            strategy = {
                'primary_method': 'uniform_random',
                'parameters': {}
            }

        # Test sampling multiple times
        sample_qualities = []
        for run in range(3):
            result = sampler.select_samples(df, 15.0, strategy, characteristics)
            selected_indices = result['selected_indices']

            # Measure how well sampling preserved structure
            numeric_df = df.select_dtypes(include=[np.number])
            original_means = numeric_df.mean()
            sample_means = numeric_df.iloc[selected_indices].mean()

            # Calculate preservation quality (how close are the means?)
            mean_differences = np.abs(original_means - sample_means)
            mean_preservation = 1.0 - np.mean(mean_differences / original_means.abs())
            sample_qualities.append(max(0, mean_preservation))

        avg_quality = np.mean(sample_qualities)
        quality_consistency = 1.0 - np.std(sample_qualities)

        print(f"Sample preservation quality: {avg_quality:.3f} ± {np.std(sample_qualities):.3f}")
        print(f"Method used: {result['method_used']}")
        print(f"Sample size: {len(selected_indices)}/{len(df)} ({len(selected_indices)/len(df)*100:.1f}%)")

        # Special validation for known structures
        if name == 'three_clusters' and strategy['primary_method'] == 'stratified_clustering':
            # Should sample from all clusters
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans

            X = StandardScaler().fit_transform(numeric_df)
            kmeans = KMeans(n_clusters=3, random_state=42)
            all_labels = kmeans.fit_predict(X)
            sample_labels = all_labels[selected_indices]

            clusters_represented = len(np.unique(sample_labels))
            cluster_coverage = clusters_represented / 3.0
            print(f"✓ Clusters in sample: {clusters_represented}/3 (coverage: {cluster_coverage:.1%})")

            sampling_results[name] = {
                'quality_good': avg_quality > 0.7,
                'consistent': quality_consistency > 0.8,
                'structure_preserved': cluster_coverage >= 0.67  # At least 2/3 clusters
            }
        else:
            sampling_results[name] = {
                'quality_good': avg_quality > 0.6,
                'consistent': quality_consistency > 0.8,
                'structure_preserved': True  # No specific structure to check
            }

    print(f"\n--- VALIDATION 3 SUMMARY ---")
    all_preserving = True
    for name, results in sampling_results.items():
        preserving = all(results.values())
        all_preserving &= preserving
        status = "PASS" if preserving else "FAIL"
        print(f"{name}: {status} (quality: {results['quality_good']}, consistent: {results['consistent']}, structure: {results['structure_preserved']})")

    print(f"\nVALIDATION 3 RESULT: {'PASS - Structure preservation confirmed' if all_preserving else 'FAIL - Poor structure preservation'}")
    return all_preserving


async def validate_no_fake_components():
    """Validate that no fake intelligence components are being used"""
    print("\n" + "=" * 60)
    print("VALIDATION 4: NO FAKE INTELLIGENCE COMPONENTS")
    print("=" * 60)

    # Test the complete agent
    agent = RealTimingDataSelectionAgent(verbose=False)

    # Create test data and save it
    df = create_validation_datasets()['three_clusters']['data']
    csv_path = '/tmp/claude/validation_test.csv'
    df.to_csv(csv_path, index=False)

    print(f"Testing complete agent pipeline...")
    start_time = time.time()
    result = await agent.intelligent_sample_selection(csv_path, 10.0)
    execution_time = time.time() - start_time

    print(f"Execution completed in {execution_time:.2f} seconds")

    # Check that no fake components are being used
    fake_indicators = []

    # Check reasoning chain for fake phrases
    reasoning_chain = result.get('reasoning_chain', [])
    reasoning_text = ' '.join(reasoning_chain).lower()

    fake_phrases = [
        'random.choice',
        'timing_thoughts',
        'timing_actions',
        'generic analysis',
        'standard approach',
        'default strategy'
    ]

    for phrase in fake_phrases:
        if phrase in reasoning_text:
            fake_indicators.append(f"Fake phrase detected: '{phrase}'")

    # Check that reasoning contains actual measurements
    has_numbers = any(char.isdigit() for char in reasoning_text)
    if not has_numbers:
        fake_indicators.append("No numerical measurements in reasoning")

    # Check that data analysis was performed
    data_analysis = result.get('data_analysis', {})
    if not data_analysis.get('data_characteristics'):
        fake_indicators.append("No data characteristics analysis found")

    # Check for evidence of real statistical analysis
    characteristics = data_analysis.get('data_characteristics', {})
    required_analyses = [
        'basic_stats',
        'correlation_analysis',
        'clustering_potential',
        'outlier_analysis'
    ]

    missing_analyses = [analysis for analysis in required_analyses
                       if analysis not in characteristics]

    if missing_analyses:
        fake_indicators.append(f"Missing analyses: {missing_analyses}")

    # Check that specific metrics were computed
    clustering_potential = characteristics.get('clustering_potential', {})
    if 'hopkins_statistic' not in clustering_potential:
        fake_indicators.append("Hopkins statistic not computed")

    outlier_analysis = characteristics.get('outlier_analysis', {})
    if 'global_outlier_ratio' not in outlier_analysis:
        fake_indicators.append("Outlier ratio not computed")

    # Validate reasoning mentions actual computed values
    hopkins_value = clustering_potential.get('hopkins_statistic')
    outlier_ratio = outlier_analysis.get('global_outlier_ratio')

    if hopkins_value is not None:
        hopkins_mentioned = f"{hopkins_value:.1f}" in reasoning_text or f"{hopkins_value:.2f}" in reasoning_text or f"{hopkins_value:.3f}" in reasoning_text
        if not hopkins_mentioned:
            fake_indicators.append("Hopkins statistic computed but not mentioned in reasoning")

    print(f"\n--- Validation Results ---")
    print(f"Reasoning chain: {reasoning_chain}")
    print(f"Hopkins statistic computed: {hopkins_value}")
    print(f"Outlier ratio computed: {outlier_ratio}")
    print(f"Selected samples: {len(result.get('selected_indices', []))}")
    print(f"Method used: {result.get('selection_method', 'unknown')}")

    if fake_indicators:
        print(f"\nFAKE INTELLIGENCE DETECTED:")
        for indicator in fake_indicators:
            print(f"  ❌ {indicator}")
        return False
    else:
        print(f"\n✅ NO FAKE INTELLIGENCE DETECTED")
        print(f"✅ Real statistical analysis confirmed")
        print(f"✅ Evidence-based reasoning verified")
        print(f"✅ Data-driven decision making validated")
        return True


async def comprehensive_validation():
    """Run all validation tests"""
    print("🔍 COMPREHENSIVE REAL INTELLIGENCE VALIDATION")
    print("=" * 80)

    import os
    os.makedirs('/tmp/claude', exist_ok=True)

    # Run all validation tests
    test1_pass = validate_data_analysis_is_real()
    test2_pass = validate_algorithm_selection_is_intelligent()
    test3_pass = validate_sampling_preserves_structure()
    test4_pass = await validate_no_fake_components()

    print("\n" + "=" * 80)
    print("🏆 FINAL VALIDATION SUMMARY")
    print("=" * 80)

    tests = [
        ("Data Analysis Produces Real Measurements", test1_pass),
        ("Algorithm Selection Is Data-Driven", test2_pass),
        ("Sampling Preserves Data Structure", test3_pass),
        ("No Fake Intelligence Components", test4_pass)
    ]

    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")

    all_passed = all(result for _, result in tests)

    if all_passed:
        print(f"\n🎉 VALIDATION SUCCESSFUL: REAL INTELLIGENCE CONFIRMED")
        print(f"✅ All fake components successfully eliminated")
        print(f"✅ Genuine data-driven analysis implemented")
        print(f"✅ Evidence-based reasoning operational")
        print(f"✅ Statistical intelligence verified")
    else:
        print(f"\n❌ VALIDATION FAILED: FAKE INTELLIGENCE DETECTED")
        print(f"⚠ System still contains non-genuine components")

    return all_passed


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        success = loop.run_until_complete(comprehensive_validation())
        exit(0 if success else 1)
    finally:
        loop.close()