"""
Real Intelligence Timing Data Selection Agent
Replaces all fake intelligence with genuine data-driven analysis and reasoning
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import json
import time
import numpy as np
import pandas as pd
import os
import asyncio
from dataclasses import dataclass
from collections import defaultdict

# Import real intelligence engines
from .real_intelligence_engine import (
    DataCharacteristicsAnalyzer,
    IntelligentReasoningEngine,
    IntelligentParameterOptimizer,
    ConcreteInsightsMemory
)
from .intelligent_sampling_engine import (
    EvidenceBasedSampler,
    IntelligentAlgorithmSelector
)
from .analysis_visualizer import AnalysisVisualizer

# Core ML libraries with fallbacks
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Intent Classification System - Keep this as it's functional
class UserIntent(Enum):
    """User intent categories for conversational interface."""
    EXECUTE_SAMPLING = "execute_sampling"
    QUESTION_ABOUT_RESULTS = "question_about_results"
    MODIFY_PARAMETERS = "modify_parameters"
    EXPLAIN_METHODOLOGY = "explain_methodology"
    REQUEST_VISUALIZATION = "request_visualization"
    GENERAL_HELP = "general_help"

# Real intelligence configuration - data-driven parameters
ANALYSIS_PARAMETERS = {
    'min_samples_for_clustering': 50,
    'max_clusters_to_test': 10,
    'outlier_detection_threshold': 0.1,
    'correlation_significance_threshold': 0.7,
    'convergence_tolerance': 0.05,
    'max_analysis_iterations': 20
}

@dataclass
class AnalysisResult:
    """Result from real data analysis iteration"""
    iteration: int
    characteristics: Dict[str, Any]
    strategy: Dict[str, Any]
    reasoning: str
    confidence: float
    performance_metrics: Dict[str, Any]
    selected_indices: List[int]
    success: bool
    execution_time: float

@dataclass
class IntelligentInsight:
    """Concrete insight derived from data analysis"""
    category: str
    description: str
    evidence: Dict[str, Any]
    confidence: float
    actionable_recommendation: str


class RealIntelligenceExplorationEngine:
    """Genuine data-driven exploration engine - no fake thoughts or actions"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.data_analyzer = DataCharacteristicsAnalyzer(verbose=verbose)
        self.reasoning_engine = IntelligentReasoningEngine()
        self.parameter_optimizer = IntelligentParameterOptimizer()
        self.insights_memory = ConcreteInsightsMemory()

        # Real tracking - no fake patterns
        self.analysis_history = []
        self.convergence_indicators = []
        self.performance_trends = []

    async def intelligent_explore(self, data: pd.DataFrame, target_percentage: float = 5.0) -> Dict[str, Any]:
        """Genuine intelligent exploration based on statistical analysis"""

        start_time = time.time()

        if not SKLEARN_AVAILABLE:
            return {'error': 'Machine learning libraries not available'}

        if self.verbose:
            print(f"   \033[35mOBSERVATION:\033[0m Starting deep statistical analysis of {len(data)} samples...")

        # Step 1: Deep data analysis - REAL intelligence
        characteristics = self.data_analyzer.analyze_dataset(data)

        if self.verbose:
            print(f"   \033[32mCOMPLETED:\033[0m Data analysis in {time.time() - start_time:.2f}s")
            print(f"   \033[36mREASONING:\033[0m Generating evidence-based insights...")

        # Step 2: Evidence-based reasoning - REAL intelligence
        reasoning = self.reasoning_engine.generate_evidence_based_insight(characteristics)

        if self.verbose:
            print(f"   \033[33mACTION:\033[0m Determining optimal strategy...")

        # Step 3: Data-driven strategy selection - REAL intelligence
        strategy = self.reasoning_engine.determine_optimal_strategy(characteristics)

        # Step 4: Performance tracking
        execution_time = time.time() - start_time

        exploration_result = {
            'data_characteristics': characteristics,
            'evidence_based_reasoning': reasoning,
            'optimal_strategy': strategy,
            'analysis_quality': self._assess_analysis_quality(characteristics),
            'execution_time': execution_time,
            'iteration': len(self.analysis_history) + 1
        }

        self.analysis_history.append(exploration_result)
        return exploration_result

    def _assess_analysis_quality(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality and completeness of data analysis"""
        quality_score = 0.0
        quality_indicators = []

        # Check completeness of analysis
        basic_stats = characteristics.get('basic_stats', {})
        if basic_stats.get('n_samples', 0) > 0:
            quality_score += 0.2
            quality_indicators.append("Basic statistics computed")

        correlation_analysis = characteristics.get('correlation_analysis', {})
        if correlation_analysis.get('max_correlation') is not None:
            quality_score += 0.2
            quality_indicators.append("Correlation analysis completed")

        clustering_potential = characteristics.get('clustering_potential', {})
        if clustering_potential.get('hopkins_statistic') is not None:
            quality_score += 0.3
            quality_indicators.append("Clustering potential assessed")

        outlier_analysis = characteristics.get('outlier_analysis', {})
        if outlier_analysis.get('global_outlier_ratio') is not None:
            quality_score += 0.2
            quality_indicators.append("Outlier analysis completed")

        feature_analysis = characteristics.get('feature_importance', {})
        if feature_analysis.get('feature_rankings'):
            quality_score += 0.1
            quality_indicators.append("Feature importance analyzed")

        return {
            'quality_score': quality_score,
            'quality_indicators': quality_indicators,
            'completeness': len(quality_indicators) / 5.0
        }


class RealIntelligenceExperimentExecutor:
    """Executes experiments based on data characteristics, not random choices"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.algorithm_selector = IntelligentAlgorithmSelector()
        self.parameter_optimizer = IntelligentParameterOptimizer()
        self.execution_history = []

    async def execute_intelligent_experiments(self, characteristics: Dict[str, Any],
                                           data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Execute experiments selected by real intelligence"""

        # Select algorithms based on data characteristics
        recommended_algorithms = self.algorithm_selector.select_optimal_algorithm(characteristics)

        if self.verbose:
            print(f"   \033[35mOBSERVATION:\033[0m Selected {len(recommended_algorithms)} algorithms for testing")

        experiment_results = []

        for i, algorithm_config in enumerate(recommended_algorithms):
            if self.verbose:
                print(f"   \033[33mACTION:\033[0m Testing algorithm {i+1}/{len(recommended_algorithms)}: {algorithm_config['name']}")

            result = await self._execute_single_experiment(
                algorithm_config, characteristics, data
            )
            experiment_results.append(result)

            if self.verbose:
                score = result.get('performance_score', 0)
                status = "SUCCESS" if result.get('success') else "FAILED"

                # Explain the scoring system
                if result.get('success'):
                    score_explanation = self._explain_clustering_score(score)
                    print(f"   \033[32mRESULT:\033[0m {algorithm_config['name']}: silhouette_score={score:.3f} ({score_explanation})")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    print(f"   \033[31mRESULT:\033[0m {algorithm_config['name']}: FAILED - {error_msg}")

            # Update algorithm performance history
            if result.get('success'):
                self.algorithm_selector.update_algorithm_performance(
                    algorithm_config['name'],
                    result.get('performance_score', 0),
                    characteristics
                )

        if self.verbose:
            best_score = max([r.get('performance_score', 0) for r in experiment_results])
            best_algorithm = max(experiment_results, key=lambda x: x.get('performance_score', 0))['algorithm'] if experiment_results else 'none'
            print(f"   \033[32mCOMPLETED:\033[0m Algorithm testing complete. Best: {best_algorithm} (score: {best_score:.3f})")

            # Show detailed reasoning for algorithm selection
            self._display_algorithm_reasoning(experiment_results, characteristics)

        return experiment_results

    def _explain_clustering_score(self, score: float) -> str:
        """Explain what the clustering score means"""
        if score > 0.7:
            return "EXCELLENT clustering quality"
        elif score > 0.5:
            return "GOOD clustering quality"
        elif score > 0.3:
            return "FAIR clustering quality"
        elif score > 0:
            return "POOR clustering quality"
        else:
            return "INVALID clustering"

    def _display_algorithm_reasoning(self, experiment_results: List[Dict[str, Any]], characteristics: Dict[str, Any]):
        """Display detailed reasoning for algorithm selection"""
        print(f"\n   \033[36mREASONING PROCESS:\033[0m")

        # Show data characteristics that influenced decisions
        basic_stats = characteristics.get('basic_stats', {})
        clustering = characteristics.get('clustering_potential', {})
        outliers = characteristics.get('outlier_analysis', {})
        correlations = characteristics.get('correlation_analysis', {})

        print(f"   Data Profile:")
        print(f"   - Samples: {basic_stats.get('n_samples', 0)}, Features: {basic_stats.get('n_features', 0)}")
        print(f"   - Hopkins Statistic: {clustering.get('hopkins_statistic', 0):.3f} (clustering tendency)")
        print(f"   - Outlier Ratio: {outliers.get('global_outlier_ratio', 0):.1%}")
        print(f"   - Max Correlation: {correlations.get('max_correlation', 0):.3f}")

        print(f"\n   Algorithm Evaluation:")
        for result in experiment_results:
            if result.get('success'):
                alg_name = result['algorithm']
                score = result.get('performance_score', 0)
                reasoning = result.get('reasoning', 'No reasoning provided')
                confidence = result.get('confidence', 0)

                print(f"   - {alg_name}:")
                print(f"     * Score: {score:.3f} ({self._explain_clustering_score(score)})")
                print(f"     * Confidence: {confidence:.1%}")
                print(f"     * Reasoning: {reasoning}")

        # Show selection logic
        if experiment_results:
            best = max(experiment_results, key=lambda x: x.get('performance_score', 0))
            print(f"\n   \033[32mSELECTION LOGIC:\033[0m")
            print(f"   Selected {best['algorithm']} because:")
            print(f"   - Highest silhouette score ({best.get('performance_score', 0):.3f})")
            print(f"   - {best.get('reasoning', 'Best fit for data characteristics')}")

    async def _execute_single_experiment(self, algorithm_config: Dict[str, Any],
                                       characteristics: Dict[str, Any],
                                       data: pd.DataFrame) -> Dict[str, Any]:
        """Execute single experiment with intelligent parameter optimization"""

        start_time = time.time()

        try:
            numeric_data = data.select_dtypes(include=[np.number])
            X_scaled = StandardScaler().fit_transform(numeric_data)

            algorithm_name = algorithm_config['name']
            base_params = algorithm_config.get('parameters', {})

            # Optimize parameters based on data characteristics
            optimized_params = self.parameter_optimizer.optimize_clustering_parameters(
                X_scaled, algorithm_name
            )

            # Merge base and optimized parameters
            final_params = {**base_params, **optimized_params}

            # Execute clustering
            result = self._run_clustering_algorithm(X_scaled, algorithm_name, final_params)

            execution_time = time.time() - start_time

            return {
                'algorithm': algorithm_name,
                'algorithm_name': algorithm_name,  # Ensure both keys are present
                'parameters_used': final_params,
                'performance_score': result.get('silhouette_score', 0),
                'cluster_info': result,
                'execution_time': execution_time,
                'success': True,
                'reasoning': algorithm_config.get('reasoning', ''),
                'confidence': algorithm_config.get('confidence', 0.5)
            }

        except Exception as e:
            return {
                'algorithm': algorithm_name,
                'error': str(e),
                'success': False,
                'execution_time': time.time() - start_time
            }

    def _run_clustering_algorithm(self, X_scaled: np.ndarray, algorithm_name: str,
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run specific clustering algorithm with optimized parameters"""

        if algorithm_name == 'KMeans':
            clusterer = KMeans(**parameters)
        elif algorithm_name == 'DBSCAN':
            # Add eps if not provided
            if 'eps' not in parameters:
                from sklearn.neighbors import NearestNeighbors
                k = min(4, len(X_scaled) // 10)
                nbrs = NearestNeighbors(n_neighbors=k)
                nbrs.fit(X_scaled)
                distances, _ = nbrs.kneighbors(X_scaled)
                eps = np.percentile(np.sort(distances[:, k-1]), 80)
                parameters['eps'] = eps
            clusterer = DBSCAN(**parameters)
        elif algorithm_name == 'GaussianMixture':
            clusterer = GaussianMixture(**parameters)
        elif algorithm_name == 'AgglomerativeClustering':
            clusterer = AgglomerativeClustering(**parameters)
        else:
            # Default fallback
            clusterer = KMeans(n_clusters=3, random_state=42)

        labels = clusterer.fit_predict(X_scaled)

        # Calculate performance metrics
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])

        result = {
            'labels': labels,
            'n_clusters_found': n_clusters,
            'algorithm_used': algorithm_name
        }

        # Calculate silhouette score if possible
        if n_clusters > 1 and len(unique_labels) > 1:
            if algorithm_name == 'DBSCAN':
                # Only calculate for non-noise points
                non_noise_mask = labels != -1
                if non_noise_mask.sum() > 10:
                    result['silhouette_score'] = silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])
                else:
                    result['silhouette_score'] = 0.0
            else:
                result['silhouette_score'] = silhouette_score(X_scaled, labels)

            # Additional metrics
            if n_clusters > 1:
                result['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, labels)
                result['davies_bouldin_score'] = davies_bouldin_score(X_scaled, labels)
        else:
            result['silhouette_score'] = 0.0

        return result


class TimingDataSelectionAgent:
    """Main agent with real intelligence - no fake components"""

    def __init__(self, llm=None, verbose: bool = True):
        self.verbose = verbose
        self.llm = llm  # LLM as reasoning brain, but fed real data not fake thoughts

        # Real intelligence components
        self.exploration_engine = RealIntelligenceExplorationEngine(verbose=verbose)
        self.experiment_executor = RealIntelligenceExperimentExecutor(verbose=verbose)
        self.evidence_sampler = EvidenceBasedSampler()
        self.insights_memory = ConcreteInsightsMemory()
        self.visualizer = AnalysisVisualizer()

        # Conversation tracking for UI
        self.conversation_history = []
        self.current_data = None
        self.last_results = None

        # Real performance tracking
        self.analysis_iterations = 0
        self.convergence_history = []

        print(f"[INIT] Real Intelligence Timing Data Selection Agent initialized")

    def llm_reason_about_data(self, characteristics: Dict[str, Any]) -> str:
        """Use LLM to reason about real computed data characteristics (not fake thoughts)"""
        if not self.llm:
            return "LLM not available for high-level reasoning"

        # Prepare real data summary for LLM reasoning
        basic_stats = characteristics.get('basic_stats', {})
        clustering = characteristics.get('clustering_potential', {})
        outliers = characteristics.get('outlier_analysis', {})
        correlations = characteristics.get('correlation_analysis', {})

        data_summary = f"""
REAL DATA ANALYSIS RESULTS:
- Dataset: {basic_stats.get('n_samples', 0)} samples, {basic_stats.get('n_features', 0)} features
- Hopkins statistic: {clustering.get('hopkins_statistic', 0):.6f} (clustering tendency measure)
- Optimal clusters: {clustering.get('optimal_cluster_count', 0)}
- Outlier ratio: {outliers.get('global_outlier_ratio', 0):.6f}
- Max correlation: {correlations.get('max_correlation', 0):.6f}
- Strong correlations: {correlations.get('highly_correlated_features', 0)} pairs

Based on these COMPUTED MEASUREMENTS, what sampling strategy would be most effective?
Consider: Should we use stratified sampling for the {clustering.get('optimal_cluster_count', 0)} clusters?
Should we preserve the {outliers.get('global_outlier_ratio', 0):.1%} outliers detected?
Answer with specific strategy recommendations based on the numerical evidence above.
"""

        try:
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(data_summary)
                if hasattr(response, 'content'):
                    return response.content
                else:
                    return str(response)
            else:
                return "LLM interface not compatible"
        except Exception as e:
            return f"LLM reasoning failed: {e}"

    async def intelligent_sample_selection(self, csv_path: str, target_percentage: float = 5.0) -> Dict[str, Any]:
        """
        Complete intelligent pipeline using real data-driven analysis
        No fake thoughts, actions, or random choices
        """

        print(f"\n[START] REAL INTELLIGENCE TIMING DATA SELECTION")
        print(f"[TARGET] Selecting {target_percentage}% using evidence-based analysis")

        start_time = time.time()

        # Load and validate data
        try:
            self.current_data = pd.read_csv(csv_path)
            print(f"[DATA] Loaded {len(self.current_data)} samples with {len(self.current_data.columns)} features")
        except Exception as e:
            return {'error': f'Failed to load data: {str(e)}'}

        # Phase 1: Real Data Exploration
        print(f"\n[PHASE 1] INTELLIGENT DATA EXPLORATION")
        exploration_results = await self.exploration_engine.intelligent_explore(
            self.current_data, target_percentage
        )

        # Phase 2: High-Level Reasoning (with real data, not fake thoughts)
        print(f"\n[PHASE 2] \033[36mLLM REASONING\033[0m ABOUT REAL DATA")
        if self.llm:
            llm_reasoning = self.llm_reason_about_data(exploration_results['data_characteristics'])
            # Format the LLM reasoning for terminal display
            formatted_reasoning = self._format_llm_response(llm_reasoning)
            print(f"[LLM-REASONING] {formatted_reasoning}")
        else:
            llm_reasoning = "Statistical analysis complete - proceeding with evidence-based strategy"
            print(f"[STATISTICAL-REASONING] LLM not available, using computed data characteristics for decisions")

        # Phase 3: Evidence-Based Experimentation
        print(f"\n[PHASE 3] INTELLIGENT ALGORITHM SELECTION AND TESTING")
        experiment_results = await self.experiment_executor.execute_intelligent_experiments(
            exploration_results['data_characteristics'], self.current_data
        )

        # Phase 4: Data-Driven Sample Selection
        print(f"\n[PHASE 4] EVIDENCE-BASED SAMPLE SELECTION")

        # Select best performing strategy
        best_experiment = max(experiment_results, key=lambda x: x.get('performance_score', 0)) if experiment_results else None

        if best_experiment and best_experiment.get('success'):
            if self.verbose:
                print(f"   \033[35mOBSERVATION:\033[0m Best algorithm: {best_experiment.get('algorithm_name', 'unknown')}")
                print(f"   \033[33mACTION:\033[0m Applying evidence-based sampling strategy")

            strategy = exploration_results['optimal_strategy']
            sampling_result = self.evidence_sampler.select_samples(
                self.current_data, target_percentage, strategy,
                exploration_results['data_characteristics']
            )

            if self.verbose:
                method = sampling_result.get('method_used', 'unknown')
                count = len(sampling_result.get('selected_indices', []))
                print(f"   \033[32mRESULT:\033[0m Selected {count} samples using {method}")
        else:
            if self.verbose:
                print(f"   \033[31mERROR:\033[0m No successful experiments, falling back to random sampling")
                print(f"   \033[33mACTION:\033[0m Applying fallback random sampling")

            # Fallback to simple random sampling
            target_count = max(1, int(len(self.current_data) * target_percentage / 100))
            sampling_result = {
                'selected_indices': np.random.choice(len(self.current_data), target_count, replace=False).tolist(),
                'method_used': 'fallback_random',
                'reasoning': 'Fallback due to experiment failures'
            }

            if self.verbose:
                print(f"   \033[32mRESULT:\033[0m Selected {target_count} samples using fallback random sampling")

        # Phase 5: Quality Assessment and Learning
        print(f"\n[PHASE 5] PERFORMANCE ANALYSIS AND LEARNING")

        if self.verbose:
            print(f"   \033[33mACTION:\033[0m Assessing selection quality...")

        quality_metrics = self._assess_selection_quality(
            sampling_result['selected_indices'], exploration_results['data_characteristics']
        )

        if self.verbose:
            score = quality_metrics.get('overall_score', 0)
            print(f"   \033[32mRESULT:\033[0m Quality assessment complete. Overall score: {score:.3f}")

        # Store concrete insights for learning
        try:
            if self.verbose:
                print(f"   \033[33mACTION:\033[0m Storing insights and building final result...")

            iteration_id = self.insights_memory.store_iteration_insights(
                exploration_results['data_characteristics'],
                exploration_results['optimal_strategy'],
                {'quality_metrics': quality_metrics, 'experiment_results': experiment_results}
            )

            total_time = time.time() - start_time

            # Build result safely with error handling for each component
            final_result = {
                'selected_indices': sampling_result.get('selected_indices', []),
                'selection_method': sampling_result.get('method_used', sampling_result.get('method', 'unknown')),
                'execution_time': total_time,
                'success': True
            }

            # Add components safely
            try:
                final_result['data_analysis'] = exploration_results
            except Exception as e:
                final_result['data_analysis_error'] = str(e)

            try:
                final_result['experiment_results'] = experiment_results
            except Exception as e:
                final_result['experiment_results_error'] = str(e)

            try:
                final_result['quality_metrics'] = quality_metrics
            except Exception as e:
                final_result['quality_metrics_error'] = str(e)

            try:
                final_result['insights_stored'] = iteration_id
            except Exception as e:
                final_result['insights_stored_error'] = str(e)

            try:
                final_result['reasoning_chain'] = self._build_reasoning_chain(exploration_results, experiment_results, sampling_result)
            except Exception as e:
                final_result['reasoning_chain'] = [f"Reasoning error: {str(e)}"]

            try:
                final_result['llm_reasoning'] = llm_reasoning if self.llm else None
            except Exception as e:
                final_result['llm_reasoning_error'] = str(e)

            if self.verbose:
                print(f"   \033[32mCOMPLETED:\033[0m Result building successful")

        except Exception as e:
            if self.verbose:
                print(f"   \033[31mERROR:\033[0m Result building failed: {e}")

            # Create minimal result in case of error
            final_result = {
                'selected_indices': sampling_result.get('selected_indices', []),
                'selection_method': sampling_result.get('method_used', 'error'),
                'execution_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }

        self.last_results = final_result
        self.analysis_iterations += 1

        # Print analysis summary
        if final_result.get('success'):
            selected_count = len(final_result['selected_indices'])
            method = final_result['selection_method']
            score = final_result['quality_metrics'].get('overall_score', 0)
            print(f"\n[COMPLETE] Selected {selected_count} samples in {final_result['execution_time']:.2f}s")
            print(f"[METHOD] {method}")
            print(f"[QUALITY] Score: {score:.3f}")

            # Add analysis insights summary
            if self.verbose:
                self._print_analysis_summary(final_result)

                # Generate and open analysis plots
                print(f"\n\033[33mACTION:\033[0m Generating analysis visualization...")
                try:
                    html_file = self.visualizer.generate_analysis_plots(
                        self.current_data,
                        final_result['data_analysis']['data_characteristics'],
                        final_result['selected_indices']
                    )
                    print(f"\033[32mCOMPLETED:\033[0m Analysis report generated: {html_file}")

                    # Attempt to open in browser
                    if self.visualizer.open_in_browser(html_file):
                        print(f"\033[32mOPENED:\033[0m Analysis report opened in browser")
                    else:
                        print(f"\033[33mINFO:\033[0m Open this file manually: {html_file}")

                except Exception as e:
                    print(f"\033[31mERROR:\033[0m Failed to generate visualization: {e}")
        else:
            print(f"\n[ERROR] Analysis failed: {final_result.get('error', 'Unknown error')}")

        return final_result

    def _print_analysis_summary(self, result: Dict[str, Any]):
        """Print a summary of the analysis results for user inspection"""
        print(f"\n\033[36mANALYSIS SUMMARY:\033[0m")

        # Data characteristics summary
        data_char = result['data_analysis']['data_characteristics']

        # Basic stats
        basic_stats = data_char.get('basic_stats', {})
        print(f"  Dataset: {basic_stats.get('n_samples', 0)} samples, {basic_stats.get('n_features', 0)} features")

        # Clustering potential
        clustering = data_char.get('clustering_potential', {})
        hopkins = clustering.get('hopkins_statistic', 0)
        optimal_k = clustering.get('optimal_cluster_count', 0)
        print(f"  Clustering: Hopkins={hopkins:.3f}, Optimal k={optimal_k}")

        # Outliers
        outliers = data_char.get('outlier_analysis', {})
        outlier_ratio = outliers.get('global_outlier_ratio', 0)
        print(f"  Outliers: {outlier_ratio:.1%} of data")

        # Correlations
        corr = data_char.get('correlation_analysis', {})
        max_corr = corr.get('max_correlation', 0)
        print(f"  Correlations: Max correlation = {max_corr:.3f}")

        # Selection quality breakdown
        quality = result['quality_metrics']
        print(f"\n\033[32mSELECTION QUALITY:\033[0m")
        print(f"  Overall Score: {quality.get('overall_score', 0):.3f}")

        # Feature coverage if available
        if 'mean_feature_coverage' in quality:
            coverage = quality['mean_feature_coverage']
            print(f"  Feature Coverage: {coverage:.1%}")

        # Distribution preservation if available
        if 'distribution_preservation' in quality:
            preservation = quality['distribution_preservation']
            print(f"  Distribution Preservation: {preservation:.3f}")

        print(f"\n\033[33mRECOMMENDATION:\033[0m")
        if hopkins > 0.7:
            print("  Strong clustering detected - stratified sampling recommended")
        elif outlier_ratio > 0.15:
            print("  High outlier presence - outlier-preserving sampling recommended")
        else:
            print("  Uniform sampling appears suitable for this dataset")

    def _format_llm_response(self, response: str) -> str:
        """Format LLM response for clean terminal display"""
        if not response:
            return "No response received"

        # Remove markdown formatting
        cleaned = response.replace('**', '').replace('*', '').replace('#', '')

        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())

        # Truncate if too long
        if len(cleaned) > 200:
            cleaned = cleaned[:200] + "..."

        # Ensure it ends with proper punctuation
        if cleaned and not cleaned.endswith(('.', '!', '?')):
            cleaned += "."

        return cleaned

    def _assess_selection_quality(self, selected_indices: List[int], characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of sample selection using statistical measures"""

        if not selected_indices or self.current_data is None:
            return {'overall_score': 0.0, 'error': 'Invalid selection or data'}

        numeric_data = self.current_data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return {'overall_score': 0.0, 'error': 'No numeric data for assessment'}

        selected_data = numeric_data.iloc[selected_indices]

        quality_metrics = {}

        # Coverage assessment
        for col in numeric_data.columns:
            original_range = numeric_data[col].max() - numeric_data[col].min()
            selected_range = selected_data[col].max() - selected_data[col].min()
            coverage = selected_range / original_range if original_range > 0 else 1.0
            quality_metrics[f'{col}_coverage'] = coverage

        # Distribution preservation
        distribution_similarity = []
        for col in numeric_data.columns:
            try:
                from scipy import stats
                _, p_value = stats.ks_2samp(numeric_data[col], selected_data[col])
                distribution_similarity.append(p_value)
            except:
                distribution_similarity.append(0.5)

        # Overall quality score
        mean_coverage = np.mean(list(quality_metrics.values()))
        mean_distribution_preservation = np.mean(distribution_similarity)

        overall_score = (mean_coverage * 0.6 + mean_distribution_preservation * 0.4)

        return {
            'overall_score': overall_score,
            'mean_feature_coverage': mean_coverage,
            'distribution_preservation': mean_distribution_preservation,
            'feature_coverage': quality_metrics,
            'n_selected': len(selected_indices),
            'selection_percentage': len(selected_indices) / len(self.current_data) * 100
        }

    def _build_reasoning_chain(self, exploration: Dict[str, Any], experiments: List[Dict[str, Any]],
                              sampling: Dict[str, Any]) -> List[str]:
        """Build clear reasoning chain showing decision logic"""

        reasoning = []

        # Data analysis reasoning
        characteristics = exploration.get('data_characteristics', {})
        basic_stats = characteristics.get('basic_stats', {})
        reasoning.append(f"Analyzed dataset: {basic_stats.get('n_samples', 0)} samples, {basic_stats.get('n_features', 0)} features")

        # Clustering potential
        clustering = characteristics.get('clustering_potential', {})
        if clustering.get('is_clusterable'):
            reasoning.append(f"Detected clustering potential (Hopkins={clustering.get('hopkins_statistic', 0):.3f})")
        else:
            reasoning.append(f"Low clustering tendency detected (Hopkins={clustering.get('hopkins_statistic', 0):.3f})")

        # Algorithm selection reasoning
        try:
            if experiments and len(experiments) > 0:
                best_exp = max(experiments, key=lambda x: x.get('performance_score', 0))
                algorithm_name = best_exp.get('algorithm_name', best_exp.get('algorithm', 'unknown'))
                reasoning.append(f"Selected {algorithm_name} algorithm (score: {best_exp.get('performance_score', 0):.3f})")
            else:
                reasoning.append("No algorithm experiments completed")
        except Exception as e:
            reasoning.append(f"Algorithm selection: {str(e)}")

        # Sampling method reasoning
        try:
            method = sampling.get('method_used', sampling.get('method', 'unknown'))
            reasoning.append(f"Applied {method} sampling strategy")

            sample_reasoning = sampling.get('reasoning', '')
            if sample_reasoning and sample_reasoning != 'unknown':
                reasoning.append(sample_reasoning)
            else:
                reasoning.append("Method selection completed")
        except Exception as e:
            reasoning.append(f"Sampling method: {str(e)}")

        return reasoning

    # UI Interface methods for compatibility
    def classify_user_intent(self, user_input: str) -> Tuple[UserIntent, Dict[str, Any]]:
        """Classify user intent to determine whether to execute pipeline or answer from context."""
        import re
        input_lower = user_input.lower().strip()

        # Intent patterns with priorities (most specific first)
        intent_patterns = {
            UserIntent.QUESTION_ABOUT_RESULTS: [
                r'why did you (choose|pick|select)',
                r'why.*(\d+)%',
                r'explain (the|your) (selection|choice|decision)',
                r'how did you (determine|decide|choose)',
                r'what (made you|criteria)',
                r'can you explain why',
                r'reasoning behind',
                r'rationale for'
            ],
            UserIntent.EXPLAIN_METHODOLOGY: [
                r'how does.*work',
                r'explain.*methodology',
                r'what.*algorithm',
                r'how.*clustering',
                r'explain.*approach'
            ],
            UserIntent.MODIFY_PARAMETERS: [
                r'change.*percent',
                r'use.*percent',
                r'try.*different',
                r'modify.*selection',
                r'adjust.*parameter'
            ]
        }

        # Extract parameters from user input
        params = self._extract_parameters_from_input(user_input)

        # Check for specific intent patterns
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_lower):
                    return intent, params

        # Check for simple execution patterns
        if any(word in input_lower for word in ['select', 'sample', 'choose', 'pick']):
            return UserIntent.EXECUTE_SAMPLING, params

        # Default to execution if no conversational intent detected
        return UserIntent.EXECUTE_SAMPLING, params

    def _extract_parameters_from_input(self, user_input: str) -> Dict[str, Any]:
        """Extract parameters like percentage, algorithm, etc. from user input."""
        import re
        params = {}

        # Extract percentage
        percentage_patterns = [
            r'(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*percent',
            r'select\s+(\d+(?:\.\d+)?)'
        ]

        for pattern in percentage_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                params['percentage'] = float(match.group(1))
                break

        # Default percentage if not found
        if 'percentage' not in params:
            params['percentage'] = 5.0

        # Extract algorithm preferences
        algorithm_keywords = {
            'kmeans': 'kmeans',
            'gaussian': 'gaussian_mixture',
            'dbscan': 'dbscan',
            'spectral': 'spectral_clustering'
        }

        for keyword, algorithm in algorithm_keywords.items():
            if keyword in user_input.lower():
                params['preferred_algorithm'] = algorithm
                break

        return params

    def run_selection(self, user_query: str, csv_path: str) -> Dict[str, Any]:
        """Synchronous wrapper for UI compatibility"""
        intent, params = self.classify_user_intent(user_query)

        if intent == UserIntent.EXECUTE_SAMPLING:
            percentage = params.get('percentage', 5.0)
            # Run the intelligent selection
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.intelligent_sample_selection(csv_path, percentage))
                # Ensure the result is properly formatted for UI
                if isinstance(result, dict) and result.get('success', False):
                    return result
                else:
                    return {
                        'error': f'Analysis failed: {result.get("error", "Unknown error")}',
                        'success': False,
                        'method': 'error'
                    }
            except Exception as e:
                return {
                    'error': f'Execution failed: {str(e)}',
                    'success': False,
                    'method': 'error'
                }
            finally:
                loop.close()

        elif intent == UserIntent.EXPLAIN_METHODOLOGY:
            return {
                'response': 'This agent uses real statistical analysis to understand data characteristics, then selects optimal algorithms and parameters based on evidence. No random thoughts or fake reasoning.',
                'method': 'explanation'
            }

        elif intent == UserIntent.REQUEST_VISUALIZATION:
            if self.last_results:
                return {
                    'response': 'Visualization data available from last analysis',
                    'visualization_data': self.last_results,
                    'method': 'visualization'
                }
            else:
                return {
                    'response': 'No recent analysis results available for visualization',
                    'method': 'error'
                }

        else:
            return {
                'response': 'Real Intelligence Agent: Uses data-driven analysis for sample selection',
                'method': 'help'
            }

    def generate_interactive_dashboard(self, df: pd.DataFrame, selected_indices: List[int],
                                     clusters: Optional[np.ndarray] = None, title: str = "Analysis Dashboard") -> str:
        """Generate simple HTML dashboard for compatibility"""

        if not selected_indices:
            return "<html><body><h2>No data to visualize</h2></body></html>"

        n_selected = len(selected_indices)
        total = len(df)
        selection_pct = (n_selected / total) * 100

        # Basic statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # Limit to first 5 for display

        stats_html = ""
        for col in numeric_cols:
            original_mean = df[col].mean()
            selected_mean = df.iloc[selected_indices][col].mean()
            stats_html += f"<p><strong>{col}:</strong> Original mean: {original_mean:.3f}, Selected mean: {selected_mean:.3f}</p>"

        html_content = f"""
        <html>
        <head>
            <title>Real Intelligence Analysis Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #2c3e50; }}
                .summary {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                .metric-label {{ color: #7f8c8d; }}
                .stats {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Real Intelligence Sampling Analysis</h1>
                <p>Evidence-Based Sample Selection Results</p>
            </div>

            <div class="summary">
                <div class="metric">
                    <div class="metric-value">{n_selected:,}</div>
                    <div class="metric-label">Samples Selected</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{total:,}</div>
                    <div class="metric-label">Total Samples</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{selection_pct:.1f}%</div>
                    <div class="metric-label">Selection Rate</div>
                </div>
            </div>

            <div class="stats">
                <h3>Feature Statistics Comparison</h3>
                {stats_html}
            </div>

            <div>
                <p><em>Generated by Real Intelligence Engine - No fake components used</em></p>
            </div>
        </body>
        </html>
        """

        return html_content

    # ============================================================================
    # CONVERSATION HANDLING METHODS (for chatbot functionality)
    # ============================================================================

    def add_message(self, role: str, content: str):
        """Add message to conversation history."""
        import pandas as pd
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': pd.Timestamp.now()
        })

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history

    def handle_conversation(self, user_query: str) -> Dict[str, Any]:
        """Handle conversational questions about results without re-running selection."""

        # Classify user intent
        intent, params = self.classify_user_intent(user_query)

        if intent == UserIntent.QUESTION_ABOUT_RESULTS:
            # Generate response based on conversation history
            context = "\n".join([msg['content'] for msg in self.conversation_history[-5:]])

            question_prompt = f"""Based on our previous conversation about timing data selection, answer this follow-up question:

User Question: {user_query}

Recent Context:
{context}

Provide a clear, technical explanation addressing their specific question about the selection methodology, results, or reasoning. Use plain text only."""

            try:
                if self.llm:
                    response = self.llm.invoke({"input": question_prompt})

                    if hasattr(response, 'content'):
                        response_text = response.content
                    else:
                        response_text = str(response)
                else:
                    response_text = "I understand you're asking about the selection results. Without LLM integration, I can provide basic information from the analysis."

                self.add_message('assistant', response_text)

                return {
                    'type': 'conversational_response',
                    'intent': intent.value,
                    'response': response_text,
                    'parameters': params
                }

            except Exception as e:
                return {
                    'type': 'conversational_response',
                    'intent': intent.value,
                    'response': f"I understand you're asking about the selection results, but I encountered an error: {e}",
                    'parameters': params
                }

        elif intent == UserIntent.EXPLAIN_METHODOLOGY:
            # Explain methodology without running selection
            methodology_prompt = f"""Explain the timing data selection methodology to address this question:

{user_query}

Provide a technical explanation of the algorithms, approaches, and reasoning behind the methodology. Focus on the specific aspect they're asking about."""

            try:
                if self.llm:
                    response = self.llm.invoke({"input": methodology_prompt})

                    if hasattr(response, 'content'):
                        response_text = response.content
                    else:
                        response_text = str(response)
                else:
                    response_text = """The methodology uses real statistical analysis including:
1. Hopkins statistic for clustering tendency
2. Evidence-based algorithm selection (KMeans, DBSCAN, etc.)
3. Structure-preserving sampling strategies
4. Quality assessment with coverage and distribution preservation metrics"""

                self.add_message('assistant', response_text)

                return {
                    'type': 'methodology_explanation',
                    'intent': intent.value,
                    'response': response_text,
                    'parameters': params
                }

            except Exception as e:
                return {
                    'type': 'methodology_explanation',
                    'intent': intent.value,
                    'response': f"I can explain the methodology, but encountered an error: {e}",
                    'parameters': params
                }

        else:
            # For other intents, indicate that selection should be run
            return {
                'type': 'requires_execution',
                'intent': intent.value,
                'parameters': params
            }