"""
Intelligent Dynamic Reasoning System for Timing Data Analysis
Replaces template-based "Mad Libs" with real statistical analysis
"""

import json
import numpy as np
from typing import Dict, List, Any

class IntelligentTimingAnalyzer:
    """
    Provides dynamic, context-aware analysis of timing data statistics.
    No more templates - genuine insights based on calculated values.
    """

    def __init__(self, llm):
        """Initialize with LLM for intelligent analysis."""
        self.llm = llm

    def analyze_observation_data(self, observation: Dict[str, Any]) -> str:
        """
        Generate intelligent observation based on actual calculated statistics.

        Args:
            observation: Dictionary containing real calculated stats

        Returns:
            Intelligent analysis text from LLM
        """
        # Extract key statistics for analysis
        stats_summary = self._extract_key_statistics(observation)

        # Create dynamic prompt with actual data
        analysis_prompt = f"""You are a Senior Timing Engineer with 15+ years experience at TSMC/Samsung/Intel.

ACTUAL CALCULATED STATISTICS FOR THIS DATASET:
{json.dumps(stats_summary, indent=2)}

CRITICAL ANALYSIS REQUIRED:
Analyze these SPECIFIC numbers and provide professional timing engineer insights:

1. CORRELATION ANALYSIS: What do the actual correlation values tell you about process variation relationships?

2. VARIABILITY ASSESSMENT: Based on the sigma_by_nominal distribution, identify timing risk areas.

3. DATA QUALITY: Are there any anomalies in the statistics that could affect Monte Carlo accuracy?

4. CLUSTERING READINESS: Given these specific correlations and distributions, predict clustering effectiveness.

PROVIDE SPECIFIC TECHNICAL ANALYSIS (3-4 sentences):
Reference the actual numbers above. Identify ONE specific concern or opportunity for timing signoff based on these statistics. Do NOT use generic templates.

Use plain text only, no special symbols."""

        try:
            # Get intelligent analysis from LLM
            response = self.llm.invoke(analysis_prompt)
            analysis_text = response.content if hasattr(response, 'content') else str(response)

            return analysis_text

        except Exception as e:
            return f"Analysis error: {e}. Using statistical summary: {self._fallback_analysis(stats_summary)}"

    def analyze_strategy_decision(self, observation: Dict[str, Any], target_percentage: float) -> str:
        """
        Generate intelligent strategic reasoning based on data characteristics.

        Args:
            observation: Dataset analysis results
            target_percentage: Target selection percentage

        Returns:
            Strategic analysis from LLM
        """
        # Prepare strategic context
        strategy_context = self._build_strategy_context(observation, target_percentage)

        strategy_prompt = f"""You are a Senior Timing Engineer making strategic decisions for Monte Carlo sample selection.

STRATEGIC CONTEXT:
{json.dumps(strategy_context, indent=2)}

STRATEGIC DECISION REQUIRED:
Based on the specific characteristics of THIS dataset, provide your professional recommendation:

1. ALGORITHM SELECTION: Given the correlation patterns, should we use K-means or GMM? Why?

2. CLUSTER COUNT: What's the optimal number of clusters for this specific data distribution?

3. SAMPLING STRATEGY: How should uncertainty-based sampling be applied given these statistics?

4. BUSINESS IMPACT: What risks/benefits does this approach have for timing signoff accuracy?

PROVIDE STRATEGIC REASONING (4-5 sentences):
Make specific recommendations based on the actual statistics. Connect your technical choices to timing signoff implications. Avoid generic advice.

Use plain text only, no special symbols."""

        try:
            response = self.llm.invoke(strategy_prompt)
            strategy_text = response.content if hasattr(response, 'content') else str(response)

            return strategy_text

        except Exception as e:
            return f"Strategy analysis error: {e}. Using fallback: {self._fallback_strategy(strategy_context)}"

    def analyze_clustering_results(self, clustering_metrics: List[str], pca_info: Dict[str, Any]) -> str:
        """
        Generate intelligent analysis of clustering algorithm performance.

        Args:
            clustering_metrics: List of algorithm performance metrics
            pca_info: PCA analysis results

        Returns:
            Technical decision analysis
        """
        # Build technical analysis context
        technical_context = {
            "pca_analysis": pca_info,
            "clustering_metrics": clustering_metrics,
            "algorithm_comparison": self._parse_clustering_metrics(clustering_metrics)
        }

        decision_prompt = f"""You are a Senior Timing Engineer evaluating clustering algorithm performance.

TECHNICAL ANALYSIS DATA:
{json.dumps(technical_context, indent=2)}

TECHNICAL DECISION REQUIRED:
Based on these specific performance metrics, make your technical recommendation:

1. ALGORITHM PERFORMANCE: Which clustering algorithm performs better for this timing data? Cite specific metric values.

2. PCA EFFECTIVENESS: Is the PCA compression appropriate for timing feature preservation?

3. CLUSTER VALIDATION: Do the clustering metrics indicate good separation for timing path diversity?

4. SELECTION CONFIDENCE: How confident are you in the uncertainty-based sampling for this clustering result?

PROVIDE TECHNICAL DECISION (3-4 sentences):
Make a specific recommendation with numeric justification. Explain WHY this choice optimizes timing characterization. Reference actual BIC/inertia values.

Use plain text only, no special symbols."""

        try:
            response = self.llm.invoke(decision_prompt)
            decision_text = response.content if hasattr(response, 'content') else str(response)

            return decision_text

        except Exception as e:
            return f"Decision analysis error: {e}. Using fallback: {self._fallback_decision(technical_context)}"

    def analyze_selection_results(self, selection_summary: Dict[str, Any]) -> str:
        """
        Generate intelligent analysis of final sample selection.

        Args:
            selection_summary: Results of uncertainty-based sampling

        Returns:
            Selection validation analysis
        """
        validation_prompt = f"""You are a Senior Timing Engineer validating Monte Carlo sample selection results.

SELECTION RESULTS:
{json.dumps(selection_summary, indent=2)}

VALIDATION ANALYSIS REQUIRED:
Based on these specific selection results, provide your professional assessment:

1. COVERAGE ANALYSIS: Does the selection adequately cover timing corners based on cluster distribution?

2. UNCERTAINTY VALIDATION: Are the selected samples truly high-uncertainty cases for model training?

3. SIGNOFF IMPACT: What are the implications for timing accuracy and characterization cost?

4. RISK ASSESSMENT: Are there any gaps or biases in this selection that could affect silicon outcomes?

PROVIDE VALIDATION ASSESSMENT (3-4 sentences):
Give specific feedback on this selection quality. Identify any concerns for timing signoff accuracy. Connect to business impact.

Use plain text only, no special symbols."""

        try:
            response = self.llm.invoke(validation_prompt)
            validation_text = response.content if hasattr(response, 'content') else str(response)

            return validation_text

        except Exception as e:
            return f"Validation error: {e}. Using fallback: {self._fallback_validation(selection_summary)}"

    def _extract_key_statistics(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and summarize key statistics for analysis."""
        stats = {
            "dataset_size": observation.get('total_samples', 0),
            "feature_count": observation.get('n_features', 0),
            "cell_types": len(observation.get('cell_types', {})),
            "high_correlations": len(observation.get('high_correlations', [])),
            "correlation_details": {},
            "feature_statistics": {}
        }

        # Extract correlation details
        for corr in observation.get('high_correlations', []):
            pair = f"{corr['feature1']}_vs_{corr['feature2']}"
            stats["correlation_details"][pair] = round(corr['correlation'], 3)

        # Extract key feature statistics
        timing_stats = observation.get('timing_statistics', {})
        key_features = ['nominal_delay', 'lib_sigma_delay_late', 'sigma_by_nominal']

        for feature in key_features:
            if feature in timing_stats:
                feature_data = timing_stats[feature]
                stats["feature_statistics"][feature] = {
                    "mean": round(feature_data.get('mean', 0), 4),
                    "std": round(feature_data.get('std', 0), 4),
                    "coefficient_of_variation": round(feature_data.get('cv', 0), 4),
                    "range": f"{round(feature_data.get('min', 0), 3)} to {round(feature_data.get('max', 0), 3)}"
                }

        return stats

    def _build_strategy_context(self, observation: Dict[str, Any], target_percentage: float) -> Dict[str, Any]:
        """Build strategic decision context."""
        target_count = int(observation['total_samples'] * target_percentage / 100)

        return {
            "selection_target": {
                "percentage": target_percentage,
                "sample_count": target_count,
                "total_samples": observation['total_samples']
            },
            "data_characteristics": {
                "dimensionality": observation['n_features'],
                "diversity": len(observation.get('cell_types', {})),
                "correlation_complexity": len(observation.get('high_correlations', []))
            },
            "timing_specific_factors": self._identify_timing_factors(observation),
            "clustering_considerations": self._assess_clustering_readiness(observation)
        }

    def _parse_clustering_metrics(self, metrics: List[str]) -> Dict[str, Any]:
        """Parse clustering performance metrics."""
        parsed = {"algorithms": []}

        for metric_line in metrics:
            if "K-means" in metric_line and "GMM" in metric_line:
                # Extract numbers from metric line
                import re
                numbers = re.findall(r'[\d.]+', metric_line)
                if len(numbers) >= 3:
                    parsed["algorithms"].append({
                        "k_value": int(float(numbers[0])),
                        "kmeans_inertia": float(numbers[1]),
                        "gmm_bic": float(numbers[2])
                    })

        return parsed

    def _identify_timing_factors(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Identify timing-specific characteristics."""
        factors = {}

        # Analyze correlations for timing significance
        timing_correls = observation.get('high_correlations', [])
        for corr in timing_correls:
            if 'delay' in corr['feature1'] and 'sigma' in corr['feature2']:
                factors['process_variation_strength'] = abs(corr['correlation'])

        # Analyze feature statistics for timing patterns
        timing_stats = observation.get('timing_statistics', {})
        if 'sigma_by_nominal' in timing_stats:
            sigma_stats = timing_stats['sigma_by_nominal']
            factors['variability_range'] = {
                "min": sigma_stats.get('min', 0),
                "max": sigma_stats.get('max', 0),
                "mean": sigma_stats.get('mean', 0)
            }

        return factors

    def _assess_clustering_readiness(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Assess readiness for clustering based on data characteristics."""
        readiness = {}

        # Correlation-based assessment
        n_correlations = len(observation.get('high_correlations', []))
        if n_correlations > 3:
            readiness['pca_effectiveness'] = 'High'
        elif n_correlations > 1:
            readiness['pca_effectiveness'] = 'Medium'
        else:
            readiness['pca_effectiveness'] = 'Low'

        # Diversity assessment
        n_cell_types = len(observation.get('cell_types', {}))
        if n_cell_types > 10:
            readiness['clustering_potential'] = 'High'
        elif n_cell_types > 5:
            readiness['clustering_potential'] = 'Medium'
        else:
            readiness['clustering_potential'] = 'Low'

        return readiness

    def _fallback_analysis(self, stats: Dict[str, Any]) -> str:
        """Fallback analysis when LLM is unavailable."""
        summary = f"Dataset contains {stats['dataset_size']} samples with {stats['feature_count']} features. "

        if stats['correlation_details']:
            max_corr = max(abs(v) for v in stats['correlation_details'].values())
            summary += f"Strongest correlation: {max_corr:.3f} indicates good clustering potential. "

        return summary

    def _fallback_strategy(self, context: Dict[str, Any]) -> str:
        """Fallback strategy when LLM is unavailable."""
        target = context['selection_target']
        return f"Selecting {target['sample_count']} samples ({target['percentage']}%) using uncertainty-based sampling with GMM clustering."

    def _fallback_decision(self, context: Dict[str, Any]) -> str:
        """Fallback decision when LLM is unavailable."""
        return "GMM recommended for timing data with overlapping distributions. Proceeding with uncertainty-based sampling."

    def _fallback_validation(self, summary: Dict[str, Any]) -> str:
        """Fallback validation when LLM is unavailable."""
        return f"Selected {summary.get('n_selected', 0)} samples with uncertainty-based criteria. Review cluster coverage for timing corners."