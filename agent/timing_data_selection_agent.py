"""
Timing-Aware Data Selection Agent
Senior timing engineer expertise for Monte Carlo sample selection
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import json
import re
import numpy as np
import pandas as pd
import os

# Import core ML libraries at module level to avoid import errors
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from scipy.spatial.distance import cdist
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not available - some features may not work")

# Import visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[WARNING] Plotly not available - will use fallback visualization")


# Intent Classification System
class UserIntent(Enum):
    """User intent categories for conversational Q&A."""
    EXECUTE_SAMPLING = "execute_sampling"
    QUESTION_ABOUT_RESULTS = "question_about_results"
    MODIFY_PARAMETERS = "modify_parameters"
    EXPLAIN_METHODOLOGY = "explain_methodology"
    REQUEST_VISUALIZATION = "request_visualization"
    GENERAL_HELP = "general_help"
from agentic_timing_prompts import (
    AGENTIC_TIMING_SYSTEM_PROMPT as TIMING_SYSTEM_PROMPT,
    AGENTIC_EXPLORE_PROMPT as TIMING_OBSERVE_PROMPT,
    AGENTIC_STRATEGY_PROMPT as TIMING_THINK_PROMPT,
    AGENTIC_EXECUTE_PROMPT as TIMING_ACT_PROMPT,
    AGENTIC_LLM_PARAMETERS,
    VALIDATION_BOUNDARIES,
    ITERATION_TRIGGERS
)

# Fallback decide prompt for compatibility
from TIMING_DECIDE_PROMPT import TIMING_DECIDE_PROMPT

AGENTIC_MODE = True
print("[AGENT] Using Agentic Mode: Autonomous exploration with self-validation")


class TimingDataSelectionAgent:
    """
    Timing-aware agent for intelligent Monte Carlo sample selection.

    Features:
    - Senior timing engineer domain expertise
    - Active learning principles (uncertainty sampling)
    - Process variation awareness
    - Business impact focus (cost reduction)

    Workflow:
    1. OBSERVE: Analyze timing characteristics with domain knowledge
    2. THINK: Strategic reasoning about selection approach
    3. DECIDE: Algorithm selection with timing-specific criteria
    4. ACT: Uncertainty-based sampling for critical corners
    """

    def __init__(self, llm, verbose: bool = True):
        """
        Initialize the timing-aware agent.

        Args:
            llm: LangChain LLM instance
            verbose: Whether to print reasoning steps
        """
        self.llm = llm
        self.verbose = verbose
        self.conversation_history = []
        self.current_data = None
        self.current_features = None
        self.scaler = None
        self.reasoning_log = []
        self._imports_loaded = False

        # Timing domain system prompt
        self.system_prompt = TIMING_SYSTEM_PROMPT

        # Agentic capabilities
        self.agentic_mode = AGENTIC_MODE
        self.validation_boundaries = VALIDATION_BOUNDARIES if AGENTIC_MODE else {}
        self.iteration_triggers = ITERATION_TRIGGERS if AGENTIC_MODE else {}
        self.iteration_count = 0
        self.max_iterations = 3

        if self.agentic_mode:
            print("[AGENT] Initialized in AGENTIC MODE with autonomous exploration")

    def _load_imports(self):
        """Load heavy imports only when needed."""
        if self._imports_loaded:
            return

        # Import LangChain components (these are loaded dynamically)
        global ChatPromptTemplate, HumanMessage, SystemMessage
        from langchain.prompts import ChatPromptTemplate
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError:
            from langchain.schema import HumanMessage, SystemMessage

        # Initialize scaler if needed
        if self.scaler is None and SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()

        self._imports_loaded = True

    def add_message(self, role: str, content: str):
        """Add message to conversation history."""
        self._load_imports()
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': pd.Timestamp.now()
        })

    def log_reasoning(self, stage: str, content: str):
        """Log agent reasoning."""
        self._load_imports()
        self.reasoning_log.append({
            'stage': stage,
            'content': content,
            'timestamp': pd.Timestamp.now()
        })
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"{stage}")
            print(f"{'='*80}")
            print(content)

    def validate_selection_quality(self, selected_indices: List[int], labels: np.ndarray) -> Dict[str, Any]:
        """Validate selection quality against agentic boundaries."""
        if not self.agentic_mode or not self.validation_boundaries:
            return {"validation": "skipped", "quality": "assumed_good"}

        validation_results = {}
        self._load_imports()

        try:
            # Cell type coverage validation
            if self.current_data is not None and 'cell_type' in self.current_data.columns:
                total_cell_types = self.current_data['cell_type'].nunique()
                selected_cell_types = self.current_data.iloc[selected_indices]['cell_type'].nunique()
                cell_coverage = selected_cell_types / total_cell_types

                min_coverage = self.validation_boundaries.get('minimum_cell_type_coverage', 0.8)
                validation_results['cell_coverage'] = {
                    'achieved': cell_coverage,
                    'required': min_coverage,
                    'passed': cell_coverage >= min_coverage
                }

            # Cluster balance validation
            if len(labels) > 0:
                cluster_sizes = [np.sum(labels[selected_indices] == i) for i in np.unique(labels)]
                if len(cluster_sizes) > 1:
                    max_size = max(cluster_sizes)
                    min_size = min([s for s in cluster_sizes if s > 0])
                    imbalance_ratio = max_size / min_size if min_size > 0 else float('inf')

                    max_imbalance = self.validation_boundaries.get('maximum_cluster_imbalance', 3.0)
                    validation_results['cluster_balance'] = {
                        'imbalance_ratio': imbalance_ratio,
                        'max_allowed': max_imbalance,
                        'passed': imbalance_ratio <= max_imbalance
                    }

            # Overall validation status
            all_passed = all(result.get('passed', True) for result in validation_results.values()
                           if isinstance(result, dict) and 'passed' in result)

            validation_results['overall_status'] = 'PASSED' if all_passed else 'FAILED'
            validation_results['requires_iteration'] = not all_passed

            if self.verbose and validation_results.get('requires_iteration'):
                print(f"\n[VALIDATION] Quality check FAILED - iteration required")
                for key, result in validation_results.items():
                    if isinstance(result, dict) and not result.get('passed', True):
                        print(f"  {key}: {result}")

            return validation_results

        except Exception as e:
            return {"validation": "error", "error": str(e), "quality": "unknown"}

    def parse_user_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query with timing domain understanding."""
        self._load_imports()
        parsing_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Parse this timing engineer's request for intelligent Monte Carlo sampling.

Query: "{query}"

CRITICAL: If NO percentage is mentioned in the query, set selection_percentage to null.
ONLY use a percentage if explicitly stated (e.g., "8%", "select 5%", "10 percent").

Extract and return ONLY valid JSON with these fields:
- selection_percentage: float if specified, null if not mentioned (e.g., 8.0 or null)
- selection_criteria: string ("uncertainty", "diversity", "random")
- clustering_preference: string or null ("gmm", "kmeans", or null for auto)
- additional_requirements: string or null for any special timing requirements

For timing library characterization, always use uncertainty-based sampling (active learning).
If no percentage specified, the system will determine optimal percentage based on data analysis.

Return ONLY the JSON object, nothing else.""")
        ])

        chain = parsing_prompt | self.llm
        response = chain.invoke({})

        if hasattr(response, 'content'):
            text = response.content
        else:
            text = str(response)

        text = text.replace('```json', '').replace('```', '').strip()

        try:
            params = json.loads(text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    params = json.loads(json_match.group())
                except json.JSONDecodeError:
                    params = {
                        'selection_percentage': None,  # No default - will determine from data
                        'selection_criteria': 'uncertainty',
                        'clustering_preference': 'gmm',
                        'additional_requirements': 'timing_signoff_focused'
                    }
            else:
                params = {
                    'selection_percentage': None,  # No default - will determine from data
                    'selection_criteria': 'uncertainty',
                    'clustering_preference': 'gmm',
                    'additional_requirements': 'timing_signoff_focused'
                }

        # Store parameters for later processing after data is loaded
        self.add_message('assistant', f"Parsed query parameters: {params}")
        return params

    def observe(self, csv_path: str, target_percentage: float = 5.0) -> Dict[str, Any]:
        """OBSERVE stage with timing domain analysis."""
        self._load_imports()
        print("\nSTAGE 1: OBSERVE (Timing Domain Analysis)")
        print("-" * 80)

        self.current_data = pd.read_csv(csv_path)
        print(f"Analyzing {len(self.current_data)} timing arc samples...")

        # Timing-focused feature selection
        timing_features = [
            'nominal_delay', 'lib_sigma_delay_late',
            'nominal_tran', 'lib_sigma_tran_late',
            'sigma_by_nominal', 'early_sigma_by_late_sigma',
            'stdev_by_late_sigma', 'mnshift_by_late_sigma',
            'skew_by_late_sigma', 'cross_signal_enc',
            'tran_nominal_by_delay_nominal', 'tran_late_by_delay_late',
            'mc_err_delay_late'
        ]

        available_cols = self.current_data.columns.tolist()
        feature_cols = [col for col in timing_features if col in available_cols]

        if len(feature_cols) < 3:
            feature_cols = [col for col in available_cols
                          if col != 'arc_pt' and self.current_data[col].dtype in ['float64', 'int64']]

        self.current_features = self.current_data[feature_cols].values
        print(f"Selected {len(feature_cols)} timing-critical features")

        observation = {
            'total_samples': len(self.current_data),
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
            'timing_statistics': {},
            'cell_types': {}
        }

        # Timing-specific analysis
        for col in feature_cols:
            if col in self.current_data.columns:
                observation['timing_statistics'][col] = {
                    'mean': float(self.current_data[col].mean()),
                    'std': float(self.current_data[col].std()),
                    'min': float(self.current_data[col].min()),
                    'max': float(self.current_data[col].max()),
                    'cv': float(self.current_data[col].std() / self.current_data[col].mean()) if self.current_data[col].mean() != 0 else 0
                }

        # Process variation analysis
        corr_matrix = self.current_data[feature_cols].corr()
        timing_correlations = []

        for i, col1 in enumerate(feature_cols):
            for j, col2 in enumerate(feature_cols[i+1:], i+1):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    timing_correlations.append({
                        'feature1': col1,
                        'feature2': col2,
                        'correlation': float(corr_val)
                    })

        observation['high_correlations'] = timing_correlations

        # Cell type analysis
        try:
            if 'arc_pt' in self.current_data.columns:
                cell_types = self.current_data['arc_pt'].str.extract(r'^([A-Z0-9]+)')[0]
                observation['cell_types'] = cell_types.value_counts().to_dict()
            else:
                observation['cell_types'] = {'unknown': len(self.current_data)}
        except:
            observation['cell_types'] = {'unknown': len(self.current_data)}

        # CRITICAL FIX: Calculate actual statistics for prompt injection
        calculated_stats = []
        key_features = ['nominal_delay', 'lib_sigma_delay_late', 'sigma_by_nominal']

        for feature in key_features:
            if feature in feature_cols:
                stats = observation['timing_statistics'][feature]
                calculated_stats.append(f"- {feature}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, range={stats['min']:.4f} to {stats['max']:.4f}")

        correlation_details = []
        for corr in timing_correlations:
            correlation_details.append(f"- {corr['feature1']} vs {corr['feature2']}: r={corr['correlation']:.3f}")

        # Sigma_by_nominal specific analysis
        sigma_analysis = "No sigma_by_nominal data available"
        if 'sigma_by_nominal' in feature_cols and 'sigma_by_nominal' in observation['timing_statistics']:
            sigma_stats = observation['timing_statistics']['sigma_by_nominal']
            sigma_analysis = f"Range: {sigma_stats['min']:.3f} to {sigma_stats['max']:.3f}, Mean: {sigma_stats['mean']:.3f}, Std: {sigma_stats['std']:.3f}"

        # Generate timing domain observation with ACTUAL DATA
        target_count = int(observation['total_samples'] * target_percentage / 100)

        try:
            observe_prompt = TIMING_OBSERVE_PROMPT.format(
                total_samples=observation['total_samples'],
                target_count=target_count,
                target_percentage=target_percentage,
                n_features=observation['n_features'],
                n_cell_types=len(observation['cell_types']),
                calculated_stats='\n'.join(calculated_stats) if calculated_stats else "No key timing features found in dataset",
                correlation_details='\n'.join(correlation_details) if correlation_details else "No high correlations detected",
                sigma_analysis=sigma_analysis
            )
        except KeyError as e:
            print(f"[ERROR] Missing parameter in OBSERVE prompt: {e}")
            observe_prompt = f"Analyze this timing dataset with {observation['total_samples']} samples for {target_percentage}% selection."

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=observe_prompt)
        ])

        chain = prompt_template | self.llm
        observation_reasoning = chain.invoke({})

        if hasattr(observation_reasoning, 'content'):
            observation_text = observation_reasoning.content
        else:
            observation_text = str(observation_reasoning)

        self.add_message('assistant', observation_text)
        self.log_reasoning('OBSERVE', observation_text)

        return observation

    def think(self, observation: Dict[str, Any], target_percentage: float) -> Dict[str, Any]:
        """THINK stage with timing strategy reasoning."""
        self._load_imports()
        print("\nSTAGE 2: THINK (Strategic Timing Analysis)")
        print("-" * 80)

        target_count = int(observation['total_samples'] * target_percentage / 100)

        # Build exploration findings summary for agentic prompts
        exploration_findings = f"""Dataset Analysis Results:
- Total Samples: {observation['total_samples']:,}
- Target Selection: {target_percentage:.1f}% = {target_count:,} samples
- Feature Dimensions: {observation['n_features']}
- Cell Type Diversity: {len(observation['cell_types'])} types
- High Correlations Found: {len(observation['high_correlations'])} pairs
- Statistical Complexity: {len(observation.get('timing_statistics', {}))} timing features analyzed

Key Findings: This dataset shows {'high' if len(observation['high_correlations']) > 3 else 'moderate'} correlation complexity
and {'diverse' if len(observation['cell_types']) > 10 else 'limited'} cell type diversity, suggesting
{'advanced clustering strategies' if len(observation['high_correlations']) > 3 else 'standard sampling approaches'} may be optimal."""

        # Debug: Print parameters being passed
        if self.verbose:
            print(f"[DEBUG] Formatting THINK prompt with:")
            print(f"  - exploration_findings length: {len(exploration_findings)}")
            print(f"  - target_count: {target_count}")

        try:
            think_prompt = TIMING_THINK_PROMPT.format(
                exploration_findings=exploration_findings,
                target_count=target_count
            )
        except KeyError as e:
            print(f"[ERROR] Missing parameter in THINK prompt: {e}")
            # Provide fallback prompt without formatting
            think_prompt = f"Develop a sampling strategy for {target_count} samples from this timing dataset. Use the exploration findings to guide your approach."

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=think_prompt)
        ])

        chain = prompt_template | self.llm
        thinking_reasoning = chain.invoke({})

        if hasattr(thinking_reasoning, 'content'):
            thinking_text = thinking_reasoning.content
        else:
            thinking_text = str(thinking_reasoning)

        self.add_message('assistant', thinking_text)
        self.log_reasoning('THINK', thinking_text)

        strategy = {
            'target_percentage': target_percentage,
            'target_count': target_count,
            'use_pca': True,
            'variance_threshold': 0.92,
            'n_clusters_range': [8, 10, 12],
            'selection_method': 'uncertainty_based',
            'timing_focus': True,
            'reasoning': thinking_text,
            'exploration_findings': exploration_findings
        }

        return strategy

    def decide(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """DECIDE stage with timing algorithm selection."""
        self._load_imports()
        print("\nSTAGE 3: DECIDE (Timing Algorithm Selection)")
        print("-" * 80)

        # PCA for timing feature compression
        print("Applying PCA for timing feature optimization...")
        features_scaled = self.scaler.fit_transform(self.current_features)
        pca = PCA()
        pca.fit(features_scaled)

        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= strategy['variance_threshold']) + 1

        pca_final = PCA(n_components=n_components)
        features_pca = pca_final.fit_transform(features_scaled)

        print(f"PCA: {len(pca.explained_variance_ratio_)} -> {n_components} components")
        print(f"Variance preserved: {cumsum[n_components-1]*100:.1f}%")

        # Timing-aware clustering comparison
        print("\nTesting clustering algorithms for timing data...")
        results = {}
        metrics = []

        for k in strategy['n_clusters_range']:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(features_pca)
            kmeans_inertia = kmeans.inertia_

            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm_labels = gmm.fit_predict(features_pca)
            gmm_bic = gmm.bic(features_pca)

            results[k] = {
                'kmeans': {'inertia': kmeans_inertia, 'labels': kmeans_labels},
                'gmm': {'bic': gmm_bic, 'labels': gmm_labels}
            }

            metric_str = f"k={k}: K-means inertia={kmeans_inertia:.0f}, GMM BIC={gmm_bic:.0f}"
            metrics.append(metric_str)
            print(f"  {metric_str}")

        # Calculate assessment based on variance explained
        variance_pct = cumsum[n_components-1]
        if variance_pct > 0.9:
            assessment = 'Excellent'
        elif variance_pct > 0.85:
            assessment = 'Good'
        else:
            assessment = 'Acceptable'

        decide_prompt = TIMING_DECIDE_PROMPT.format(
            original_features=len(self.current_features[0]),
            pca_components=n_components,
            variance_explained=variance_pct*100,
            assessment=assessment,
            clustering_metrics='\n'.join(metrics)
        )

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=decide_prompt)
        ])

        chain = prompt_template | self.llm
        decision_reasoning = chain.invoke({})

        if hasattr(decision_reasoning, 'content'):
            decision_text = decision_reasoning.content
        else:
            decision_text = str(decision_reasoning)

        best_k = self._extract_cluster_count(decision_text)
        best_algo = 'gmm' if 'gmm' in decision_text.lower() or 'gaussian' in decision_text.lower() else 'kmeans'

        self.add_message('assistant', decision_text)
        self.log_reasoning('DECIDE', decision_text)

        # Fit final model
        if best_algo == 'gmm':
            final_model = GaussianMixture(n_components=best_k, random_state=42)
            final_labels = final_model.fit_predict(features_pca)
            centroids = final_model.means_
        else:
            final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            final_labels = final_model.fit_predict(features_pca)
            centroids = final_model.cluster_centers_

        distances = cdist(features_pca, centroids, metric='euclidean')
        min_distances = np.min(distances, axis=1)

        decision = {
            'pca': {
                'n_components': n_components,
                'variance_explained': float(cumsum[n_components-1]),
                'transformer': pca_final
            },
            'clustering': {
                'algorithm': best_algo,
                'n_clusters': best_k,
                'model': final_model,
                'labels': final_labels,
                'centroids': centroids,
                'distances': min_distances
            },
            'features_pca': features_pca
        }

        return decision

    def act(self, decision: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """ACT stage with timing-optimized uncertainty sampling."""
        self._load_imports()
        print("\nSTAGE 4: ACT (Timing-Optimized Sample Selection)")
        print("-" * 80)

        target_count = strategy['target_count']
        labels = decision['clustering']['labels']
        distances = decision['clustering']['distances']
        n_clusters = decision['clustering']['n_clusters']

        cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
        base_per_cluster = [int(size * strategy['target_percentage'] / 100) for size in cluster_sizes]

        # Timing-aware uncertainty boosting
        cluster_uncertainties = []
        for i in range(n_clusters):
            mask = labels == i
            if np.any(mask):
                uncertainty = np.percentile(distances[mask], 90)
            else:
                uncertainty = 0
            cluster_uncertainties.append(uncertainty)

        max_uncertainty = max(cluster_uncertainties) if cluster_uncertainties else 1
        uncertainty_weights = [u / max_uncertainty for u in cluster_uncertainties]

        # Enhanced allocation for timing corners
        adjusted_per_cluster = []
        for i in range(n_clusters):
            base = base_per_cluster[i]
            boost = int(base * uncertainty_weights[i] * 0.3)
            adjusted_per_cluster.append(base + boost)

        total_adjusted = sum(adjusted_per_cluster)
        final_per_cluster = [int(c * target_count / total_adjusted) for c in adjusted_per_cluster]

        diff = target_count - sum(final_per_cluster)
        if diff > 0:
            highest_uncertainty_cluster = np.argmax(uncertainty_weights)
            final_per_cluster[highest_uncertainty_cluster] += diff

        # Uncertainty-based selection (samples far from centroids)
        selected_indices = []
        selection_details = []

        for i in range(n_clusters):
            mask = labels == i
            cluster_indices = np.where(mask)[0]
            cluster_distances = distances[mask]

            n_select = final_per_cluster[i]

            sorted_idx = np.argsort(cluster_distances)[::-1]
            selected = cluster_indices[sorted_idx[:n_select]]
            selected_indices.extend(selected.tolist())

            selection_details.append(
                f"Cluster {i}: {n_select}/{cluster_sizes[i]} samples ({n_select/cluster_sizes[i]*100:.1f}%)"
            )
            print(f"  {selection_details[-1]}")

        selected_df = self.current_data.iloc[selected_indices].copy()
        selected_df['cluster_id'] = labels[selected_indices]
        selected_df['uncertainty_score'] = distances[selected_indices]

        act_prompt = TIMING_ACT_PROMPT.format(
            total_samples=len(self.current_data),
            target_percentage=strategy['target_percentage'],
            target_count=target_count,
            n_selected=len(selected_indices),
            n_clusters=n_clusters,
            selection_details='\n'.join(selection_details)
        )

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=act_prompt)
        ])

        chain = prompt_template | self.llm
        action_reasoning = chain.invoke({})

        if hasattr(action_reasoning, 'content'):
            action_text = action_reasoning.content
        else:
            action_text = str(action_reasoning)

        self.add_message('assistant', action_text)
        self.log_reasoning('ACT', action_text)

        result = {
            'selected_df': selected_df,
            'selected_indices': selected_indices,
            'n_selected': len(selected_indices),
            'cluster_distribution': final_per_cluster,
            'uncertainty_weights': uncertainty_weights,
            'selection_details': selection_details,
            'expected_cost_reduction': '50% (10% to 5% Monte Carlo coverage)'
        }

        return result

    def act_agentic(self, strategy: Dict[str, Any], target_percentage: float) -> Dict[str, Any]:
        """Agentic ACT stage with autonomous decision-making and self-validation."""
        self._load_imports()
        print("\nSTAGE 3: AGENTIC EXECUTION (Autonomous Decision + Action)")
        print("-" * 80)

        # Generate execution plan with autonomous decision-making
        target_count = int(len(self.current_data) * target_percentage / 100)

        try:
            act_prompt = TIMING_ACT_PROMPT.format(
                exploration_findings=strategy.get('exploration_findings', 'Dataset exploration completed'),
                validated_strategy=strategy.get('reasoning', 'Autonomous strategy developed'),
                target_count=target_count,
                total_samples=len(self.current_data),
                algorithm_choice="Autonomous GMM clustering",
                algorithm_config="Adaptive parameters with self-validation"
            )
        except KeyError as e:
            print(f"[ERROR] Missing parameter in ACT prompt: {e}")
            act_prompt = f"Execute sampling strategy to select {target_count} samples from {len(self.current_data)} total samples."

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=act_prompt)
        ])

        chain = prompt_template | self.llm
        execution_reasoning = chain.invoke({})

        if hasattr(execution_reasoning, 'content'):
            execution_text = execution_reasoning.content
        else:
            execution_text = str(execution_reasoning)

        self.add_message('assistant', execution_text)
        self.log_reasoning('AGENTIC EXECUTION', execution_text)

        # Execute the sampling with iteration capability
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Apply PCA and clustering (similar to standard decide logic)
                features_scaled = self.scaler.fit_transform(self.current_features)
                pca = PCA()
                pca.fit(features_scaled)

                cumsum = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumsum >= 0.92) + 1

                pca_final = PCA(n_components=n_components)
                features_pca = pca_final.fit_transform(features_scaled)

                # Use GMM as default for agentic mode (handles overlapping distributions better)
                best_k = min(10, len(self.current_data) // 100)  # Adaptive cluster count
                final_model = GaussianMixture(n_components=best_k, random_state=42)
                final_labels = final_model.fit_predict(features_pca)
                centroids = final_model.means_

                distances = cdist(features_pca, centroids, metric='euclidean')
                min_distances = np.min(distances, axis=1)

                # Use the target_count calculated earlier (no recalculation needed)

                # Select samples far from centroids (high uncertainty)
                uncertainty_scores = min_distances
                selected_indices = np.argsort(-uncertainty_scores)[:target_count]

                # Validate selection quality
                validation_results = self.validate_selection_quality(selected_indices, final_labels)

                if not validation_results.get('requires_iteration', False) or attempt == max_attempts - 1:
                    # Selection is good or we've exhausted attempts
                    break
                else:
                    print(f"\n[ITERATION {attempt + 1}] Validation failed, adjusting strategy...")
                    # Adjust target count or approach for next iteration
                    target_count = int(target_count * 1.1)  # Slightly increase sample count

            except Exception as e:
                print(f"[ERROR] Execution attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise

        # Create results
        selected_df = self.current_data.iloc[selected_indices].copy()
        selected_df['cluster_id'] = final_labels[selected_indices]
        selected_df['uncertainty_score'] = uncertainty_scores[selected_indices]

        result = {
            'selected_df': selected_df,
            'selected_indices': selected_indices,
            'n_selected': len(selected_indices),
            'cluster_distribution': [np.sum(final_labels[selected_indices] == i) for i in range(best_k)],
            'uncertainty_weights': uncertainty_scores[selected_indices],
            'selection_details': [f"Selected {len(selected_indices)} high-uncertainty samples"],
            'expected_cost_reduction': f'{100 - target_percentage:.0f}% reduction (from 10% to {target_percentage:.1f}%)',
            'validation_results': validation_results,
            'agentic_iterations': attempt + 1
        }

        # Create synthetic decision structure for compatibility
        decision = {
            'pca': {
                'n_components': n_components,
                'variance_explained': float(cumsum[n_components-1]),
                'transformer': pca_final
            },
            'clustering': {
                'algorithm': 'gmm',
                'n_clusters': best_k,
                'model': final_model,
                'labels': final_labels,
                'centroids': centroids,
                'distances': min_distances
            },
            'features_pca': features_pca
        }

        # Store decision for return compatibility
        self.last_decision = decision

        return result

    def run_selection(self, user_query: str, csv_path: str) -> Dict[str, Any]:
        """Main workflow with timing domain expertise."""
        self._load_imports()
        print("\n" + "=" * 80)
        print("TIMING-AWARE DATA SELECTION AGENT")
        print("=" * 80)

        self.add_message('user', user_query)

        print("\nParsing timing engineer requirements...")
        params = self.parse_user_query(user_query)

        observation = self.observe(csv_path, params.get('selection_percentage', 5.0))

        # Handle null percentage - determine optimal percentage based on actual dataset size
        if params.get('selection_percentage') is None:
            data_size = len(self.current_data)
            if data_size > 50000:
                optimal_percentage = 3.0  # Large datasets need less percentage
            elif data_size > 20000:
                optimal_percentage = 5.0  # Medium datasets
            else:
                optimal_percentage = 8.0  # Smaller datasets can afford higher percentage

            params['selection_percentage'] = optimal_percentage
            print(f"No percentage specified. Data-driven selection: {optimal_percentage}% for {data_size:,} samples")

        strategy = self.think(observation, params['selection_percentage'])

        if self.agentic_mode:
            # In agentic mode, strategy includes decision-making
            # Skip separate decide stage and let act handle both
            print("\n[AGENTIC] Strategy includes autonomous decision-making")
            result = self.act_agentic(strategy, params['selection_percentage'])
        else:
            # Standard mode with separate decide stage
            decision = self.decide(strategy)
            result = self.act(decision, strategy)

        print("\n" + "=" * 80)
        print("TIMING-OPTIMIZED SELECTION COMPLETE")
        print("=" * 80)
        print(f"Selected {result['n_selected']}/{len(self.current_data)} samples ({result['n_selected']/len(self.current_data)*100:.1f}%)")
        print(f"Expected cost reduction: {result['expected_cost_reduction']}")
        print(f"Active learning: Uncertainty-based sampling for timing robustness")

        # Handle decision structure for agentic vs standard mode
        if self.agentic_mode:
            decision = getattr(self, 'last_decision', {})

        return {
            'observation': observation,
            'strategy': strategy,
            'decision': decision,
            'result': result,
            'reasoning_log': self.reasoning_log,
            'conversation_history': self.conversation_history,
            'parsed_params': params,
            'agentic_mode': self.agentic_mode
        }

    def _extract_cluster_count(self, text: str) -> int:
        """Extract cluster count from LLM response."""
        try:
            matches = re.findall(r'(\d+)\s*cluster', text, re.IGNORECASE)
            if matches:
                return int(matches[0])

            matches = re.findall(r'k\s*=\s*(\d+)', text)
            if matches:
                return int(matches[0])

            matches = re.findall(r'(?:optimal|best|choose).*?(\d+)', text, re.IGNORECASE)
            if matches:
                return int(matches[0])
        except:
            pass

        return 10

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history

    # SAFE ALLOCATION METHODS
    def safe_int(self, value: Union[int, float, str, None], default: int = 0) -> int:
        """Convert any value to safe integer, defaulting to 0 for None/invalid values."""
        if value is None:
            return default
        try:
            if isinstance(value, str):
                # Extract numbers from string responses like "30 samples" or "None allocated"
                numbers = re.findall(r'\d+', value)
                return int(numbers[0]) if numbers else default
            return max(0, int(float(value)))  # Ensure non-negative
        except (ValueError, TypeError):
            return default

    def safe_sample_allocation(self, strategy_results: Dict[str, Any], total_target: int) -> Dict[str, int]:
        """Safely allocate samples across different strategies with None-type protection."""

        # Define all possible allocation strategies
        allocation_strategies = [
            'grid_sampling',
            'uncertainty_sampling',
            'boundary_sampling',
            'sparse_region_exploration',
            'validation_holdout',
            'representative_coverage',
            'corner_case_sampling'
        ]

        # Extract and safely convert all allocations
        safe_allocations = {}
        total_allocated = 0

        for strategy in allocation_strategies:
            raw_value = strategy_results.get(strategy, 0)
            safe_count = self.safe_int(raw_value, 0)
            safe_allocations[strategy] = safe_count
            total_allocated += safe_count

        # Handle over/under allocation
        if total_allocated > total_target:
            print(f"[WARNING] Over-allocation detected: {total_allocated} > {total_target}")
            # Proportionally reduce all non-zero allocations
            scale_factor = total_target / total_allocated
            for strategy in safe_allocations:
                if safe_allocations[strategy] > 0:
                    safe_allocations[strategy] = max(1, int(safe_allocations[strategy] * scale_factor))

            # Recalculate total after scaling
            total_allocated = sum(safe_allocations.values())

        # Handle under-allocation by adding to largest strategy
        if total_allocated < total_target:
            remaining = total_target - total_allocated
            largest_strategy = max(safe_allocations.keys(), key=lambda k: safe_allocations[k])
            safe_allocations[largest_strategy] += remaining
            print(f"[INFO] Added {remaining} samples to {largest_strategy} to reach target")

        # Final validation
        final_total = sum(safe_allocations.values())
        assert final_total == total_target, f"Allocation error: {final_total} != {total_target}"

        return safe_allocations

    # INTENT CLASSIFICATION METHODS
    def classify_user_intent(self, user_input: str) -> Tuple[UserIntent, Dict[str, Any]]:
        """Classify user intent to determine whether to execute pipeline or answer from context."""

        input_lower = user_input.lower().strip()

        # Intent patterns with priorities (most specific first)
        intent_patterns = {
            UserIntent.QUESTION_ABOUT_RESULTS: [
                r'why did you (choose|pick|select)',
                r'why.*(\\d+)%',
                r'explain (the|your) (selection|choice|decision)',
                r'how did you (decide|determine)',
                r'what.*reasoning.*behind',
                r'justify.*selection',
                r'(why|how).*samples',
                r'rationale.*for'
            ],

            UserIntent.MODIFY_PARAMETERS: [
                r'change.*to.*(\\d+)%',
                r'try.*(\\d+)%.*instead',
                r'use.*(\\d+).*samples',
                r'increase.*to.*(\\d+)',
                r'decrease.*to.*(\\d+)',
                r'modify.*percentage',
                r'adjust.*selection'
            ],

            UserIntent.REQUEST_VISUALIZATION: [
                r'show.*plot',
                r'visuali[sz]e.*results',
                r'generate.*dashboard',
                r'plot.*samples',
                r'show.*scatter',
                r'display.*chart'
            ],

            UserIntent.EXPLAIN_METHODOLOGY: [
                r'how does.*work',
                r'explain.*algorithm',
                r'what.*method.*using',
                r'describe.*approach',
                r'methodology'
            ],

            UserIntent.EXECUTE_SAMPLING: [
                r'select.*(\\d+)%',
                r'run.*sampling',
                r'perform.*selection',
                r'execute.*analysis',
                r'analyze.*dataset',
                r'sample.*(\\d+)',
                r'choose.*samples'
            ],

            UserIntent.GENERAL_HELP: [
                r'help',
                r'what.*can.*do',
                r'how.*use',
                r'commands'
            ]
        }

        # Check patterns in order of priority
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, input_lower)
                if match:
                    # Extract parameters from match
                    params = self._extract_intent_parameters(user_input, match)
                    return intent, params

        # Default to execution if no clear pattern matches
        return UserIntent.EXECUTE_SAMPLING, {}

    def _extract_intent_parameters(self, user_input: str, match: re.Match) -> Dict[str, Any]:
        """Extract parameters from matched intent patterns."""
        params = {}

        # Extract percentages
        percentage_matches = re.findall(r'(\\d+)%', user_input)
        if percentage_matches:
            params['percentage'] = int(percentage_matches[0])

        # Extract sample counts
        sample_matches = re.findall(r'(\\d+)\\s*samples?', user_input, re.IGNORECASE)
        if sample_matches:
            params['sample_count'] = int(sample_matches[0])

        # Extract algorithm names
        algorithm_matches = re.findall(r'(k-means|gmm|clustering)', user_input, re.IGNORECASE)
        if algorithm_matches:
            params['algorithm'] = algorithm_matches[0].lower()

        return params

    # INTERACTIVE VISUALIZATION METHODS
    def generate_interactive_dashboard(self, df: pd.DataFrame, selected_indices: List[int],
                                     clusters: np.ndarray, centroids: np.ndarray,
                                     pca_components: Optional[np.ndarray] = None,
                                     export_html: bool = True) -> Dict[str, Any]:
        """Generate comprehensive interactive dashboard with Plotly."""

        if not PLOTLY_AVAILABLE:
            print("[WARNING] Plotly not available - using fallback visualization")
            return self._fallback_visualization(df, selected_indices, clusters)

        # Prepare data
        dashboard_data = self._prepare_dashboard_data(df, selected_indices, clusters, pca_components)

        # Create main subplot structure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Sample Selection Overview (Interactive)',
                'Cluster Quality & Coverage Analysis',
                'Selection Statistics by Cluster',
                'Data Distribution Validation'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "histogram"}]
            ],
            horizontal_spacing=0.12,
            vertical_spacing=0.12
        )

        # Add plots
        self._add_selection_overview(fig, dashboard_data, row=1, col=1)
        self._add_cluster_analysis(fig, dashboard_data, centroids, row=1, col=2)
        self._add_selection_statistics(fig, dashboard_data, row=2, col=1)
        self._add_distribution_analysis(fig, dashboard_data, row=2, col=2)

        # Update layout for interactivity
        fig.update_layout(
            title={
                'text': 'Agentic Timing Data Selection Dashboard',
                'x': 0.5,
                'font': {'size': 24, 'color': 'white'}
            },
            template='plotly_dark',
            showlegend=True,
            height=800,
            width=1200,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='#2F3542'
        )

        # Export standalone HTML if requested
        html_path = None
        if export_html:
            html_path = self._export_html_dashboard(fig, dashboard_data)

        return {
            'plotly_figure': fig,
            'dashboard_data': dashboard_data,
            'html_export_path': html_path,
            'interactive_features': {
                'zoom': True,
                'pan': True,
                'hover_details': True,
                'toggleable_traces': True,
                'selection_tools': True
            }
        }

    def _prepare_dashboard_data(self, df: pd.DataFrame, selected_indices: List[int],
                               clusters: np.ndarray, pca_components: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Prepare and structure data for dashboard plotting."""

        # Create selection mask
        selection_mask = np.zeros(len(df), dtype=bool)
        selection_mask[selected_indices] = True

        # Prepare coordinate data
        if pca_components is not None and pca_components.shape[1] >= 2:
            x_coords = pca_components[:, 0]
            y_coords = pca_components[:, 1]
            coord_labels = ('PCA Component 1', 'PCA Component 2')
        else:
            # Use first two numeric columns as fallback
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:2]
            x_coords = df[numeric_cols[0]].values if len(numeric_cols) > 0 else np.arange(len(df))
            y_coords = df[numeric_cols[1]].values if len(numeric_cols) > 1 else np.random.random(len(df))
            coord_labels = (numeric_cols[0] if len(numeric_cols) > 0 else 'Index',
                          numeric_cols[1] if len(numeric_cols) > 1 else 'Random')

        # Calculate cluster statistics
        unique_clusters = np.unique(clusters)
        cluster_stats = {}

        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            selected_in_cluster = np.sum(selection_mask & cluster_mask)
            total_in_cluster = np.sum(cluster_mask)

            cluster_stats[cluster_id] = {
                'total_samples': total_in_cluster,
                'selected_samples': selected_in_cluster,
                'selection_rate': selected_in_cluster / total_in_cluster if total_in_cluster > 0 else 0,
                'cluster_size_pct': total_in_cluster / len(df) * 100
            }

        return {
            'coordinates': {
                'x': x_coords,
                'y': y_coords,
                'labels': coord_labels
            },
            'selection_mask': selection_mask,
            'clusters': clusters,
            'cluster_stats': cluster_stats,
            'summary': {
                'total_samples': len(df),
                'selected_count': len(selected_indices),
                'selection_percentage': len(selected_indices) / len(df) * 100,
                'num_clusters': len(unique_clusters)
            }
        }

    def _add_selection_overview(self, fig, dashboard_data: Dict, row: int, col: int):
        """Add interactive sample selection overview plot."""

        coords = dashboard_data['coordinates']
        selection_mask = dashboard_data['selection_mask']
        clusters = dashboard_data['clusters']

        # Color palette
        selected_color = '#FF6B6B'  # Red for selected samples
        unselected_color = '#4ECDC4'  # Teal for unselected

        # Unselected samples (background)
        unselected_mask = ~selection_mask
        fig.add_trace(
            go.Scatter(
                x=coords['x'][unselected_mask],
                y=coords['y'][unselected_mask],
                mode='markers',
                marker=dict(
                    color=clusters[unselected_mask],
                    colorscale='Viridis',
                    size=6,
                    opacity=0.4,
                    line=dict(width=1, color='white')
                ),
                name='Unselected',
                hovertemplate='<b>Unselected Sample</b><br>' +
                            f'{coords["labels"][0]}: %{{x:.3f}}<br>' +
                            f'{coords["labels"][1]}: %{{y:.3f}}<br>' +
                            'Cluster: %{marker.color}<extra></extra>',
                showlegend=True
            ),
            row=row, col=col
        )

        # Selected samples (highlighted)
        selected_mask = selection_mask
        fig.add_trace(
            go.Scatter(
                x=coords['x'][selected_mask],
                y=coords['y'][selected_mask],
                mode='markers',
                marker=dict(
                    color=selected_color,
                    size=10,
                    opacity=0.8,
                    line=dict(width=2, color='white'),
                    symbol='diamond'
                ),
                name='Selected',
                hovertemplate='<b>SELECTED Sample</b><br>' +
                            f'{coords["labels"][0]}: %{{x:.3f}}<br>' +
                            f'{coords["labels"][1]}: %{{y:.3f}}<br>' +
                            'Status: Selected for analysis<extra></extra>',
                showlegend=True
            ),
            row=row, col=col
        )

        # Update axes
        fig.update_xaxes(title_text=coords['labels'][0], row=row, col=col)
        fig.update_yaxes(title_text=coords['labels'][1], row=row, col=col)

    def _add_cluster_analysis(self, fig, dashboard_data: Dict, centroids: np.ndarray, row: int, col: int):
        """Add cluster quality and coverage analysis."""

        coords = dashboard_data['coordinates']
        clusters = dashboard_data['clusters']
        cluster_stats = dashboard_data['cluster_stats']

        # Plot cluster centers
        unique_clusters = np.unique(clusters)
        cluster_colors = px.colors.qualitative.Set3[:len(unique_clusters)]

        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = clusters == cluster_id
            stats = cluster_stats[cluster_id]

            # Cluster samples
            fig.add_trace(
                go.Scatter(
                    x=coords['x'][cluster_mask],
                    y=coords['y'][cluster_mask],
                    mode='markers',
                    marker=dict(
                        color=cluster_colors[i],
                        size=7,
                        opacity=0.6,
                        line=dict(width=1, color='white')
                    ),
                    name=f'Cluster {cluster_id}',
                    hovertemplate=f'<b>Cluster {cluster_id}</b><br>' +
                                f'{coords["labels"][0]}: %{{x:.3f}}<br>' +
                                f'{coords["labels"][1]}: %{{y:.3f}}<br>' +
                                f'Selected: {stats["selected_samples"]}/{stats["total_samples"]}<br>' +
                                f'Rate: {stats["selection_rate"]:.1%}<extra></extra>',
                    showlegend=True
                ),
                row=row, col=col
            )

            # Cluster centroid
            if i < len(centroids):
                fig.add_trace(
                    go.Scatter(
                        x=[centroids[i, 0]],
                        y=[centroids[i, 1]] if centroids.shape[1] > 1 else [0],
                        mode='markers',
                        marker=dict(
                            color='black',
                            size=15,
                            symbol='x',
                            line=dict(width=3, color=cluster_colors[i])
                        ),
                        name=f'Centroid {cluster_id}',
                        hovertemplate=f'<b>Cluster {cluster_id} Centroid</b><br>' +
                                    'Representative center point<extra></extra>',
                        showlegend=False
                    ),
                    row=row, col=col
                )

        # Update axes
        fig.update_xaxes(title_text=coords['labels'][0], row=row, col=col)
        fig.update_yaxes(title_text=coords['labels'][1], row=row, col=col)

    def _add_selection_statistics(self, fig, dashboard_data: Dict, row: int, col: int):
        """Add selection statistics bar chart."""

        cluster_stats = dashboard_data['cluster_stats']
        selected_color = '#FF6B6B'
        unselected_color = '#4ECDC4'

        cluster_ids = list(cluster_stats.keys())
        selected_counts = [stats['selected_samples'] for stats in cluster_stats.values()]
        total_counts = [stats['total_samples'] for stats in cluster_stats.values()]
        selection_rates = [stats['selection_rate'] * 100 for stats in cluster_stats.values()]

        # Selected samples bar
        fig.add_trace(
            go.Bar(
                x=[f'Cluster {cid}' for cid in cluster_ids],
                y=selected_counts,
                name='Selected',
                marker_color=selected_color,
                hovertemplate='<b>%{x}</b><br>Selected: %{y}<br>Rate: %{customdata:.1f}%<extra></extra>',
                customdata=selection_rates
            ),
            row=row, col=col
        )

        # Total samples outline
        fig.add_trace(
            go.Bar(
                x=[f'Cluster {cid}' for cid in cluster_ids],
                y=total_counts,
                name='Total Available',
                marker=dict(
                    color='rgba(255,255,255,0)',
                    line=dict(color=unselected_color, width=2)
                ),
                hovertemplate='<b>%{x}</b><br>Total: %{y}<br>Coverage: %{customdata:.1f}%<extra></extra>',
                customdata=[stats['cluster_size_pct'] for stats in cluster_stats.values()]
            ),
            row=row, col=col
        )

        # Update axes
        fig.update_xaxes(title_text='Clusters', row=row, col=col)
        fig.update_yaxes(title_text='Sample Count', row=row, col=col)

    def _add_distribution_analysis(self, fig, dashboard_data: Dict, row: int, col: int):
        """Add data distribution histogram."""

        coords = dashboard_data['coordinates']
        selection_mask = dashboard_data['selection_mask']
        selected_color = '#FF6B6B'
        unselected_color = '#4ECDC4'

        # All data histogram
        fig.add_trace(
            go.Histogram(
                x=coords['x'],
                name='All Data',
                opacity=0.5,
                marker_color=unselected_color,
                nbinsx=30,
                hovertemplate='<b>All Data</b><br>Range: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=row, col=col
        )

        # Selected data histogram
        fig.add_trace(
            go.Histogram(
                x=coords['x'][selection_mask],
                name='Selected Data',
                opacity=0.7,
                marker_color=selected_color,
                nbinsx=30,
                hovertemplate='<b>Selected Data</b><br>Range: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=row, col=col
        )

        # Update axes
        fig.update_xaxes(title_text=coords['labels'][0], row=row, col=col)
        fig.update_yaxes(title_text='Frequency', row=row, col=col)

    def _export_html_dashboard(self, fig, dashboard_data: Dict) -> str:
        """Export interactive dashboard as standalone HTML."""

        summary = dashboard_data['summary']
        html_filename = f"timing_dashboard_{summary['selected_count']}_samples.html"
        html_path = os.path.join(os.getcwd(), html_filename)

        # Add summary annotation
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text=f"<b>Summary:</b> {summary['selected_count']:,} samples selected " +
                 f"({summary['selection_percentage']:.1f}%) from {summary['total_samples']:,} total " +
                 f"across {summary['num_clusters']} clusters",
            showarrow=False,
            font=dict(size=14, color="white"),
            align="left",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=1
        )

        # Export with full interactivity
        fig.write_html(
            html_path,
            include_plotlyjs=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'timing_dashboard',
                    'height': 800,
                    'width': 1200,
                    'scale': 2
                }
            }
        )

        print(f"Interactive dashboard exported: {html_path}")
        return html_path

    def _fallback_visualization(self, df: pd.DataFrame, selected_indices: List[int], clusters: np.ndarray) -> Dict[str, Any]:
        """Fallback visualization when Plotly is not available."""

        print("[INFO] Using basic fallback visualization")

        # Create simple text-based summary
        total_samples = len(df)
        selected_count = len(selected_indices)
        selection_percentage = selected_count / total_samples * 100
        num_clusters = len(np.unique(clusters))

        summary_text = f"""
        TIMING DATA SELECTION SUMMARY
        =============================
        Total samples: {total_samples:,}
        Selected: {selected_count:,} ({selection_percentage:.1f}%)
        Clusters: {num_clusters}

        Selection distributed across {num_clusters} timing clusters
        using uncertainty-based sampling for critical corner coverage.
        """

        return {
            'summary_text': summary_text,
            'dashboard_data': {
                'total_samples': total_samples,
                'selected_count': selected_count,
                'selection_percentage': selection_percentage,
                'num_clusters': num_clusters
            },
            'plotly_figure': None,
            'html_export_path': None,
            'interactive_features': {'text_summary': True}
        }