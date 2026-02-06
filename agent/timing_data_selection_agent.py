"""
Timing-Aware Data Selection Agent
Senior timing engineer expertise for Monte Carlo sample selection
"""

from typing import Dict, List, Any, Optional
import json
import re
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

        global pd, np, PCA, StandardScaler, KMeans, GaussianMixture, cdist
        global ChatPromptTemplate, HumanMessage, SystemMessage

        import pandas as pd
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.mixture import GaussianMixture
        from scipy.spatial.distance import cdist

        from langchain.prompts import ChatPromptTemplate
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError:
            from langchain.schema import HumanMessage, SystemMessage

        if self.scaler is None:
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

    def observe(self, csv_path: str) -> Dict[str, Any]:
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
        observe_prompt = TIMING_OBSERVE_PROMPT.format(
            total_samples=observation['total_samples'],
            n_features=observation['n_features'],
            n_cell_types=len(observation['cell_types']),
            n_high_corr=len(timing_correlations),
            calculated_stats='\n'.join(calculated_stats) if calculated_stats else "No key timing features found in dataset",
            correlation_details='\n'.join(correlation_details) if correlation_details else "No high correlations detected",
            sigma_analysis=sigma_analysis
        )

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

        think_prompt = TIMING_THINK_PROMPT.format(
            total_samples=observation['total_samples'],
            target_percentage=target_percentage,
            target_count=target_count,
            n_features=observation['n_features'],
            n_cell_types=len(observation['cell_types']),
            n_high_corr=len(observation['high_correlations'])
        )

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
            'timing_focus': True
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

        observation_summary = f"""Dataset has {len(self.current_data)} samples with {self.current_features.shape[1]} features.
Target selection: {target_percentage:.1f}% ({int(len(self.current_data) * target_percentage / 100)} samples)
Strategy reasoning: {strategy.get('reasoning', 'Strategic analysis completed')}"""

        # Generate execution plan with autonomous decision-making
        act_prompt = TIMING_ACT_PROMPT.format(
            exploration_findings=observation_summary,
            validated_strategy=strategy.get('reasoning', 'Autonomous strategy developed'),
            target_count=int(len(self.current_data) * target_percentage / 100),
            algorithm_choice="Autonomous selection",
            algorithm_config="Self-optimizing parameters"
        )

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

                # Uncertainty-based sampling
                target_count = int(len(self.current_data) * target_percentage / 100)

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

        observation = self.observe(csv_path)

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