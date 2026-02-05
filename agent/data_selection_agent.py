"""
LangChain-based Data Selection Agent
Supports natural language queries for intelligent sample selection
Terminal/CLI version - no markdown symbols
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from typing import Dict, List, Any, Optional
import json
from langchain.prompts import ChatPromptTemplate
from langchain.llms.base import LLM
from langchain.schema import HumanMessage, SystemMessage
import re


class DataSelectionAgent:
    """
    LangChain-based agent for intelligent data selection.
    
    Workflow:
    1. Parse user query to extract selection requirements
    2. OBSERVE: Analyze data characteristics
    3. THINK: Reason about selection strategy
    4. DECIDE: Choose clustering parameters
    5. ACT: Execute selection and explain reasoning
    """
    
    def __init__(self, llm: LLM, verbose: bool = True):
        """
        Initialize the agent.
        
        Args:
            llm: LangChain LLM instance (e.g., Ollama)
            verbose: Whether to print reasoning steps
        """
        self.llm = llm
        self.verbose = verbose
        self.conversation_history = []
        self.current_data = None
        self.current_features = None
        self.scaler = StandardScaler()
        self.reasoning_log = []
        
        # System prompt for consistency
        self.system_prompt = """You are an expert AI assistant specializing in data analysis and machine learning.
Your role is to help users intelligently select representative training data from large datasets.

Demonstrate clear reasoning through these stages:
1. OBSERVE - Analyze data characteristics
2. THINK - Reason about selection strategy
3. DECIDE - Choose specific parameters
4. ACT - Execute and explain the selection

Be analytical, transparent in your reasoning, and provide specific recommendations backed by data analysis.
Do not use any special markdown symbols or emojis in your response."""
    
    def add_message(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': pd.Timestamp.now()
        })
    
    def log_reasoning(self, stage: str, content: str):
        """Log agent reasoning."""
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
    
    def parse_user_query(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query to extract parameters.
        
        Example: "I need to select 8% of training data for this CSV file"
        """
        parsing_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Parse this user query and extract data selection parameters.
            
Query: "{query}"

Extract and return ONLY valid JSON (no markdown, no extra text) with these fields:
- selection_percentage: float between 1 and 100 (e.g., 8.0)
- selection_criteria: string (e.g., "uncertainty", "diversity", "random")
- clustering_preference: string or null (e.g., "gmm", "kmeans", or null for auto)
- additional_requirements: string or null for any special requirements

Return ONLY the JSON object, nothing else.""")
        ])
        
        chain = parsing_prompt | self.llm
        response = chain.invoke({})
        
        # Extract text from response
        if hasattr(response, 'content'):
            text = response.content
        else:
            text = str(response)
        
        # Clean up the response - remove markdown code blocks if present
        text = text.replace('```json', '').replace('```', '').strip()
        
        # Parse JSON from response
        try:
            params = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    params = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # Default fallback
                    params = {
                        'selection_percentage': 5.0,
                        'selection_criteria': 'uncertainty',
                        'clustering_preference': None,
                        'additional_requirements': None
                    }
            else:
                # Default fallback
                params = {
                    'selection_percentage': 5.0,
                    'selection_criteria': 'uncertainty',
                    'clustering_preference': None,
                    'additional_requirements': None
                }
        
        self.add_message('assistant', f"Parsed your requirements: selection_percentage={params.get('selection_percentage')}%, criteria={params.get('selection_criteria')}")
        return params
    
    def observe(self, csv_path: str) -> Dict[str, Any]:
        """
        Stage 1: OBSERVE - Analyze data characteristics.
        """
        print("\nSTAGE 1: OBSERVE")
        print("-" * 80)
        
        # Load data
        self.current_data = pd.read_csv(csv_path)
        print(f"Loaded {len(self.current_data)} samples")
        
        # Feature selection
        preferred_cols = [
            'nominal_delay', 'lib_sigma_delay_late',
            'nominal_tran', 'lib_sigma_tran_late',
            'sigma_by_nominal', 'early_sigma_by_late_sigma',
            'stdev_by_late_sigma', 'cross_sigma_sec'
        ]
        
        available_cols = self.current_data.columns.tolist()
        feature_cols = [col for col in preferred_cols if col in available_cols]
        
        if len(feature_cols) < 3:
            feature_cols = [col for col in available_cols 
                          if col != 'arc_pt' and self.current_data[col].dtype in ['float64', 'int64']]
        
        self.current_features = self.current_data[feature_cols].values
        print(f"Using {len(feature_cols)} features")
        
        # Analysis
        observation = {
            'total_samples': len(self.current_data),
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
            'statistics': {},
            'cell_types': {}
        }
        
        # Statistics
        for col in feature_cols:
            observation['statistics'][col] = {
                'mean': float(self.current_data[col].mean()),
                'std': float(self.current_data[col].std()),
                'min': float(self.current_data[col].min()),
                'max': float(self.current_data[col].max()),
                'q25': float(self.current_data[col].quantile(0.25)),
                'q75': float(self.current_data[col].quantile(0.75))
            }
        
        # Cell type distribution
        try:
            cell_types = self.current_data['arc_pt'].str.extract(r'^([A-Z0-9]+)D')[0]
            observation['cell_types'] = cell_types.value_counts().to_dict()
        except:
            observation['cell_types'] = {}
        
        # Correlations
        corr_matrix = self.current_data[feature_cols].corr()
        high_corr = []
        for i in range(len(feature_cols)):
            for j in range(i+1, len(feature_cols)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append({
                        'feature1': feature_cols[i],
                        'feature2': feature_cols[j],
                        'correlation': float(corr_matrix.iloc[i, j])
                    })
        observation['high_correlations'] = high_corr
        
        # Generate observation with LLM
        observe_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Analyze this dataset and provide your OBSERVATION.

Dataset: {observation['total_samples']} samples, {observation['n_features']} features

Features: {', '.join(feature_cols)}

Cell Types: {len(observation['cell_types'])} types
Top types: {dict(list(observation['cell_types'].items())[:3])}

High Correlations: {len(high_corr)} pairs found

Provide a 2-3 sentence OBSERVATION analyzing:
1. Data structure and quality
2. Notable patterns or correlations  
3. Readiness for clustering and selection

Use plain text only, no special symbols.""")
        ])
        
        chain = observe_prompt | self.llm
        observation_reasoning = chain.invoke({})
        
        # Extract text from response
        if hasattr(observation_reasoning, 'content'):
            observation_text = observation_reasoning.content
        else:
            observation_text = str(observation_reasoning)
        
        self.add_message('assistant', observation_text)
        self.log_reasoning('OBSERVE', observation_text)
        
        return observation
    
    def think(self, observation: Dict[str, Any], target_percentage: float) -> Dict[str, Any]:
        """
        Stage 2: THINK - Reason about selection strategy.
        """
        print("\nSTAGE 2: THINK")
        print("-" * 80)
        
        target_count = int(observation['total_samples'] * target_percentage / 100)
        
        think_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Based on the dataset analysis, reason about the selection strategy.

Dataset Context:
- Total: {observation['total_samples']} samples
- Target: {target_percentage:.1f}% = {target_count} samples
- Features: {observation['n_features']} dimensions
- Cell types: {len(observation['cell_types'])} types
- High correlations: {len(observation['high_correlations'])} pairs

Think through these questions:
1. Why use clustering vs random sampling?
2. What clustering algorithm would work best?
3. How many clusters should we create?
4. Should we apply PCA first?
5. How do we ensure representative sampling?

Provide 3-4 sentences of strategic reasoning. Use plain text only.""")
        ])
        
        chain = think_prompt | self.llm
        thinking_reasoning = chain.invoke({})
        
        # Extract text from response
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
            'variance_threshold': 0.90,
            'n_clusters_range': [6, 8, 10],
            'selection_method': 'uncertainty_based',
            'pca_info': None,
            'clustering_info': None
        }
        
        return strategy
    
    def decide(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 3: DECIDE - Choose specific parameters and execute.
        """
        print("\nSTAGE 3: DECIDE")
        print("-" * 80)
        
        # Apply PCA
        print("Applying PCA...")
        features_scaled = self.scaler.fit_transform(self.current_features)
        pca = PCA()
        pca.fit(features_scaled)
        
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= strategy['variance_threshold']) + 1
        
        pca_final = PCA(n_components=n_components)
        features_pca = pca_final.fit_transform(features_scaled)
        
        print(f"PCA: {len(pca.explained_variance_ratio_)} -> {n_components} components")
        print(f"Variance explained: {cumsum[n_components-1]*100:.1f}%")
        
        strategy['pca_info'] = {
            'n_components': n_components,
            'variance_explained': float(cumsum[n_components-1]),
            'original_features': len(self.current_features[0])
        }
        
        # Test clustering
        print("\nTesting clustering algorithms...")
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
        
        # Use LLM to choose optimal parameters
        decide_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Based on clustering metrics, make your DECISION on parameters.

PCA Results:
- Reduced from {strategy['pca_info']['original_features']} to {n_components} dimensions
- Captured {strategy['pca_info']['variance_explained']*100:.1f}% of variance

Clustering Metrics Tested:
{chr(10).join(metrics)}

Make your DECISION (2-3 sentences) recommending:
1. Best clustering algorithm (K-means or GMM)
2. Optimal number of clusters
3. Brief justification

Use plain text only, no special symbols.""")
        ])
        
        chain = decide_prompt | self.llm
        decision_reasoning = chain.invoke({})
        
        # Extract text from response
        if hasattr(decision_reasoning, 'content'):
            decision_text = decision_reasoning.content
        else:
            decision_text = str(decision_reasoning)
        
        # Extract cluster count from LLM response
        best_k = self._extract_cluster_count(decision_text)
        best_algo = 'gmm' if 'gmm' in decision_text.lower() else 'kmeans'
        
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
        
        # Calculate distances
        distances = cdist(features_pca, centroids, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        
        # Cluster analysis
        cluster_info = []
        for i in range(best_k):
            mask = final_labels == i
            cluster_size = np.sum(mask)
            cluster_info.append({
                'id': i,
                'size': int(cluster_size),
                'percentage': float(cluster_size / len(self.current_data) * 100)
            })
        
        strategy['clustering_info'] = {
            'algorithm': best_algo,
            'n_clusters': best_k,
            'clusters': cluster_info
        }
        
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
            'features_pca': features_pca,
            'cluster_info': cluster_info
        }
        
        return decision
    
    def act(self, decision: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 4: ACT - Execute selection.
        """
        print("\nSTAGE 4: ACT")
        print("-" * 80)
        
        target_count = strategy['target_count']
        labels = decision['clustering']['labels']
        distances = decision['clustering']['distances']
        n_clusters = decision['clustering']['n_clusters']
        
        # Calculate samples per cluster
        cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
        base_per_cluster = [int(size * strategy['target_percentage'] / 100) for size in cluster_sizes]
        
        # Uncertainty-based boosting
        cluster_uncertainties = []
        for i in range(n_clusters):
            mask = labels == i
            avg_distance = np.mean(distances[mask]) if np.any(mask) else 0
            cluster_uncertainties.append(avg_distance)
        
        max_uncertainty = max(cluster_uncertainties) if cluster_uncertainties else 1
        uncertainty_weights = [u / max_uncertainty for u in cluster_uncertainties]
        
        # Adjust samples
        adjusted_per_cluster = []
        for i in range(n_clusters):
            base = base_per_cluster[i]
            boost = int(base * uncertainty_weights[i] * 0.2)
            adjusted_per_cluster.append(base + boost)
        
        # Normalize to target
        total_adjusted = sum(adjusted_per_cluster)
        final_per_cluster = [int(c * target_count / total_adjusted) for c in adjusted_per_cluster]
        
        diff = target_count - sum(final_per_cluster)
        if diff > 0:
            final_per_cluster[0] += diff
        
        # Select samples (uncertainty-based)
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
                f"Cluster {i}: Selected {n_select}/{cluster_sizes[i]} ({n_select/cluster_sizes[i]*100:.1f}%)"
            )
            print(f"  {selection_details[-1]}")
        
        # Create result
        selected_df = self.current_data.iloc[selected_indices].copy()
        selected_df['cluster_id'] = labels[selected_indices]
        selected_df['distance_to_centroid'] = distances[selected_indices]
        
        # Generate action explanation with LLM
        act_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Explain the final ACTION and sample selection.

Selection Summary:
- Dataset: {len(self.current_data)} samples
- Target: {strategy['target_percentage']:.1f}% = {target_count} samples
- Selected: {len(selected_indices)} samples
- Method: Uncertainty-based sampling
- Clusters: {n_clusters} clusters, all represented

Selection Details:
{chr(10).join(selection_details)}

Provide 2-3 sentences explaining:
1. Why this selection is representative
2. How uncertainty-based sampling improves training
3. Expected benefits

Use plain text only, no special symbols.""")
        ])
        
        chain = act_prompt | self.llm
        action_reasoning = chain.invoke({})
        
        # Extract text from response
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
            'selection_details': selection_details
        }
        
        return result
    
    def run_selection(self, user_query: str, csv_path: str) -> Dict[str, Any]:
        """
        Main entry point - orchestrate entire selection workflow.
        
        Args:
            user_query: Natural language query
            csv_path: Path to CSV file
            
        Returns:
            Complete results with all stages
        """
        print("\n" + "=" * 80)
        print("DATA SELECTION AGENT - LangChain Powered")
        print("=" * 80)
        
        # Add user query to history
        self.add_message('user', user_query)
        
        # Parse query
        print("\nParsing your request...")
        params = self.parse_user_query(user_query)
        
        # Run OBSERVE stage
        observation = self.observe(csv_path)
        
        # Run THINK stage
        strategy = self.think(observation, params['selection_percentage'])
        
        # Run DECIDE stage
        decision = self.decide(strategy)
        
        # Run ACT stage
        result = self.act(decision, strategy)
        
        print("\n" + "=" * 80)
        print("SELECTION COMPLETE")
        print("=" * 80)
        print(f"Selected {result['n_selected']} samples ({result['n_selected']/len(self.current_data)*100:.1f}%)")
        
        return {
            'observation': observation,
            'strategy': strategy,
            'decision': decision,
            'result': result,
            'reasoning_log': self.reasoning_log,
            'conversation_history': self.conversation_history,
            'parsed_params': params
        }
    
    def _extract_cluster_count(self, text: str) -> int:
        """Extract cluster count from LLM response."""
        try:
            # Look for patterns like "8 clusters" or "k=8"
            matches = re.findall(r'(\d+)\s*cluster', text, re.IGNORECASE)
            if matches:
                return int(matches[0])
            
            # Also try k= pattern
            matches = re.findall(r'k\s*=\s*(\d+)', text)
            if matches:
                return int(matches[0])
        except:
            pass
        
        return 8  # Default
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history
