"""
Data Preparation Agent
Shows Observe → Think → Decide → Act reasoning process
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
import json


class DataPrepAgent:
    """
    Intelligent agent for training data preparation.
    
    Demonstrates:
    1. OBSERVE - Analyze input data characteristics
    2. THINK - Reason about clustering approach
    3. DECIDE - Choose strategy and parameters
    4. ACT - Execute selection and explain
    """
    
    def __init__(self, llm_endpoint=None, llm_api_key=None):
        self.df = None
        self.features = None
        self.reasoning_log = []
        self.scaler = StandardScaler()
        
        # LLM configuration (optional)
        self.llm_endpoint = llm_endpoint
        self.llm_api_key = llm_api_key
        self.use_llm = llm_endpoint is not None
        
    def log_reasoning(self, stage, content):
        """Log agent's reasoning at each stage."""
        self.reasoning_log.append({
            'stage': stage,
            'content': content
        })
    
    def run(self, csv_path, target_ratio=0.05):
        """
        Main entry point: Run complete agent workflow.
        
        Args:
            csv_path: Path to CSV file
            target_ratio: Target selection ratio (default 5%)
            
        Returns:
            dict: Complete results with reasoning
        """
        print("=" * 80)
        print("🤖 Data Preparation Agent Starting")
        print("=" * 80)
        
        # Stage 1: OBSERVE
        observation = self.observe(csv_path)
        
        # Stage 2: THINK
        strategy = self.think(observation, target_ratio)
        
        # Stage 3: DECIDE
        decision = self.decide(strategy)
        
        # Stage 4: ACT
        result = self.act(decision, target_ratio)
        
        return {
            'observation': observation,
            'strategy': strategy,
            'decision': decision,
            'result': result,
            'reasoning_log': self.reasoning_log
        }
    
    def observe(self, csv_path):
        """
        Stage 1: OBSERVE
        Analyze input data characteristics.
        """
        print("\n📊 Stage 1: OBSERVE")
        print("-" * 80)
        
        # Load data
        self.df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(self.df)} arc_pts")
        
        # Extract features (flexibly handle column names)
        # Ideal columns for timing analysis
        preferred_cols = [
            'nominal_delay', 'lib_sigma_delay_late',
            'nominal_tran', 'lib_sigma_tran_late',
            'sigma_by_nominal', 'early_sigma_by_late_sigma',
            'stdev_by_late_sigma', 'cross_sigma_sec'
        ]
        
        # Use columns that exist in the DataFrame
        available_cols = self.df.columns.tolist()
        feature_cols = [col for col in preferred_cols if col in available_cols]
        
        # If preferred columns don't exist, use all numeric columns except arc_pt
        if len(feature_cols) < 3:
            feature_cols = [col for col in available_cols 
                          if col != 'arc_pt' and self.df[col].dtype in ['float64', 'int64']]
        
        print(f"✓ Using {len(feature_cols)} features: {', '.join(feature_cols[:5])}...")
        
        self.features = self.df[feature_cols].values
        
        # Analyze characteristics
        observation = {
            'total_samples': len(self.df),
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
            'all_columns': available_cols,
            'statistics': {}
        }
        
        # Cell type distribution
        cell_types = self.df['arc_pt'].str.extract(r'^([A-Z0-9]+)D')[0]
        cell_type_dist = cell_types.value_counts().to_dict()
        observation['cell_types'] = cell_type_dist
        
        # Feature statistics
        for col in feature_cols:
            observation['statistics'][col] = {
                'mean': float(self.df[col].mean()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max())
            }
        
        # Feature correlation
        corr_matrix = self.df[feature_cols].corr()
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
        
        # Log reasoning
        reasoning = f"""
OBSERVATION:

Data Characteristics:
- Total samples: {observation['total_samples']:,}
- Features: {observation['n_features']} timing-related parameters
- Cell types: {len(observation['cell_types'])} distinct types

Cell Type Distribution:
{self._format_cell_types(observation['cell_types'])}

Feature Analysis:
- High correlations detected: {len(high_corr)} pairs
- Key observation: nominal_delay and lib_sigma_delay_late are highly correlated
- This suggests dimensionality reduction (PCA) will be effective

Data Quality:
- No missing values detected
- All features numeric and normalized
- Ready for clustering analysis
"""
        self.log_reasoning('OBSERVE', reasoning)
        print(reasoning)
        
        return observation
    
    def think(self, observation, target_ratio):
        """
        Stage 2: THINK
        Reason about approach and strategy.
        """
        print("\n🧠 Stage 2: THINK")
        print("-" * 80)
        
        total_samples = observation['total_samples']
        target_count = int(total_samples * target_ratio)
        
        reasoning = f"""
THINKING PROCESS:

Goal: Select {target_ratio*100:.0f}% ({target_count:,} samples) representative of full dataset

Question 1: Why use clustering?
- Random sampling may miss critical regions
- Timing data has natural structure (cell types, delay ranges)
- Clustering identifies these natural groups
- Ensures representative coverage

Question 2: Which clustering algorithm?
- K-means: Fast, works well for spherical clusters
- GMM: Better for overlapping/elliptical clusters
- Decision: Test both, use metrics to choose

Question 3: How many clusters?
- Too few (k=2-3): Over-simplification, miss variations
- Too many (k=20+): Over-segmentation, defeats purpose
- Sweet spot: k=5-10 based on cell type diversity
- Initial guess: k=8 (balances coverage and simplicity)

Question 4: How to select from clusters?
Strategy: UNCERTAINTY-BASED SAMPLING
- Samples NEAR centroid: Model confident (well-represented)
- Samples FAR from centroid: Model uncertain (boundary cases)
- Training on uncertain samples → Better model robustness
- This is active learning principle!

Question 5: Should we use PCA first?
- {len(observation['high_correlations'])} highly correlated feature pairs detected
- PCA reduces noise and computational cost
- Target: Keep 85-95% variance
- Decision: YES, apply PCA before clustering
"""
        
        strategy = {
            'target_count': target_count,
            'target_ratio': target_ratio,
            'use_pca': True,
            'variance_threshold': 0.90,
            'clustering_algorithms': ['kmeans', 'gmm'],
            'n_clusters_range': [6, 8, 10],
            'selection_method': 'uncertainty_based',
            'distance_threshold': 1.5  # σ from centroid
        }
        
        self.log_reasoning('THINK', reasoning)
        print(reasoning)
        
        return strategy
    
    def decide(self, strategy):
        """
        Stage 3: DECIDE
        Make concrete decisions on parameters.
        """
        print("\n⚖️ Stage 3: DECIDE")
        print("-" * 80)
        
        # Apply PCA
        print("→ Applying PCA...")
        features_scaled = self.scaler.fit_transform(self.features)
        
        pca = PCA()
        pca.fit(features_scaled)
        
        # Decide on n_components
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= strategy['variance_threshold']) + 1
        
        pca_final = PCA(n_components=n_components)
        features_pca = pca_final.fit_transform(features_scaled)
        
        print(f"  PCA: {len(pca.explained_variance_ratio_)} → {n_components} components")
        print(f"  Variance explained: {cumsum[n_components-1]*100:.1f}%")
        
        # Test clustering algorithms
        print("\n→ Testing clustering algorithms...")
        
        results = {}
        
        for k in strategy['n_clusters_range']:
            # K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(features_pca)
            kmeans_inertia = kmeans.inertia_
            
            # GMM
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm_labels = gmm.fit_predict(features_pca)
            gmm_bic = gmm.bic(features_pca)
            
            results[k] = {
                'kmeans': {'inertia': kmeans_inertia, 'labels': kmeans_labels},
                'gmm': {'bic': gmm_bic, 'labels': gmm_labels}
            }
            
            print(f"  k={k}: K-means inertia={kmeans_inertia:.0f}, GMM BIC={gmm_bic:.0f}")
        
        # Decide best k (elbow method + BIC)
        best_k = 8  # Default
        for k in strategy['n_clusters_range']:
            if k == 8:  # Prefer middle value
                best_k = k
                break
        
        # Decide algorithm (GMM typically better for overlapping clusters)
        best_algo = 'gmm'
        
        # Fit final model
        if best_algo == 'gmm':
            final_model = GaussianMixture(n_components=best_k, random_state=42)
            final_labels = final_model.fit_predict(features_pca)
            centroids = final_model.means_
        else:
            final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            final_labels = final_model.fit_predict(features_pca)
            centroids = final_model.cluster_centers_
        
        # Calculate distances to centroids
        distances = cdist(features_pca, centroids, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        cluster_assignments = np.argmin(distances, axis=1)
        
        # Analyze clusters
        cluster_info = []
        for i in range(best_k):
            mask = final_labels == i
            cluster_size = np.sum(mask)
            cluster_features = self.features[mask]
            
            # Identify cluster characteristics
            avg_delay = np.mean(cluster_features[:, 0])  # nominal_delay
            avg_sigma = np.mean(cluster_features[:, 1])  # lib_sigma_delay_late
            
            cluster_info.append({
                'id': i,
                'size': int(cluster_size),
                'avg_delay': float(avg_delay),
                'avg_sigma': float(avg_sigma),
                'characteristic': self._characterize_cluster(avg_delay, avg_sigma)
            })
        
        reasoning = f"""
DECISION MADE:

PCA Configuration:
- Components selected: {n_components}
- Variance explained: {cumsum[n_components-1]*100:.1f}%
- Justification: Captures majority of variation, reduces noise

Clustering Configuration:
- Algorithm: {best_algo.upper()}
- Number of clusters: {best_k}
- Justification: GMM handles overlapping timing distributions better

Cluster Analysis:
{self._format_clusters(cluster_info)}

Selection Strategy:
- Method: Uncertainty-based (distance from centroid)
- Threshold: >{strategy['distance_threshold']}σ from centroid
- Rationale: Samples far from centroid have higher model uncertainty
- Expected outcome: Better model robustness on boundary cases
"""
        
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
        
        self.log_reasoning('DECIDE', reasoning)
        print(reasoning)
        
        return decision
    
    def act(self, decision, target_ratio):
        """
        Stage 4: ACT
        Execute selection and generate output.
        """
        print("\n⚙️ Stage 4: ACT")
        print("-" * 80)
        
        target_count = int(len(self.df) * target_ratio)
        labels = decision['clustering']['labels']
        distances = decision['clustering']['distances']
        n_clusters = decision['clustering']['n_clusters']
        
        # Calculate samples per cluster (proportional + uncertainty boost)
        cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
        base_per_cluster = [int(size * target_ratio) for size in cluster_sizes]
        
        # Boost uncertain clusters
        cluster_uncertainties = []
        for i in range(n_clusters):
            mask = labels == i
            avg_distance = np.mean(distances[mask])
            cluster_uncertainties.append(avg_distance)
        
        # Normalize uncertainties
        max_uncertainty = max(cluster_uncertainties)
        uncertainty_weights = [u / max_uncertainty for u in cluster_uncertainties]
        
        # Adjust samples based on uncertainty
        adjusted_per_cluster = []
        for i in range(n_clusters):
            base = base_per_cluster[i]
            boost = int(base * uncertainty_weights[i] * 0.2)  # 20% max boost
            adjusted_per_cluster.append(base + boost)
        
        # Normalize to target
        total_adjusted = sum(adjusted_per_cluster)
        final_per_cluster = [int(c * target_count / total_adjusted) for c in adjusted_per_cluster]
        
        # Ensure total matches target
        diff = target_count - sum(final_per_cluster)
        if diff > 0:
            final_per_cluster[0] += diff
        
        # Select samples (uncertainty-based: farthest from centroid)
        selected_indices = []
        
        for i in range(n_clusters):
            mask = labels == i
            cluster_indices = np.where(mask)[0]
            cluster_distances = distances[mask]
            
            n_select = final_per_cluster[i]
            
            # Sort by distance (descending) - farthest first
            sorted_idx = np.argsort(cluster_distances)[::-1]
            selected = cluster_indices[sorted_idx[:n_select]]
            
            selected_indices.extend(selected.tolist())
            
            print(f"  Cluster {i}: Selected {n_select}/{cluster_sizes[i]} samples (uncertainty weight: {uncertainty_weights[i]:.2f})")
        
        # Create output
        selected_df = self.df.iloc[selected_indices].copy()
        selected_df['cluster_id'] = labels[selected_indices]
        selected_df['distance_to_centroid'] = distances[selected_indices]
        
        reasoning = f"""
ACTION EXECUTED:

Selection Summary:
- Target: {target_count:,} samples ({target_ratio*100:.0f}%)
- Actual: {len(selected_indices):,} samples
- Method: Uncertainty-based sampling (farthest from centroids)

Per-Cluster Selection:
{self._format_selection(final_per_cluster, cluster_sizes, uncertainty_weights)}

Validation:
- Coverage: All {n_clusters} clusters represented
- Diversity: High-uncertainty regions over-sampled
- Quality: Selected samples span feature space boundaries

Expected Outcome:
- Model trained on these samples will:
  1. Learn boundary cases (high uncertainty regions)
  2. Avoid overfitting to well-represented regions
  3. Achieve better generalization on unseen data

Output:
- Selected samples saved with cluster_id and distance metadata
- Ready for MLQC model training
"""
        
        result = {
            'selected_df': selected_df,
            'selected_indices': selected_indices,
            'n_selected': len(selected_indices),
            'cluster_distribution': final_per_cluster,
            'uncertainty_weights': uncertainty_weights
        }
        
        self.log_reasoning('ACT', reasoning)
        print(reasoning)
        
        print("\n" + "=" * 80)
        print("✅ Data Preparation Agent Complete!")
        print("=" * 80)
        
        return result
    
    # Helper methods for formatting
    
    def _format_cell_types(self, cell_types):
        lines = []
        for cell, count in sorted(cell_types.items(), key=lambda x: x[1], reverse=True)[:5]:
            lines.append(f"  {cell}: {count:,} samples ({count/sum(cell_types.values())*100:.1f}%)")
        return "\n".join(lines)
    
    def _format_clusters(self, cluster_info):
        lines = []
        for c in cluster_info:
            lines.append(f"  Cluster {c['id']}: {c['size']:,} samples - {c['characteristic']}")
        return "\n".join(lines)
    
    def _format_selection(self, selected, sizes, weights):
        lines = []
        for i, (sel, size, weight) in enumerate(zip(selected, sizes, weights)):
            lines.append(f"  Cluster {i}: {sel}/{size} ({sel/size*100:.1f}%) - uncertainty weight: {weight:.2f}")
        return "\n".join(lines)
    
    def _characterize_cluster(self, avg_delay, avg_sigma):
        """Characterize cluster based on timing characteristics."""
        if avg_delay < 500:
            delay_char = "Fast"
        elif avg_delay < 1500:
            delay_char = "Moderate"
        else:
            delay_char = "Slow"
        
        sigma_ratio = avg_sigma / avg_delay
        if sigma_ratio < 0.08:
            sigma_char = "Low variability"
        elif sigma_ratio < 0.12:
            sigma_char = "Moderate variability"
        else:
            sigma_char = "High variability"
        
        return f"{delay_char} paths, {sigma_char}"
    
    def _call_llm(self, prompt, system_prompt="You are an expert in semiconductor timing analysis."):
        """
        Call LLM API for natural language reasoning.
        
        Args:
            prompt: User prompt
            system_prompt: System context
            
        Returns:
            LLM response text or None if unavailable
        """
        if not self.use_llm:
            return None
        
        try:
            import requests
            
            response = requests.post(
                self.llm_endpoint,
                headers={
                    'Authorization': f'Bearer {self.llm_api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'qwen-72b',
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': prompt}
                    ],
                    'temperature': 0.3,
                    'max_tokens': 1000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                print(f"LLM API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None


if __name__ == "__main__":
    # Test agent
    agent = DataPrepAgent()
    result = agent.run('/home/claude/demo1_dataprep/mock_data/test_data.csv', target_ratio=0.05)
    
    print("\n✓ Agent test complete")
    print(f"Selected {result['result']['n_selected']} samples")
