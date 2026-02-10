"""
Intelligent Sampling Engine
Replaces fake uncertainty sampling with real evidence-based sample selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class EvidenceBasedSampler:
    """Implements sampling strategies based on concrete data analysis"""

    def __init__(self):
        self.scaler = None
        self.last_analysis = None

    def select_samples(self, data: pd.DataFrame, target_percentage: float,
                      strategy: Dict[str, Any], characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sample selection based on data-driven strategy"""

        # Prepare numeric data
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return {'error': 'No numeric data available for sampling'}

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(numeric_data)

        target_count = max(1, int(len(data) * target_percentage / 100))

        method = strategy.get('primary_method', 'random')

        # Execute appropriate sampling method
        if method == 'stratified_clustering':
            return self._stratified_clustering_sampling(X_scaled, target_count, strategy, data)
        elif method == 'outlier_preserving':
            return self._outlier_preserving_sampling(X_scaled, target_count, strategy, data)
        elif method == 'uniform_random':
            return self._uniform_random_sampling(X_scaled, target_count, data)
        else:
            return self._fallback_sampling(X_scaled, target_count, data)

    def _stratified_clustering_sampling(self, X_scaled: np.ndarray, target_count: int,
                                       strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Sample proportionally from natural clusters in the data"""

        n_clusters = strategy.get('parameters', {}).get('n_clusters', 3)

        # Apply clustering
        clustering_results = self._apply_optimal_clustering(X_scaled, n_clusters)
        labels = clustering_results['labels']

        # Calculate samples per cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_valid_samples = len(labels[labels != -1])  # Exclude noise points

        selected_indices = []

        for label, count in zip(unique_labels, counts):
            if label == -1:  # Skip noise points for now
                continue

            cluster_mask = labels == label
            cluster_indices = np.where(cluster_mask)[0]

            # Proportional allocation
            cluster_target = max(1, int(target_count * count / total_valid_samples))
            cluster_target = min(cluster_target, len(cluster_indices))

            # Select representative samples from cluster center and edges
            cluster_data = X_scaled[cluster_mask]
            cluster_center = np.mean(cluster_data, axis=0)

            # Distance from center
            distances = np.linalg.norm(cluster_data - cluster_center, axis=1)

            # Select mix of center and edge samples
            center_count = max(1, cluster_target // 2)
            edge_count = cluster_target - center_count

            # Center samples (closest to centroid)
            center_indices = np.argsort(distances)[:center_count]

            # Edge samples (farthest from center, but not outliers)
            if edge_count > 0:
                sorted_distances = np.argsort(distances)
                # Take from 60-90th percentile to avoid extreme outliers
                start_idx = int(len(sorted_distances) * 0.6)
                end_idx = int(len(sorted_distances) * 0.9)
                edge_candidates = sorted_distances[start_idx:end_idx]

                if len(edge_candidates) >= edge_count:
                    edge_indices = np.random.choice(edge_candidates, edge_count, replace=False)
                else:
                    edge_indices = edge_candidates

            else:
                edge_indices = []

            # Convert back to original indices
            selected_cluster_indices = np.concatenate([
                cluster_indices[center_indices],
                cluster_indices[edge_indices] if len(edge_indices) > 0 else []
            ])

            selected_indices.extend(selected_cluster_indices.tolist())

        # Handle noise points if we need more samples
        if len(selected_indices) < target_count and -1 in unique_labels:
            noise_mask = labels == -1
            noise_indices = np.where(noise_mask)[0]
            needed = target_count - len(selected_indices)
            noise_sample = np.random.choice(noise_indices, min(needed, len(noise_indices)), replace=False)
            selected_indices.extend(noise_sample.tolist())

        # Adjust to exact target count
        if len(selected_indices) > target_count:
            selected_indices = np.random.choice(selected_indices, target_count, replace=False).tolist()
        elif len(selected_indices) < target_count:
            remaining = [i for i in range(len(X_scaled)) if i not in selected_indices]
            needed = target_count - len(selected_indices)
            if remaining:
                additional = np.random.choice(remaining, min(needed, len(remaining)), replace=False)
                selected_indices.extend(additional.tolist())

        return {
            'selected_indices': selected_indices[:target_count],
            'method_used': 'stratified_clustering',
            'clustering_info': clustering_results,
            'reasoning': f"Selected {len(selected_indices)} samples proportionally from {n_clusters} natural clusters"
        }

    def _outlier_preserving_sampling(self, X_scaled: np.ndarray, target_count: int,
                                    strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Preserve outliers while sampling representative normal cases"""

        outlier_ratio = strategy.get('parameters', {}).get('outlier_ratio', 0.2)

        # Detect outliers using Isolation Forest
        iso_forest = IsolationForest(contamination=outlier_ratio, random_state=42)
        outlier_labels = iso_forest.fit_predict(X_scaled)

        outlier_indices = np.where(outlier_labels == -1)[0]
        normal_indices = np.where(outlier_labels == 1)[0]

        # Allocate samples
        outlier_target = min(len(outlier_indices), max(1, int(target_count * 0.3)))  # At least 30% outliers
        normal_target = target_count - outlier_target

        selected_indices = []

        # Select all outliers if possible, otherwise sample them
        if outlier_target >= len(outlier_indices):
            selected_indices.extend(outlier_indices.tolist())
        else:
            selected_outliers = np.random.choice(outlier_indices, outlier_target, replace=False)
            selected_indices.extend(selected_outliers.tolist())

        # Sample normal cases using clustering if we have enough
        if normal_target > 0 and len(normal_indices) > 0:
            if len(normal_indices) >= normal_target * 3:  # Enough for clustering
                normal_data = X_scaled[normal_indices]
                n_clusters = min(5, len(normal_indices) // 10, normal_target)

                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(normal_data)

                    samples_per_cluster = normal_target // n_clusters
                    remainder = normal_target % n_clusters

                    for cluster_id in range(n_clusters):
                        cluster_mask = cluster_labels == cluster_id
                        cluster_normal_indices = normal_indices[cluster_mask]

                        cluster_target = samples_per_cluster + (1 if cluster_id < remainder else 0)
                        cluster_target = min(cluster_target, len(cluster_normal_indices))

                        if cluster_target > 0:
                            selected = np.random.choice(cluster_normal_indices, cluster_target, replace=False)
                            selected_indices.extend(selected.tolist())
                else:
                    # Too few for clustering, random sample
                    selected_normal = np.random.choice(normal_indices, min(normal_target, len(normal_indices)), replace=False)
                    selected_indices.extend(selected_normal.tolist())
            else:
                # Random sample from normal cases
                selected_normal = np.random.choice(normal_indices, min(normal_target, len(normal_indices)), replace=False)
                selected_indices.extend(selected_normal.tolist())

        return {
            'selected_indices': selected_indices[:target_count],
            'method_used': 'outlier_preserving',
            'outlier_info': {
                'total_outliers': len(outlier_indices),
                'selected_outliers': sum(1 for i in selected_indices if i in outlier_indices),
                'outlier_ratio_in_sample': sum(1 for i in selected_indices if i in outlier_indices) / len(selected_indices)
            },
            'reasoning': f"Preserved {sum(1 for i in selected_indices if i in outlier_indices)} outliers out of {len(outlier_indices)} detected"
        }

    def _uniform_random_sampling(self, X_scaled: np.ndarray, target_count: int,
                                data: pd.DataFrame) -> Dict[str, Any]:
        """Simple random sampling when data lacks clear structure"""

        selected_indices = np.random.choice(len(X_scaled), target_count, replace=False).tolist()

        return {
            'selected_indices': selected_indices,
            'method_used': 'uniform_random',
            'reasoning': f"Random sampling used due to lack of clear data structure"
        }

    def _fallback_sampling(self, X_scaled: np.ndarray, target_count: int,
                          data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback sampling method"""

        selected_indices = np.random.choice(len(X_scaled), target_count, replace=False).tolist()

        return {
            'selected_indices': selected_indices,
            'method_used': 'fallback_random',
            'reasoning': "Fallback random sampling applied"
        }

    def _apply_optimal_clustering(self, X_scaled: np.ndarray, n_clusters: int) -> Dict[str, Any]:
        """Apply clustering and return best result based on metrics"""

        clustering_results = []

        # Try KMeans
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(X_scaled)

            if len(np.unique(kmeans_labels)) > 1:
                kmeans_score = silhouette_score(X_scaled, kmeans_labels)
                clustering_results.append({
                    'algorithm': 'kmeans',
                    'labels': kmeans_labels,
                    'score': kmeans_score,
                    'n_clusters_found': len(np.unique(kmeans_labels))
                })
        except:
            pass

        # Try DBSCAN with optimized parameters
        try:
            # Optimize eps using k-nearest neighbors
            k = min(4, len(X_scaled) // 10)
            if k >= 2:
                nbrs = NearestNeighbors(n_neighbors=k)
                nbrs.fit(X_scaled)
                distances, _ = nbrs.kneighbors(X_scaled)
                kth_distances = np.sort(distances[:, k-1])
                eps = np.percentile(kth_distances, 80)
                min_samples = max(2, int(np.log(len(X_scaled))))

                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                dbscan_labels = dbscan.fit_predict(X_scaled)

                # Only consider if it found reasonable clusters
                unique_labels = np.unique(dbscan_labels)
                n_clusters_found = len(unique_labels[unique_labels != -1])

                if n_clusters_found >= 2 and n_clusters_found <= n_clusters * 2:
                    # Calculate score only on non-noise points
                    non_noise_mask = dbscan_labels != -1
                    if non_noise_mask.sum() > 10:  # Need enough non-noise points
                        dbscan_score = silhouette_score(X_scaled[non_noise_mask], dbscan_labels[non_noise_mask])
                        clustering_results.append({
                            'algorithm': 'dbscan',
                            'labels': dbscan_labels,
                            'score': dbscan_score,
                            'n_clusters_found': n_clusters_found,
                            'noise_ratio': (dbscan_labels == -1).mean()
                        })
        except:
            pass

        # Try Gaussian Mixture if we have enough samples
        if len(X_scaled) >= n_clusters * 10:
            try:
                gmm = GaussianMixture(n_components=n_clusters, random_state=42)
                gmm_labels = gmm.fit_predict(X_scaled)

                if len(np.unique(gmm_labels)) > 1:
                    gmm_score = silhouette_score(X_scaled, gmm_labels)
                    clustering_results.append({
                        'algorithm': 'gaussian_mixture',
                        'labels': gmm_labels,
                        'score': gmm_score,
                        'n_clusters_found': len(np.unique(gmm_labels))
                    })
            except:
                pass

        # Select best clustering result
        if clustering_results:
            best_result = max(clustering_results, key=lambda x: x['score'])
            return best_result
        else:
            # Fallback to simple KMeans
            kmeans = KMeans(n_clusters=min(n_clusters, len(X_scaled)), random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            return {
                'algorithm': 'kmeans_fallback',
                'labels': labels,
                'score': 0.0,
                'n_clusters_found': len(np.unique(labels))
            }


class IntelligentAlgorithmSelector:
    """Selects optimal algorithms based on data characteristics, not text parsing"""

    def __init__(self):
        self.algorithm_performance_history = {}

    def select_optimal_algorithm(self, characteristics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select algorithms based on data characteristics"""

        algorithms = []

        # Analyze data characteristics
        n_samples = characteristics.get('basic_stats', {}).get('n_samples', 0)
        n_features = characteristics.get('basic_stats', {}).get('n_features', 0)
        outlier_ratio = characteristics.get('outlier_analysis', {}).get('global_outlier_ratio', 0)
        hopkins_stat = characteristics.get('clustering_potential', {}).get('hopkins_statistic', 0.5)
        max_correlation = characteristics.get('correlation_analysis', {}).get('max_correlation', 0)

        # KMeans selection logic
        if hopkins_stat > 0.6 and outlier_ratio < 0.15 and n_samples >= 100:
            algorithms.append({
                'name': 'KMeans',
                'confidence': min(0.9, hopkins_stat + 0.2),
                'reasoning': f"High clustering tendency (Hopkins={hopkins_stat:.3f}) with low outlier ratio ({outlier_ratio:.3f})",
                'parameters': self._get_kmeans_params(characteristics)
            })

        # DBSCAN selection logic
        if outlier_ratio > 0.1 or hopkins_stat > 0.7:
            confidence = 0.7 if outlier_ratio > 0.2 else 0.6
            algorithms.append({
                'name': 'DBSCAN',
                'confidence': confidence,
                'reasoning': f"High outlier ratio ({outlier_ratio:.3f}) or strong clustering tendency",
                'parameters': self._get_dbscan_params(characteristics)
            })

        # Gaussian Mixture selection logic
        if (max_correlation > 0.6 and n_features > 3 and n_samples >= n_features * 20):
            algorithms.append({
                'name': 'GaussianMixture',
                'confidence': 0.75,
                'reasoning': f"High feature correlation ({max_correlation:.3f}) suggests soft clustering appropriate",
                'parameters': self._get_gmm_params(characteristics)
            })

        # Agglomerative Clustering for small datasets
        if n_samples < 1000 and hopkins_stat > 0.5:
            algorithms.append({
                'name': 'AgglomerativeClustering',
                'confidence': 0.65,
                'reasoning': f"Small dataset size ({n_samples}) suitable for hierarchical clustering",
                'parameters': self._get_agglomerative_params(characteristics)
            })

        # Sort by confidence
        algorithms.sort(key=lambda x: x['confidence'], reverse=True)

        # Ensure we have at least one algorithm
        if not algorithms:
            algorithms.append({
                'name': 'KMeans',
                'confidence': 0.5,
                'reasoning': "Default algorithm when data characteristics are unclear",
                'parameters': {'n_clusters': 3}
            })

        return algorithms[:3]  # Return top 3 algorithms

    def _get_kmeans_params(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimal KMeans parameters based on data characteristics"""
        optimal_k = characteristics.get('clustering_potential', {}).get('optimal_cluster_count', 3)
        n_samples = characteristics.get('basic_stats', {}).get('n_samples', 0)

        return {
            'n_clusters': max(2, min(optimal_k, n_samples // 20)),
            'n_init': 20 if n_samples < 10000 else 10,
            'max_iter': 300,
            'random_state': 42
        }

    def _get_dbscan_params(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimal DBSCAN parameters based on data characteristics"""
        n_samples = characteristics.get('basic_stats', {}).get('n_samples', 0)
        n_features = characteristics.get('basic_stats', {}).get('n_features', 1)

        return {
            'min_samples': max(2, min(int(np.log(n_samples)), n_features + 1)),
            'metric': 'euclidean',
            'algorithm': 'auto'
        }

    def _get_gmm_params(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimal Gaussian Mixture parameters based on data characteristics"""
        optimal_k = characteristics.get('clustering_potential', {}).get('optimal_cluster_count', 3)

        return {
            'n_components': optimal_k,
            'covariance_type': 'full',
            'max_iter': 100,
            'random_state': 42
        }

    def _get_agglomerative_params(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimal Agglomerative parameters based on data characteristics"""
        optimal_k = characteristics.get('clustering_potential', {}).get('optimal_cluster_count', 3)

        return {
            'n_clusters': optimal_k,
            'linkage': 'ward',
            'metric': 'euclidean'
        }

    def update_algorithm_performance(self, algorithm_name: str, performance_score: float,
                                   data_characteristics: Dict[str, Any]) -> None:
        """Store algorithm performance for future reference"""
        if algorithm_name not in self.algorithm_performance_history:
            self.algorithm_performance_history[algorithm_name] = []

        self.algorithm_performance_history[algorithm_name].append({
            'score': performance_score,
            'characteristics': data_characteristics,
            'timestamp': pd.Timestamp.now()
        })

    def get_algorithm_recommendation_confidence(self, algorithm_name: str,
                                              current_characteristics: Dict[str, Any]) -> float:
        """Get confidence score for algorithm based on historical performance"""
        if algorithm_name not in self.algorithm_performance_history:
            return 0.5  # Default confidence

        history = self.algorithm_performance_history[algorithm_name]

        # Simple average of past performance
        avg_performance = np.mean([entry['score'] for entry in history])

        # Adjust based on recency (more recent results weighted higher)
        if len(history) > 1:
            recent_weight = 0.7
            older_weight = 0.3
            recent_score = history[-1]['score']
            weighted_score = recent_score * recent_weight + avg_performance * older_weight
            return min(0.95, weighted_score)

        return min(0.8, avg_performance)