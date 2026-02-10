"""
Real Intelligence Engine for Data-Driven Analysis
Replaces fake random thoughts and actions with genuine statistical analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression
import scipy.stats as stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class DataCharacteristicsAnalyzer:
    """Performs deep statistical analysis of dataset characteristics"""

    def __init__(self):
        self.analysis_cache = {}

    def analyze_dataset(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive statistical analysis of dataset"""
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return {'error': 'No numeric columns found'}

        characteristics = {
            'basic_stats': self._compute_basic_statistics(numeric_data),
            'correlation_analysis': self._analyze_correlations(numeric_data),
            'distribution_analysis': self._analyze_distributions(numeric_data),
            'outlier_analysis': self._analyze_outliers(numeric_data),
            'clustering_potential': self._assess_clustering_potential(numeric_data),
            'feature_importance': self._analyze_feature_importance(numeric_data),
            'dimensionality_assessment': self._assess_dimensionality(numeric_data)
        }

        return characteristics

    def _compute_basic_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute fundamental statistical properties"""
        return {
            'n_samples': len(data),
            'n_features': len(data.columns),
            'missing_ratio': data.isnull().sum().sum() / (len(data) * len(data.columns)),
            'mean_values': data.mean().to_dict(),
            'std_values': data.std().to_dict(),
            'skewness': data.skew().to_dict(),
            'kurtosis': data.kurtosis().to_dict(),
            'variance_ratios': (data.std() / data.mean()).to_dict()
        }

    def _analyze_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature correlations and relationships"""
        corr_matrix = data.corr()

        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value > 0.7:
                    strong_correlations.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })

        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'max_correlation': abs(corr_matrix).max().max(),
            'mean_correlation': abs(corr_matrix).mean().mean(),
            'strong_correlations': strong_correlations,
            'highly_correlated_features': len(strong_correlations)
        }

    def _analyze_distributions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze statistical distributions of features"""
        distribution_info = {}

        for col in data.columns:
            values = data[col].dropna()
            if len(values) < 3:
                continue

            # Test for normality
            _, normality_p = stats.normaltest(values)

            # Detect multimodality using dip test approximation
            hist, _ = np.histogram(values, bins=20)
            peaks = len([i for i in range(1, len(hist)-1)
                        if hist[i] > hist[i-1] and hist[i] > hist[i+1]])

            distribution_info[col] = {
                'is_normal': normality_p > 0.05,
                'normality_p_value': normality_p,
                'is_multimodal': peaks > 1,
                'num_peaks': peaks,
                'range': values.max() - values.min(),
                'iqr': values.quantile(0.75) - values.quantile(0.25)
            }

        return distribution_info

    def _analyze_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive outlier analysis"""
        X = StandardScaler().fit_transform(data)

        # Isolation Forest for outlier detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(X)
        outlier_ratio = (outlier_labels == -1).sum() / len(outlier_labels)

        # IQR method for each feature
        feature_outlier_counts = {}
        for col in data.columns:
            values = data[col]
            Q1, Q3 = values.quantile(0.25), values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((values < lower_bound) | (values > upper_bound)).sum()
            feature_outlier_counts[col] = outliers

        return {
            'global_outlier_ratio': outlier_ratio,
            'feature_outlier_counts': feature_outlier_counts,
            'total_outlier_features': sum(1 for count in feature_outlier_counts.values() if count > 0),
            'outlier_indices': np.where(outlier_labels == -1)[0].tolist()
        }

    def _assess_clustering_potential(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess how suitable data is for clustering"""
        X = StandardScaler().fit_transform(data)

        # Hopkins statistic for clustering tendency
        def hopkins_statistic(X, n_samples=min(200, len(X)//10)):
            if len(X) < 10:
                return 0.5

            sample_indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[sample_indices]

            # Random points in data space
            random_points = np.random.uniform(X.min(), X.max(), (n_samples, X.shape[1]))

            # Distances to nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=2).fit(X)
            u_distances = nbrs.kneighbors(random_points)[0][:, 1]
            w_distances = nbrs.kneighbors(X_sample)[0][:, 1]

            H = u_distances.sum() / (u_distances.sum() + w_distances.sum())
            return H

        hopkins_stat = hopkins_statistic(X)

        # Estimate optimal cluster count using elbow method
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(11, len(data)//10))

        for k in k_range:
            if k >= len(data):
                break
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(X, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(-1)

        # Find optimal k
        optimal_k = 2
        if silhouette_scores:
            optimal_k = k_range[np.argmax(silhouette_scores)]

        return {
            'hopkins_statistic': hopkins_stat,
            'is_clusterable': hopkins_stat > 0.6,
            'optimal_cluster_count': optimal_k,
            'max_silhouette_score': max(silhouette_scores) if silhouette_scores else 0,
            'clustering_tendency': 'high' if hopkins_stat > 0.7 else 'medium' if hopkins_stat > 0.5 else 'low'
        }

    def _analyze_feature_importance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature importance and discriminative power"""
        if len(data.columns) < 2:
            return {'feature_rankings': [], 'top_features': []}

        X = StandardScaler().fit_transform(data)

        # Use variance as a simple importance measure
        feature_variances = X.var(axis=0)
        feature_importance = dict(zip(data.columns, feature_variances))

        # Rank features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        return {
            'feature_importance_scores': feature_importance,
            'feature_rankings': sorted_features,
            'top_features': [feat[0] for feat in sorted_features[:5]],
            'low_variance_features': [feat[0] for feat in sorted_features if feat[1] < 0.1]
        }

    def _assess_dimensionality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess dimensionality and need for reduction"""
        X = StandardScaler().fit_transform(data)

        # PCA analysis
        pca = PCA()
        pca.fit(X)

        # Find components explaining 95% variance
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        components_95 = np.argmax(cumsum_variance >= 0.95) + 1
        components_90 = np.argmax(cumsum_variance >= 0.90) + 1

        return {
            'original_dimensions': X.shape[1],
            'effective_dimensions_90': components_90,
            'effective_dimensions_95': components_95,
            'dimension_reduction_potential': X.shape[1] - components_95,
            'first_component_variance': pca.explained_variance_ratio_[0],
            'needs_dimensionality_reduction': components_95 < X.shape[1] * 0.8
        }


class IntelligentReasoningEngine:
    """Generates data-driven insights and reasoning based on statistical evidence"""

    def __init__(self):
        self.reasoning_patterns = {}

    def generate_evidence_based_insight(self, characteristics: Dict[str, Any]) -> str:
        """Generate reasoning based on actual data characteristics"""
        insights = []

        # Analyze basic statistics
        basic_stats = characteristics.get('basic_stats', {})
        if basic_stats:
            insights.append(f"Dataset contains {basic_stats['n_samples']} samples with {basic_stats['n_features']} features")

            if basic_stats['missing_ratio'] > 0.1:
                insights.append(f"High missing data ratio ({basic_stats['missing_ratio']:.2f}) requires imputation strategy")

        # Analyze correlations
        corr_analysis = characteristics.get('correlation_analysis', {})
        if corr_analysis.get('highly_correlated_features', 0) > 0:
            strong_corr = corr_analysis['strong_correlations'][0] if corr_analysis['strong_correlations'] else None
            if strong_corr:
                insights.append(f"Strong correlation detected between {strong_corr['feature1']} and {strong_corr['feature2']} (r={strong_corr['correlation']:.3f})")

        # Analyze outliers
        outlier_analysis = characteristics.get('outlier_analysis', {})
        if outlier_analysis.get('global_outlier_ratio', 0) > 0.15:
            insights.append(f"High outlier ratio ({outlier_analysis['global_outlier_ratio']:.2f}) detected - robust methods recommended")

        # Analyze clustering potential
        clustering = characteristics.get('clustering_potential', {})
        if clustering.get('is_clusterable'):
            insights.append(f"Data shows clustering tendency (Hopkins={clustering['hopkins_statistic']:.3f}) with optimal k={clustering['optimal_cluster_count']}")
        else:
            insights.append(f"Low clustering tendency (Hopkins={clustering.get('hopkins_statistic', 0):.3f}) - uniform sampling may be more appropriate")

        # Analyze dimensionality
        dim_analysis = characteristics.get('dimensionality_assessment', {})
        if dim_analysis.get('needs_dimensionality_reduction'):
            insights.append(f"Dimensionality reduction recommended: {dim_analysis['original_dimensions']} to {dim_analysis['effective_dimensions_95']} dimensions")

        return " | ".join(insights) if insights else "Basic statistical analysis completed"

    def determine_optimal_strategy(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Determine sampling strategy based on data characteristics"""
        strategy = {
            'primary_method': 'random',
            'reasoning': 'Default fallback strategy',
            'parameters': {},
            'confidence': 0.5
        }

        clustering = characteristics.get('clustering_potential', {})
        outlier_analysis = characteristics.get('outlier_analysis', {})

        # Decision logic based on data characteristics
        if clustering.get('is_clusterable', False) and clustering.get('hopkins_statistic', 0) > 0.7:
            strategy.update({
                'primary_method': 'stratified_clustering',
                'reasoning': f"High clustering tendency (Hopkins={clustering['hopkins_statistic']:.3f}) indicates natural data structure",
                'parameters': {'n_clusters': clustering.get('optimal_cluster_count', 3)},
                'confidence': 0.85
            })
        elif outlier_analysis.get('global_outlier_ratio', 0) > 0.2:
            strategy.update({
                'primary_method': 'outlier_preserving',
                'reasoning': f"High outlier ratio ({outlier_analysis['global_outlier_ratio']:.2f}) requires preserving edge cases",
                'parameters': {'outlier_ratio': outlier_analysis['global_outlier_ratio']},
                'confidence': 0.8
            })
        elif clustering.get('hopkins_statistic', 0.5) < 0.3:
            strategy.update({
                'primary_method': 'uniform_random',
                'reasoning': f"Low clustering tendency (Hopkins={clustering.get('hopkins_statistic', 0):.3f}) suggests uniform distribution",
                'parameters': {},
                'confidence': 0.75
            })

        return strategy


class IntelligentParameterOptimizer:
    """Optimizes algorithm parameters based on data characteristics"""

    def optimize_clustering_parameters(self, X: np.ndarray, algorithm: str) -> Dict[str, Any]:
        """Data-driven parameter optimization for clustering algorithms"""
        if algorithm.lower() == 'dbscan':
            return self._optimize_dbscan_params(X)
        elif algorithm.lower() == 'kmeans':
            return self._optimize_kmeans_params(X)
        else:
            return {}

    def _optimize_dbscan_params(self, X: np.ndarray) -> Dict[str, Any]:
        """Optimize DBSCAN parameters based on data characteristics"""
        # Use k-nearest neighbors to find optimal eps
        k = min(4, len(X) // 10)
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(X)
        distances, _ = nbrs.kneighbors(X)

        # Sort distances to k-th nearest neighbor
        kth_distances = np.sort(distances[:, k-1])

        # Find elbow point (use 80th percentile as heuristic)
        eps = np.percentile(kth_distances, 80)

        # Set min_samples based on dimensionality and data size
        min_samples = max(2, int(np.log(len(X))))

        return {
            'eps': eps,
            'min_samples': min_samples,
            'reasoning': f"eps={eps:.3f} from 80th percentile of {k}-NN distances, min_samples={min_samples} from log(n_samples)"
        }

    def _optimize_kmeans_params(self, X: np.ndarray) -> Dict[str, Any]:
        """Optimize KMeans parameters using gap statistic"""
        max_k = min(10, len(X) // 10)
        if max_k < 2:
            return {'n_clusters': 2, 'reasoning': 'Insufficient data for parameter optimization'}

        # Simple elbow method for k optimization
        inertias = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        # Find elbow using rate of change
        if len(inertias) > 2:
            diff1 = np.diff(inertias)
            diff2 = np.diff(diff1)
            elbow_idx = np.argmax(diff2) + 2 if len(diff2) > 0 else 0
            optimal_k = k_range[min(elbow_idx, len(k_range) - 1)]
        else:
            optimal_k = k_range[0]

        return {
            'n_clusters': optimal_k,
            'reasoning': f"Optimal k={optimal_k} selected using elbow method from inertia analysis"
        }


class ConcreteInsightsMemory:
    """Stores and retrieves concrete, actionable insights from analysis iterations"""

    def __init__(self):
        self.insights_database = {}
        self.iteration_count = 0

    def store_iteration_insights(self, characteristics: Dict[str, Any], strategy: Dict[str, Any],
                                results: Dict[str, Any]) -> int:
        """Store concrete insights from an iteration"""
        self.iteration_count += 1

        insights = {
            'iteration': self.iteration_count,
            'timestamp': pd.Timestamp.now(),
            'data_characteristics': {
                'n_samples': characteristics.get('basic_stats', {}).get('n_samples', 0),
                'n_features': characteristics.get('basic_stats', {}).get('n_features', 0),
                'hopkins_statistic': characteristics.get('clustering_potential', {}).get('hopkins_statistic', 0),
                'outlier_ratio': characteristics.get('outlier_analysis', {}).get('global_outlier_ratio', 0),
                'optimal_clusters': characteristics.get('clustering_potential', {}).get('optimal_cluster_count', 0)
            },
            'strategy_used': strategy,
            'performance_metrics': results,
            'key_discoveries': self._extract_key_discoveries(characteristics, strategy, results)
        }

        self.insights_database[self.iteration_count] = insights
        return self.iteration_count

    def _extract_key_discoveries(self, characteristics: Dict[str, Any], strategy: Dict[str, Any],
                                results: Dict[str, Any]) -> List[str]:
        """Extract concrete, actionable discoveries"""
        discoveries = []

        # Algorithm performance insights
        if 'algorithm_performance' in results:
            best_algorithm = max(results['algorithm_performance'].items(), key=lambda x: x[1])
            discoveries.append(f"Best performing algorithm: {best_algorithm[0]} (score: {best_algorithm[1]:.3f})")

        # Feature importance insights
        feature_analysis = characteristics.get('feature_importance', {})
        if feature_analysis.get('top_features'):
            top_3 = feature_analysis['top_features'][:3]
            discoveries.append(f"Most discriminative features: {', '.join(top_3)}")

        # Data structure insights
        clustering = characteristics.get('clustering_potential', {})
        if clustering.get('is_clusterable'):
            discoveries.append(f"Natural clustering structure detected: {clustering['optimal_cluster_count']} clusters optimal")

        return discoveries

    def get_learning_guidance(self) -> Dict[str, Any]:
        """Provide guidance for next iteration based on stored insights"""
        if not self.insights_database:
            return {'guidance': 'Perform initial data analysis', 'confidence': 0.5}

        latest = self.insights_database[self.iteration_count]
        guidance = {'recommendations': [], 'avoid': [], 'confidence': 0.7}

        # Analysis of past performance
        if len(self.insights_database) > 1:
            # Find consistently successful approaches
            successful_strategies = []
            for insights in self.insights_database.values():
                if insights.get('performance_metrics', {}).get('quality_score', 0) > 0.7:
                    successful_strategies.append(insights['strategy_used']['primary_method'])

            if successful_strategies:
                most_common = max(set(successful_strategies), key=successful_strategies.count)
                guidance['recommendations'].append(f"Continue using {most_common} strategy (historically successful)")

        return guidance

    def has_converged(self, tolerance: float = 0.05, lookback: int = 3) -> bool:
        """Check if analysis has converged based on stable performance"""
        if len(self.insights_database) < lookback:
            return False

        recent_scores = []
        for i in range(max(1, self.iteration_count - lookback + 1), self.iteration_count + 1):
            if i in self.insights_database:
                score = self.insights_database[i].get('performance_metrics', {}).get('quality_score', 0)
                recent_scores.append(score)

        if len(recent_scores) < 2:
            return False

        # Check if variance in recent scores is below tolerance
        score_variance = np.var(recent_scores)
        return score_variance < tolerance ** 2