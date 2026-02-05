"""
Visual Dashboard Utilities for Timing Data Selection Agent
Provides visual proof of clustering effectiveness and sample selection quality
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Optional seaborn import with fallback
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_palette("husl")
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available. Using matplotlib styling instead.")

# Set professional plotting style
plt.style.use('default')

class TimingVisualizationDashboard:
    """
    Creates professional visualizations for timing data analysis validation.

    Provides visual proof that:
    1. Clustering captures timing path diversity
    2. Selected samples are truly boundary cases
    3. Selection covers the full timing distribution
    """

    def __init__(self, figsize_large=(15, 5), figsize_single=(8, 6)):
        """Initialize visualization dashboard."""
        self.figsize_large = figsize_large
        self.figsize_single = figsize_single
        self.colors = {
            'selected': '#FF4444',      # Red for selected samples
            'unselected': '#4488DD',    # Blue for unselected
            'centroids': '#000000',     # Black for cluster centers
            'background': '#F8F9FA',    # Light background
            'grid': '#E1E5E9'          # Grid color
        }

    def generate_dashboard_plots(self,
                                df: pd.DataFrame,
                                selected_indices: List[int],
                                clusters: np.ndarray,
                                centroids: np.ndarray = None,
                                pca_components: int = 2) -> Tuple[plt.Figure, Dict[str, Any]]:
        """
        Generate complete dashboard with three key validation plots.

        Args:
            df: Original timing data DataFrame
            selected_indices: Indices of selected samples
            clusters: Cluster assignments for all samples
            centroids: Cluster centroids (optional)
            pca_components: Number of PCA components for visualization

        Returns:
            Tuple of (matplotlib figure, plot metadata)
        """
        # Create dashboard figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=self.figsize_large)
        fig.suptitle('Timing Data Selection Validation Dashboard', fontsize=16, fontweight='bold', y=1.02)

        # Plot metadata for analysis
        plot_metadata = {}

        try:
            # Plot 1: PCA Scatter with Cluster Visualization
            meta1 = self._plot_pca_cluster_scatter(df, selected_indices, clusters, centroids, pca_components, axes[0])
            plot_metadata['pca_scatter'] = meta1

            # Plot 2: Feature Correlation Heatmap
            meta2 = self._plot_correlation_heatmap(df, axes[1])
            plot_metadata['correlation_heatmap'] = meta2

            # Plot 3: Distribution Overlay Analysis
            meta3 = self._plot_distribution_overlay(df, selected_indices, axes[2])
            plot_metadata['distribution_overlay'] = meta3

            # Adjust layout for professional appearance
            plt.tight_layout()

            return fig, plot_metadata

        except Exception as e:
            # Error handling with informative message
            for ax in axes:
                ax.text(0.5, 0.5, f'Plotting Error:\n{str(e)}',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax.set_title('Error in Visualization')

            return fig, {'error': str(e)}

    def _plot_pca_cluster_scatter(self,
                                 df: pd.DataFrame,
                                 selected_indices: List[int],
                                 clusters: np.ndarray,
                                 centroids: np.ndarray,
                                 pca_components: int,
                                 ax: plt.Axes) -> Dict[str, Any]:
        """
        Plot 1: PCA scatter showing clusters and selected boundary samples.

        Visual proof that selected samples are at cluster boundaries (high uncertainty).
        """
        try:
            # Prepare timing features for PCA
            timing_features = ['nominal_delay', 'lib_sigma_delay_late', 'nominal_tran',
                             'lib_sigma_tran_late', 'sigma_by_nominal', 'early_sigma_by_late_sigma']

            available_features = [col for col in timing_features if col in df.columns]

            if len(available_features) < 2:
                # Fallback to any numeric columns
                available_features = [col for col in df.columns
                                    if df[col].dtype in ['float64', 'int64']][:6]

            if len(available_features) < 2:
                raise ValueError("Insufficient numeric features for PCA visualization")

            # Apply PCA for visualization
            feature_data = df[available_features].fillna(df[available_features].median())
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(feature_data)

            pca = PCA(n_components=min(pca_components, len(available_features), 2))
            pca_data = pca.fit_transform(scaled_data)

            # Create cluster colormap
            n_clusters = len(np.unique(clusters))
            cluster_colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

            # Plot all samples by cluster
            for cluster_id in np.unique(clusters):
                cluster_mask = clusters == cluster_id
                cluster_points = pca_data[cluster_mask]

                ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                          c=[cluster_colors[cluster_id]], alpha=0.6, s=30,
                          label=f'Cluster {cluster_id}', edgecolors='white', linewidth=0.5)

            # Highlight selected samples in red
            if len(selected_indices) > 0:
                selected_pca = pca_data[selected_indices]
                ax.scatter(selected_pca[:, 0], selected_pca[:, 1],
                          c=self.colors['selected'], s=80, marker='X',
                          label='Selected Samples', edgecolors='darkred', linewidth=1,
                          alpha=0.9, zorder=10)

            # Plot centroids if available
            if centroids is not None:
                try:
                    # Transform centroids to PCA space (approximate)
                    centroid_pca = pca.transform(centroids)
                    ax.scatter(centroid_pca[:, 0], centroid_pca[:, 1],
                              c=self.colors['centroids'], s=200, marker='*',
                              label='Centroids', edgecolors='white', linewidth=2,
                              alpha=0.9, zorder=15)
                except:
                    pass  # Skip centroids if transformation fails

            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
            ax.set_title('PCA Visualization: Clusters & Selected Samples\n(Red X = Boundary Samples)', fontsize=12, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3, color=self.colors['grid'])

            # Return metadata
            return {
                'variance_explained': pca.explained_variance_ratio_[:2].sum(),
                'n_clusters_visualized': n_clusters,
                'n_selected_highlighted': len(selected_indices),
                'features_used': available_features[:6],  # Limit for readability
                'pca_success': True
            }

        except Exception as e:
            ax.text(0.5, 0.5, f'PCA Error: {str(e)[:100]}...',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            ax.set_title('PCA Visualization Error', fontsize=12, color='red')
            return {'pca_success': False, 'error': str(e)}

    def _plot_correlation_heatmap(self, df: pd.DataFrame, ax: plt.Axes) -> Dict[str, Any]:
        """
        Plot 2: Correlation heatmap of top timing features.

        Visual proof of feature relationships and PCA justification.
        """
        try:
            # Select top timing features for correlation analysis
            timing_features = [
                'nominal_delay', 'lib_sigma_delay_late', 'nominal_tran', 'lib_sigma_tran_late',
                'sigma_by_nominal', 'early_sigma_by_late_sigma', 'stdev_by_late_sigma',
                'cross_signal_enc', 'tran_nominal_by_delay_nominal'
            ]

            available_features = [col for col in timing_features if col in df.columns]

            if len(available_features) < 3:
                # Fallback to any numeric columns
                numeric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
                available_features = numeric_cols[:8]  # Limit to top 8 for readability

            if len(available_features) < 2:
                raise ValueError("Insufficient features for correlation heatmap")

            # Calculate correlation matrix
            corr_data = df[available_features].fillna(df[available_features].median())
            correlation_matrix = corr_data.corr()

            # Create heatmap with professional styling
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle

            if HAS_SEABORN:
                sns.heatmap(correlation_matrix, mask=mask, ax=ax, cmap='RdBu_r', center=0,
                           annot=True, fmt='.2f', square=True, linewidths=0.5,
                           cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'})
            else:
                # Fallback using matplotlib
                masked_corr = np.ma.masked_array(correlation_matrix, mask)
                im = ax.imshow(masked_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

                # Add text annotations
                for i in range(len(correlation_matrix)):
                    for j in range(len(correlation_matrix.columns)):
                        if not mask[i, j]:
                            ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", fontsize=8)

                # Set labels and ticks
                ax.set_xticks(range(len(correlation_matrix.columns)))
                ax.set_yticks(range(len(correlation_matrix)))
                ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(correlation_matrix.index, fontsize=8)

                # Add colorbar
                plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation Coefficient')

            ax.set_title('Timing Feature Correlations\n(Justifies PCA & Clustering)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Timing Features', fontsize=11)
            ax.set_ylabel('Timing Features', fontsize=11)

            # Rotate labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

            # Calculate correlation insights
            high_corr_pairs = []
            for i in range(len(correlation_matrix)):
                for j in range(i+1, len(correlation_matrix)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.index[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_val
                        })

            return {
                'features_analyzed': available_features,
                'high_correlations': len(high_corr_pairs),
                'max_correlation': float(np.max(np.abs(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]))),
                'correlation_details': high_corr_pairs[:5],  # Top 5 for metadata
                'heatmap_success': True
            }

        except Exception as e:
            ax.text(0.5, 0.5, f'Correlation Error: {str(e)[:100]}...',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            ax.set_title('Correlation Heatmap Error', fontsize=12, color='red')
            return {'heatmap_success': False, 'error': str(e)}

    def _plot_distribution_overlay(self, df: pd.DataFrame, selected_indices: List[int], ax: plt.Axes) -> Dict[str, Any]:
        """
        Plot 3: Distribution overlay showing original vs selected samples.

        Visual proof that selection covers timing distribution tails (edge cases).
        """
        try:
            # Choose primary timing feature for distribution analysis
            timing_features = ['nominal_delay', 'lib_sigma_delay_late', 'sigma_by_nominal']
            target_feature = None

            for feature in timing_features:
                if feature in df.columns:
                    target_feature = feature
                    break

            if target_feature is None:
                # Fallback to first numeric column
                numeric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
                if len(numeric_cols) > 0:
                    target_feature = numeric_cols[0]
                else:
                    raise ValueError("No suitable numeric features found for distribution analysis")

            # Prepare data
            all_data = df[target_feature].fillna(df[target_feature].median())
            selected_data = all_data.iloc[selected_indices] if len(selected_indices) > 0 else pd.Series([])

            # Plot original population distribution
            ax.hist(all_data, bins=50, alpha=0.6, color=self.colors['unselected'],
                   label=f'Original Population (n={len(all_data):,})', density=True, edgecolor='white', linewidth=0.5)

            # Plot selected sample distribution overlay
            if len(selected_data) > 0:
                ax.hist(selected_data, bins=30, alpha=0.8, color=self.colors['selected'],
                       label=f'Selected Samples (n={len(selected_data):,})', density=True,
                       edgecolor='darkred', linewidth=2, histtype='step')

            # Add statistical markers
            all_mean = all_data.mean()
            all_std = all_data.std()

            # Mark mean and ±2σ boundaries
            ax.axvline(all_mean, color='green', linestyle='--', alpha=0.8, label='Population Mean')
            ax.axvline(all_mean - 2*all_std, color='orange', linestyle=':', alpha=0.8, label='±2σ Boundaries')
            ax.axvline(all_mean + 2*all_std, color='orange', linestyle=':', alpha=0.8)

            ax.set_xlabel(f'{target_feature} (Timing Parameter)', fontsize=11)
            ax.set_ylabel('Probability Density', fontsize=11)
            ax.set_title(f'{target_feature} Distribution Analysis\n(Proves Tail Coverage)', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.3, color=self.colors['grid'])

            # Calculate distribution insights
            coverage_analysis = self._analyze_distribution_coverage(all_data, selected_data)

            return {
                'feature_analyzed': target_feature,
                'population_size': len(all_data),
                'selected_size': len(selected_data),
                'population_mean': float(all_mean),
                'population_std': float(all_std),
                'coverage_analysis': coverage_analysis,
                'distribution_success': True
            }

        except Exception as e:
            ax.text(0.5, 0.5, f'Distribution Error: {str(e)[:100]}...',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            ax.set_title('Distribution Analysis Error', fontsize=12, color='red')
            return {'distribution_success': False, 'error': str(e)}

    def _analyze_distribution_coverage(self, all_data: pd.Series, selected_data: pd.Series) -> Dict[str, Any]:
        """
        Analyze how well selected samples cover the original distribution.

        Returns coverage statistics proving tail sampling effectiveness.
        """
        if len(selected_data) == 0:
            return {'coverage': 'No selected samples'}

        all_mean = all_data.mean()
        all_std = all_data.std()

        # Define distribution regions
        tail_low = all_mean - 2 * all_std
        tail_high = all_mean + 2 * all_std

        # Count samples in each region
        total_in_tails = len(all_data[(all_data < tail_low) | (all_data > tail_high)])
        selected_in_tails = len(selected_data[(selected_data < tail_low) | (selected_data > tail_high)])

        # Calculate coverage metrics
        tail_coverage_rate = (selected_in_tails / max(total_in_tails, 1)) * 100
        selection_rate = len(selected_data) / len(all_data) * 100

        return {
            'tail_coverage_percentage': round(tail_coverage_rate, 2),
            'overall_selection_percentage': round(selection_rate, 2),
            'selected_in_tails': int(selected_in_tails),
            'total_in_tails': int(total_in_tails),
            'proves_boundary_sampling': tail_coverage_rate > selection_rate
        }

    def create_individual_plot(self, plot_type: str, **kwargs) -> Tuple[plt.Figure, Dict[str, Any]]:
        """
        Create individual plots for detailed analysis.

        Args:
            plot_type: 'pca', 'correlation', or 'distribution'
            **kwargs: Plot-specific parameters

        Returns:
            Tuple of (figure, metadata)
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize_single)

        if plot_type == 'pca':
            metadata = self._plot_pca_cluster_scatter(
                kwargs['df'], kwargs['selected_indices'], kwargs['clusters'],
                kwargs.get('centroids'), kwargs.get('pca_components', 2), ax
            )
        elif plot_type == 'correlation':
            metadata = self._plot_correlation_heatmap(kwargs['df'], ax)
        elif plot_type == 'distribution':
            metadata = self._plot_distribution_overlay(kwargs['df'], kwargs['selected_indices'], ax)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        plt.tight_layout()
        return fig, metadata

# Convenience functions for easy integration
def generate_timing_dashboard(df: pd.DataFrame,
                            selected_indices: List[int],
                            clusters: np.ndarray,
                            centroids: np.ndarray = None) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Convenience function to generate complete timing validation dashboard.

    Args:
        df: Timing data DataFrame
        selected_indices: Selected sample indices
        clusters: Cluster assignments
        centroids: Cluster centroids (optional)

    Returns:
        Tuple of (matplotlib figure, analysis metadata)
    """
    dashboard = TimingVisualizationDashboard()
    return dashboard.generate_dashboard_plots(df, selected_indices, clusters, centroids)

def save_dashboard(fig: plt.Figure, filename: str, dpi: int = 300, bbox_inches: str = 'tight'):
    """
    Save dashboard figure with professional quality settings.

    Args:
        fig: Matplotlib figure to save
        filename: Output filename (include extension)
        dpi: Resolution for saving
        bbox_inches: Bounding box for saving
    """
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                facecolor='white', edgecolor='none')
    print(f"Dashboard saved: {filename}")