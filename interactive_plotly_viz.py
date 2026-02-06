"""
Interactive Plotly Visualization for Timing Data Selection
Replaces static matplotlib with interactive dashboard capabilities
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os

class InteractiveTimingDashboard:
    """
    Interactive dashboard with Plotly for timing data validation and exploration.
    """

    def __init__(self):
        self.color_palette = {
            'selected': '#FF6B6B',    # Red for selected samples
            'unselected': '#4ECDC4',  # Teal for unselected
            'cluster_1': '#45B7D1',   # Blue
            'cluster_2': '#96CEB4',   # Green
            'cluster_3': '#FFEAA7',   # Yellow
            'cluster_4': '#DDA0DD',   # Plum
            'cluster_5': '#F4A261',   # Orange
            'background': '#2F3542'   # Dark background
        }

    def generate_interactive_dashboard(
        self,
        df: pd.DataFrame,
        selected_indices: List[int],
        clusters: np.ndarray,
        centroids: np.ndarray,
        pca_components: Optional[np.ndarray] = None,
        export_html: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive interactive dashboard with Plotly.

        Args:
            df: Original dataset
            selected_indices: Indices of selected samples
            clusters: Cluster assignments for each sample
            centroids: Cluster centroids
            pca_components: PCA transformed data (optional)
            export_html: Whether to export standalone HTML

        Returns:
            Dictionary containing Plotly figures and metadata
        """

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

        # Plot 1: Sample Selection Overview
        self._add_selection_overview(fig, dashboard_data, row=1, col=1)

        # Plot 2: Cluster Quality Analysis
        self._add_cluster_analysis(fig, dashboard_data, centroids, row=1, col=2)

        # Plot 3: Selection Statistics
        self._add_selection_statistics(fig, dashboard_data, row=2, col=1)

        # Plot 4: Data Distribution
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

    def _prepare_dashboard_data(
        self,
        df: pd.DataFrame,
        selected_indices: List[int],
        clusters: np.ndarray,
        pca_components: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
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
                    color=self.color_palette['selected'],
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
                marker_color=self.color_palette['selected'],
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
                    line=dict(color=self.color_palette['unselected'], width=2)
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

        # All data histogram
        fig.add_trace(
            go.Histogram(
                x=coords['x'],
                name='All Data',
                opacity=0.5,
                marker_color=self.color_palette['unselected'],
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
                marker_color=self.color_palette['selected'],
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

        print(f"📊 Interactive dashboard exported: {html_path}")
        return html_path

    def create_comparison_dashboard(
        self,
        before_data: Dict[str, Any],
        after_data: Dict[str, Any],
        export_html: bool = True
    ) -> Dict[str, Any]:
        """
        Create side-by-side comparison dashboard for before/after analysis.

        Args:
            before_data: Previous selection results
            after_data: Current selection results
            export_html: Whether to export standalone HTML

        Returns:
            Dictionary containing comparison figure and metadata
        """

        # Create comparison subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Before: Selection Overview',
                'After: Selection Overview',
                'Before: Cluster Statistics',
                'After: Cluster Statistics'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}]
            ],
            horizontal_spacing=0.12,
            vertical_spacing=0.15
        )

        # Add before/after comparison plots
        self._add_comparison_plots(fig, before_data, after_data)

        # Update layout
        fig.update_layout(
            title={
                'text': 'Agentic Selection Comparison Dashboard',
                'x': 0.5,
                'font': {'size': 24, 'color': 'white'}
            },
            template='plotly_dark',
            height=800,
            width=1200,
            showlegend=True
        )

        # Export if requested
        html_path = None
        if export_html:
            html_filename = "timing_comparison_dashboard.html"
            html_path = os.path.join(os.getcwd(), html_filename)
            fig.write_html(html_path, include_plotlyjs=True)
            print(f"📊 Comparison dashboard exported: {html_path}")

        return {
            'comparison_figure': fig,
            'html_export_path': html_path,
            'before_summary': before_data.get('summary', {}),
            'after_summary': after_data.get('summary', {})
        }

    def _add_comparison_plots(self, fig, before_data: Dict, after_data: Dict):
        """Add before/after comparison plots to subplot figure."""

        # Before plots (left side)
        if 'coordinates' in before_data and 'selection_mask' in before_data:
            coords_before = before_data['coordinates']
            mask_before = before_data['selection_mask']

            # Before selection overview
            fig.add_trace(
                go.Scatter(
                    x=coords_before['x'][~mask_before],
                    y=coords_before['y'][~mask_before],
                    mode='markers',
                    marker=dict(color='lightgray', size=4, opacity=0.3),
                    name='Before: Unselected',
                    showlegend=True
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=coords_before['x'][mask_before],
                    y=coords_before['y'][mask_before],
                    mode='markers',
                    marker=dict(color='orange', size=8, opacity=0.7),
                    name='Before: Selected',
                    showlegend=True
                ),
                row=1, col=1
            )

        # After plots (right side)
        if 'coordinates' in after_data and 'selection_mask' in after_data:
            coords_after = after_data['coordinates']
            mask_after = after_data['selection_mask']

            # After selection overview
            fig.add_trace(
                go.Scatter(
                    x=coords_after['x'][~mask_after],
                    y=coords_after['y'][~mask_after],
                    mode='markers',
                    marker=dict(color='lightgray', size=4, opacity=0.3),
                    name='After: Unselected',
                    showlegend=True
                ),
                row=1, col=2
            )

            fig.add_trace(
                go.Scatter(
                    x=coords_after['x'][mask_after],
                    y=coords_after['y'][mask_after],
                    mode='markers',
                    marker=dict(color=self.color_palette['selected'], size=8, opacity=0.7),
                    name='After: Selected',
                    showlegend=True
                ),
                row=1, col=2
            )


def generate_plotly_dashboard(
    df: pd.DataFrame,
    selected_indices: List[int],
    clusters: np.ndarray,
    centroids: np.ndarray,
    pca_components: Optional[np.ndarray] = None,
    export_html: bool = True
) -> Dict[str, Any]:
    """
    Main function to generate interactive Plotly dashboard.
    Replaces the static matplotlib generate_dashboard_plots() function.

    Args:
        df: Original timing dataset
        selected_indices: List of selected sample indices
        clusters: Cluster assignments array
        centroids: Cluster centroid coordinates
        pca_components: Optional PCA-transformed coordinates
        export_html: Whether to export standalone HTML file

    Returns:
        Dictionary containing interactive dashboard results
    """

    dashboard = InteractiveTimingDashboard()

    try:
        results = dashboard.generate_interactive_dashboard(
            df=df,
            selected_indices=selected_indices,
            clusters=clusters,
            centroids=centroids,
            pca_components=pca_components,
            export_html=export_html
        )

        print("✅ Interactive Plotly dashboard generated successfully")
        print(f"   • Selected: {len(selected_indices)} samples")
        print(f"   • Clusters: {len(np.unique(clusters))}")
        print(f"   • Features: Zoom, Pan, Hover, Toggle traces")

        if export_html and results['html_export_path']:
            print(f"   • HTML Export: {results['html_export_path']}")

        return results

    except Exception as e:
        print(f"❌ Dashboard generation failed: {e}")

        # Fallback: simple scatter plot
        import plotly.graph_objects as go

        fallback_fig = go.Figure()
        fallback_fig.add_scatter(
            x=list(range(len(df))),
            y=[1] * len(df),
            mode='markers',
            marker=dict(color=['red' if i in selected_indices else 'blue' for i in range(len(df))]),
            name='Sample Selection'
        )

        return {
            'plotly_figure': fallback_fig,
            'dashboard_data': {'error': str(e)},
            'html_export_path': None,
            'interactive_features': {'basic_plot': True}
        }