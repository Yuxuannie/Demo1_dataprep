"""
Analysis Visualization Module
Generates HTML plots for data analysis results
"""

import pandas as pd
import numpy as np
import os
import webbrowser
from typing import Dict, Any, List
import tempfile
from pathlib import Path


class AnalysisVisualizer:
    """Generate HTML visualizations for analysis results"""

    def __init__(self):
        self.output_dir = Path(tempfile.gettempdir()) / "claude_analysis_plots"
        self.output_dir.mkdir(exist_ok=True)

    def generate_analysis_plots(self, data: pd.DataFrame, analysis_results: Dict[str, Any],
                              selected_indices: List[int] = None) -> str:
        """Generate comprehensive analysis plots and return HTML file path"""

        # Create HTML content
        html_content = self._create_html_template()

        # Add data overview
        html_content += self._create_data_overview(data, analysis_results)

        # Add correlation heatmap
        html_content += self._create_correlation_plot(data)

        # Add distribution plots
        html_content += self._create_distribution_plots(data)

        # Add clustering visualization
        html_content += self._create_clustering_plot(data, analysis_results)

        # Add selection comparison if samples selected
        if selected_indices:
            html_content += self._create_selection_comparison(data, selected_indices)
            html_content += self._create_validation_plots(data, selected_indices)

        # Close HTML
        html_content += """
        </div>
        </body>
        </html>
        """

        # Save to file
        html_file = self.output_dir / f"analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
        html_file.write_text(html_content)

        return str(html_file)

    def _create_html_template(self) -> str:
        """Create HTML template with CSS styling"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Real Intelligence Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
                h1 { color: #2c3e50; text-align: center; }
                h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                .metric { display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; }
                .plot-container { margin: 20px 0; }
                .summary { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }
            </style>
        </head>
        <body>
        <div class="container">
            <h1>Real Intelligence Analysis Report</h1>
        """

    def _create_data_overview(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> str:
        """Create data overview section"""
        basic_stats = analysis_results.get('basic_stats', {})
        clustering = analysis_results.get('clustering_potential', {})
        outliers = analysis_results.get('outlier_analysis', {})

        return f"""
        <h2>Data Overview</h2>
        <div class="summary">
            <div class="metric">
                <strong>Samples:</strong> {basic_stats.get('n_samples', 0)}
            </div>
            <div class="metric">
                <strong>Features:</strong> {basic_stats.get('n_features', 0)}
            </div>
            <div class="metric">
                <strong>Hopkins Statistic:</strong> {clustering.get('hopkins_statistic', 0):.3f}
            </div>
            <div class="metric">
                <strong>Optimal Clusters:</strong> {clustering.get('optimal_cluster_count', 0)}
            </div>
            <div class="metric">
                <strong>Outlier Ratio:</strong> {outliers.get('global_outlier_ratio', 0):.1%}
            </div>
        </div>
        """

    def _create_correlation_plot(self, data: pd.DataFrame) -> str:
        """Create correlation heatmap using simple HTML table"""
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty or len(numeric_data.columns) < 2:
            return "<p>No numeric data available for correlation analysis.</p>"

        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()

        # Create simple correlation table (first 10 features for readability)
        features = list(corr_matrix.columns[:10])

        html = """
        <h2>Feature Correlations (Top 10 Features)</h2>
        <div class="plot-container">
        <table style="border-collapse: collapse; width: 100%;">
        <tr><th style="border: 1px solid #ddd; padding: 8px;">Feature</th>"""

        for feat in features:
            html += f'<th style="border: 1px solid #ddd; padding: 8px;">{feat[:10]}</th>'
        html += "</tr>"

        for i, feat1 in enumerate(features):
            html += f'<tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>{feat1[:10]}</strong></td>'
            for feat2 in features:
                corr_val = corr_matrix.loc[feat1, feat2]
                # Color code correlations
                if abs(corr_val) > 0.7:
                    color = "#ff6b6b"  # Red for strong correlation
                elif abs(corr_val) > 0.5:
                    color = "#ffd93d"  # Yellow for moderate correlation
                else:
                    color = "#6bcf7f"  # Green for weak correlation

                html += f'<td style="border: 1px solid #ddd; padding: 8px; background-color: {color};">{corr_val:.2f}</td>'
            html += "</tr>"

        html += "</table></div>"
        return html

    def _create_distribution_plots(self, data: pd.DataFrame) -> str:
        """Create distribution summary"""
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return "<p>No numeric data available for distribution analysis.</p>"

        html = """
        <h2>Feature Distributions</h2>
        <div class="plot-container">
        <table style="border-collapse: collapse; width: 100%;">
        <tr>
            <th style="border: 1px solid #ddd; padding: 8px;">Feature</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Mean</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Std</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Min</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Max</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Skewness</th>
        </tr>"""

        # Show stats for first 15 features
        for col in list(numeric_data.columns[:15]):
            values = numeric_data[col]
            skewness = values.skew()

            html += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><strong>{col[:20]}</strong></td>
                <td style="border: 1px solid #ddd; padding: 8px;">{values.mean():.3f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{values.std():.3f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{values.min():.3f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{values.max():.3f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{skewness:.3f}</td>
            </tr>"""

        html += "</table></div>"
        return html

    def _create_clustering_plot(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> str:
        """Create clustering analysis summary"""
        clustering = analysis_results.get('clustering_potential', {})

        html = f"""
        <h2>Clustering Analysis</h2>
        <div class="summary">
            <p><strong>Hopkins Statistic:</strong> {clustering.get('hopkins_statistic', 0):.3f}</p>
            <p><strong>Clustering Tendency:</strong> {clustering.get('clustering_tendency', 'unknown')}</p>
            <p><strong>Optimal Cluster Count:</strong> {clustering.get('optimal_cluster_count', 0)}</p>
            <p><strong>Max Silhouette Score:</strong> {clustering.get('max_silhouette_score', 0):.3f}</p>
        """

        # Add interpretation
        hopkins = clustering.get('hopkins_statistic', 0)
        if hopkins > 0.75:
            interpretation = "Strong clustering structure detected - data has clear groups"
        elif hopkins > 0.5:
            interpretation = "Moderate clustering structure - some grouping present"
        else:
            interpretation = "Weak clustering structure - data appears randomly distributed"

        html += f"<p><strong>Interpretation:</strong> {interpretation}</p></div>"
        return html

    def _create_selection_comparison(self, data: pd.DataFrame, selected_indices: List[int]) -> str:
        """Create comparison between original and selected samples"""
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return "<p>No numeric data for selection comparison.</p>"

        selected_data = numeric_data.iloc[selected_indices]

        html = """
        <h2>Selection Quality Comparison</h2>
        <div class="plot-container">
        <table style="border-collapse: collapse; width: 100%;">
        <tr>
            <th style="border: 1px solid #ddd; padding: 8px;">Feature</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Original Mean</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Selected Mean</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Difference</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Coverage</th>
        </tr>"""

        for col in list(numeric_data.columns[:10]):
            orig_mean = numeric_data[col].mean()
            sel_mean = selected_data[col].mean()
            diff = abs(sel_mean - orig_mean)

            # Calculate coverage (range preservation)
            orig_range = numeric_data[col].max() - numeric_data[col].min()
            sel_range = selected_data[col].max() - selected_data[col].min()
            coverage = (sel_range / orig_range) if orig_range > 0 else 0

            html += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><strong>{col[:20]}</strong></td>
                <td style="border: 1px solid #ddd; padding: 8px;">{orig_mean:.3f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{sel_mean:.3f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{diff:.3f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{coverage:.1%}</td>
            </tr>"""

        html += f"""
        </table>
        <div class="summary">
            <p><strong>Selection Summary:</strong></p>
            <p>Selected {len(selected_indices)} samples out of {len(data)} total samples ({len(selected_indices)/len(data):.1%})</p>
        </div>
        </div>"""

        return html

    def _create_validation_plots(self, data: pd.DataFrame, selected_indices: List[int]) -> str:
        """Create validation plots to assess sample quality"""
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return "<p>No numeric data for validation plots.</p>"

        selected_data = numeric_data.iloc[selected_indices]

        html = """
        <h2>Sample Quality Validation</h2>
        <div class="plot-container">
        """

        # Create actual interactive plots instead of tables
        html += self._create_distribution_comparison_plot(numeric_data, selected_data)
        html += self._create_scatter_plot_comparison(numeric_data, selected_data)
        html += self._create_box_plot_comparison(numeric_data, selected_data)
        html += self._create_coverage_visualization(numeric_data, selected_data)

        html += "</div>"
        return html

    def _create_distribution_comparison_plot(self, original: pd.DataFrame, selected: pd.DataFrame) -> str:
        """Create distribution comparison plots using Plotly"""
        import plotly.graph_objs as go
        import plotly.io as pio
        import random

        # Get first few features for plotting
        features = list(original.columns[:4])  # Limit to 4 features for readability

        html = '<h3>Distribution Comparison</h3>'

        for i, col in enumerate(features):
            plot_id = f'distribution_plot_{i}_{random.randint(1000, 9999)}'

            # Create histogram overlays
            fig = go.Figure()

            # Original data histogram
            fig.add_trace(go.Histogram(
                x=original[col],
                name='Original',
                opacity=0.7,
                nbinsx=20,
                marker_color='blue'
            ))

            # Selected data histogram
            fig.add_trace(go.Histogram(
                x=selected[col],
                name='Selected',
                opacity=0.7,
                nbinsx=20,
                marker_color='red'
            ))

            fig.update_layout(
                title=f'Distribution Comparison: {col}',
                xaxis_title=col,
                yaxis_title='Frequency',
                barmode='overlay',
                showlegend=True,
                height=300,
                margin=dict(l=40, r=40, t=60, b=40)
            )

            plot_html = pio.to_html(fig, include_plotlyjs=False, div_id=plot_id)
            html += f'<div class="plot-container">{plot_html}</div>'

        return html

    def _create_scatter_plot_comparison(self, original: pd.DataFrame, selected: pd.DataFrame) -> str:
        """Create scatter plot to show selection distribution"""
        import plotly.graph_objs as go
        import plotly.io as pio
        import random

        if len(original.columns) < 2:
            return '<h3>Scatter Plot</h3><p>Need at least 2 features for scatter plot.</p>'

        col1, col2 = original.columns[0], original.columns[1]
        plot_id = f'scatter_plot_{random.randint(1000, 9999)}'

        fig = go.Figure()

        # Original data points
        fig.add_trace(go.Scatter(
            x=original[col1],
            y=original[col2],
            mode='markers',
            name='Original',
            marker=dict(color='lightblue', size=4, opacity=0.6)
        ))

        # Selected data points
        fig.add_trace(go.Scatter(
            x=selected[col1],
            y=selected[col2],
            mode='markers',
            name='Selected',
            marker=dict(color='red', size=8, opacity=0.8)
        ))

        fig.update_layout(
            title=f'Sample Distribution: {col1} vs {col2}',
            xaxis_title=col1,
            yaxis_title=col2,
            showlegend=True,
            height=400,
            margin=dict(l=40, r=40, t=60, b=40)
        )

        plot_html = pio.to_html(fig, include_plotlyjs=False, div_id=plot_id)
        return f'<h3>Sample Distribution</h3><div class="plot-container">{plot_html}</div>'

    def _create_box_plot_comparison(self, original: pd.DataFrame, selected: pd.DataFrame) -> str:
        """Create box plot comparison to show distribution statistics"""
        import plotly.graph_objs as go
        import plotly.io as pio
        import random

        features = list(original.columns[:4])  # Limit to 4 features
        plot_id = f'box_plot_{random.randint(1000, 9999)}'

        fig = go.Figure()

        for col in features:
            # Original data box plot
            fig.add_trace(go.Box(
                y=original[col],
                name=f'{col} (Original)',
                boxmean=True,
                marker_color='lightblue'
            ))

            # Selected data box plot
            fig.add_trace(go.Box(
                y=selected[col],
                name=f'{col} (Selected)',
                boxmean=True,
                marker_color='red'
            ))

        fig.update_layout(
            title='Box Plot Comparison: Original vs Selected',
            yaxis_title='Values',
            showlegend=True,
            height=400,
            margin=dict(l=40, r=40, t=60, b=40)
        )

        plot_html = pio.to_html(fig, include_plotlyjs=False, div_id=plot_id)
        return f'<h3>Statistical Distribution Comparison</h3><div class="plot-container">{plot_html}</div>'

    def _create_coverage_visualization(self, original: pd.DataFrame, selected: pd.DataFrame) -> str:
        """Create coverage visualization using bar chart"""
        import plotly.graph_objs as go
        import plotly.io as pio
        import random

        features = list(original.columns[:6])  # Limit to 6 features for readability
        coverage_data = []

        for col in features:
            orig_min, orig_max = original[col].min(), original[col].max()
            sel_min, sel_max = selected[col].min(), selected[col].max()

            orig_range = orig_max - orig_min
            sel_range = sel_max - sel_min
            coverage = (sel_range / orig_range) * 100 if orig_range > 0 else 100

            coverage_data.append({
                'feature': col,
                'coverage': coverage
            })

        plot_id = f'coverage_plot_{random.randint(1000, 9999)}'

        fig = go.Figure()

        colors = ['green' if x['coverage'] >= 80 else 'orange' if x['coverage'] >= 60 else 'red'
                  for x in coverage_data]

        fig.add_trace(go.Bar(
            x=[x['feature'] for x in coverage_data],
            y=[x['coverage'] for x in coverage_data],
            marker_color=colors,
            text=[f"{x['coverage']:.1f}%" for x in coverage_data],
            textposition='auto'
        ))

        fig.update_layout(
            title='Feature Range Coverage',
            xaxis_title='Features',
            yaxis_title='Coverage Percentage',
            yaxis_range=[0, 100],
            height=300,
            margin=dict(l=40, r=40, t=60, b=40)
        )

        # Add horizontal line at 80% (good coverage threshold)
        fig.add_hline(y=80, line_dash="dash", line_color="gray",
                      annotation_text="Good Coverage Threshold")

        plot_html = pio.to_html(fig, include_plotlyjs=False, div_id=plot_id)

        # Add summary
        selection_ratio = len(selected) / len(original)
        avg_coverage = np.mean([x['coverage'] for x in coverage_data])

        summary_html = f"""
        <div class="summary">
            <p><strong>Selection Summary:</strong></p>
            <p>Selected {len(selected)} samples out of {len(original)} total ({selection_ratio:.1%})</p>
            <p>Average feature coverage: {avg_coverage:.1f}%</p>
            <p>Coverage interpretation: {'Excellent' if avg_coverage >= 80 else 'Good' if avg_coverage >= 60 else 'Needs improvement'}</p>
        </div>
        """

        return f'<h3>Feature Coverage Analysis</h3><div class="plot-container">{plot_html}</div>{summary_html}'

    def open_in_browser(self, html_file_path: str):
        """Open the HTML file in the default browser"""
        try:
            webbrowser.open(f"file://{html_file_path}")
            return True
        except Exception as e:
            print(f"Could not open browser: {e}")
            return False