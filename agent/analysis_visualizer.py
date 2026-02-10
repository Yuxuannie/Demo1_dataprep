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

        # Distribution comparison
        html += self._create_distribution_comparison(numeric_data, selected_data)

        # Coverage analysis
        html += self._create_coverage_analysis(numeric_data, selected_data)

        # Outlier preservation analysis
        html += self._create_outlier_preservation(numeric_data, selected_data)

        # Representative sample check
        html += self._create_representativeness_check(numeric_data, selected_data)

        html += "</div>"
        return html

    def _create_distribution_comparison(self, original: pd.DataFrame, selected: pd.DataFrame) -> str:
        """Compare distributions between original and selected data"""
        html = """
        <h3>Distribution Preservation</h3>
        <table style="border-collapse: collapse; width: 100%;">
        <tr>
            <th style="border: 1px solid #ddd; padding: 8px;">Feature</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Original Mean±Std</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Selected Mean±Std</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Mean Diff</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Std Ratio</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Quality</th>
        </tr>"""

        for col in list(original.columns[:10]):
            orig_mean, orig_std = original[col].mean(), original[col].std()
            sel_mean, sel_std = selected[col].mean(), selected[col].std()

            mean_diff = abs(sel_mean - orig_mean) / orig_std if orig_std > 0 else 0
            std_ratio = sel_std / orig_std if orig_std > 0 else 1

            # Quality assessment
            if mean_diff < 0.2 and 0.8 <= std_ratio <= 1.2:
                quality = "EXCELLENT"
                color = "#d4edda"
            elif mean_diff < 0.5 and 0.6 <= std_ratio <= 1.4:
                quality = "GOOD"
                color = "#fff3cd"
            else:
                quality = "POOR"
                color = "#f8d7da"

            html += f"""
            <tr style="background-color: {color};">
                <td style="border: 1px solid #ddd; padding: 8px;"><strong>{col[:15]}</strong></td>
                <td style="border: 1px solid #ddd; padding: 8px;">{orig_mean:.3f}±{orig_std:.3f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{sel_mean:.3f}±{sel_std:.3f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{mean_diff:.3f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{std_ratio:.3f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;"><strong>{quality}</strong></td>
            </tr>"""

        html += "</table>"
        return html

    def _create_coverage_analysis(self, original: pd.DataFrame, selected: pd.DataFrame) -> str:
        """Analyze feature space coverage"""
        html = """
        <h3>Feature Space Coverage</h3>
        <table style="border-collapse: collapse; width: 100%;">
        <tr>
            <th style="border: 1px solid #ddd; padding: 8px;">Feature</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Original Range</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Selected Range</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Coverage %</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Missing Extremes</th>
        </tr>"""

        for col in list(original.columns[:10]):
            orig_min, orig_max = original[col].min(), original[col].max()
            sel_min, sel_max = selected[col].min(), selected[col].max()

            orig_range = orig_max - orig_min
            sel_range = sel_max - sel_min
            coverage = (sel_range / orig_range) * 100 if orig_range > 0 else 100

            # Check if extremes are missing
            missing_low = sel_min > orig_min + 0.1 * orig_range
            missing_high = sel_max < orig_max - 0.1 * orig_range
            missing_extremes = []
            if missing_low:
                missing_extremes.append("LOW")
            if missing_high:
                missing_extremes.append("HIGH")

            missing_text = ", ".join(missing_extremes) if missing_extremes else "None"

            html += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><strong>{col[:15]}</strong></td>
                <td style="border: 1px solid #ddd; padding: 8px;">{orig_min:.3f} to {orig_max:.3f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{sel_min:.3f} to {sel_max:.3f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{coverage:.1f}%</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{missing_text}</td>
            </tr>"""

        html += "</table>"
        return html

    def _create_outlier_preservation(self, original: pd.DataFrame, selected: pd.DataFrame) -> str:
        """Check if outliers are preserved in selection"""
        from scipy import stats

        html = """
        <h3>Outlier Preservation Analysis</h3>
        <table style="border-collapse: collapse; width: 100%;">
        <tr>
            <th style="border: 1px solid #ddd; padding: 8px;">Feature</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Original Outliers</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Selected Outliers</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Preservation Rate</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Assessment</th>
        </tr>"""

        for col in list(original.columns[:8]):
            # Use Z-score method to identify outliers
            orig_z_scores = np.abs(stats.zscore(original[col]))
            sel_z_scores = np.abs(stats.zscore(selected[col]))

            orig_outliers = (orig_z_scores > 2.5).sum()
            sel_outliers = (sel_z_scores > 2.5).sum()

            # Calculate expected outliers in selection
            selection_ratio = len(selected) / len(original)
            expected_outliers = orig_outliers * selection_ratio

            preservation_rate = (sel_outliers / expected_outliers) if expected_outliers > 0 else 1.0

            if preservation_rate >= 0.8:
                assessment = "GOOD"
                color = "#d4edda"
            elif preservation_rate >= 0.5:
                assessment = "FAIR"
                color = "#fff3cd"
            else:
                assessment = "POOR"
                color = "#f8d7da"

            html += f"""
            <tr style="background-color: {color};">
                <td style="border: 1px solid #ddd; padding: 8px;"><strong>{col[:15]}</strong></td>
                <td style="border: 1px solid #ddd; padding: 8px;">{orig_outliers}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{sel_outliers}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{preservation_rate:.1%}</td>
                <td style="border: 1px solid #ddd; padding: 8px;"><strong>{assessment}</strong></td>
            </tr>"""

        html += "</table>"
        return html

    def _create_representativeness_check(self, original: pd.DataFrame, selected: pd.DataFrame) -> str:
        """Overall representativeness assessment"""
        selection_ratio = len(selected) / len(original)

        html = f"""
        <h3>Representativeness Summary</h3>
        <div class="summary">
            <p><strong>Selection Ratio:</strong> {selection_ratio:.1%} ({len(selected)} of {len(original)} samples)</p>
        """

        # Calculate overall quality metrics
        total_features = len(original.columns)
        quality_scores = []

        for col in original.columns[:15]:  # Limit to prevent performance issues
            if original[col].dtype in ['int64', 'float64']:
                orig_mean, orig_std = original[col].mean(), original[col].std()
                sel_mean = selected[col].mean()

                if orig_std > 0:
                    mean_diff = abs(sel_mean - orig_mean) / orig_std
                    quality_scores.append(1 - min(mean_diff, 1))

        avg_quality = np.mean(quality_scores) if quality_scores else 0.5

        if avg_quality > 0.8:
            overall_assessment = "EXCELLENT - Highly representative sample"
            color = "#d4edda"
        elif avg_quality > 0.6:
            overall_assessment = "GOOD - Representative sample with minor deviations"
            color = "#fff3cd"
        elif avg_quality > 0.4:
            overall_assessment = "FAIR - Some features may be under/over-represented"
            color = "#ffeaa7"
        else:
            overall_assessment = "POOR - Sample may not be representative"
            color = "#f8d7da"

        html += f"""
            <p><strong>Overall Quality Score:</strong> {avg_quality:.3f}</p>
            <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <strong>Assessment:</strong> {overall_assessment}
            </div>
        """

        # Recommendations
        html += """
            <p><strong>Recommendations:</strong></p>
            <ul>
        """

        if avg_quality < 0.6:
            html += "<li>Consider increasing sample size for better representation</li>"
        if selection_ratio < 0.05:
            html += "<li>Very small sample - results may not generalize well</li>"
        elif selection_ratio > 0.3:
            html += "<li>Large sample selected - consider if smaller subset would suffice</li>"

        html += """
            <li>Check distribution plots above for feature-specific issues</li>
            <li>Verify outlier preservation matches your analysis needs</li>
            </ul>
        </div>
        """

        return html

    def open_in_browser(self, html_file_path: str):
        """Open the HTML file in the default browser"""
        try:
            webbrowser.open(f"file://{html_file_path}")
            return True
        except Exception as e:
            print(f"Could not open browser: {e}")
            return False