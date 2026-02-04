"""
Flask Server for Demo 1: Data Preparation Agent
Handles file upload, agent execution, and results delivery
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
import pandas as pd
import numpy as np
import json
from werkzeug.utils import secure_filename

# Add agent to path
sys.path.append(os.path.dirname(__file__))
from agent.data_prep_agent import DataPrepAgent

# Try to load LLM configuration (optional)
try:
    from config import QWEN_CONFIG
    LLM_ENABLED = QWEN_CONFIG.get('enabled', False)
    LLM_ENDPOINT = QWEN_CONFIG.get('api_url') if LLM_ENABLED else None
    LLM_API_KEY = QWEN_CONFIG.get('api_key') if LLM_ENABLED else None
    print(f"✓ LLM Configuration loaded: {'Enabled' if LLM_ENABLED else 'Disabled (using templates)'}")
except ImportError:
    LLM_ENABLED = False
    LLM_ENDPOINT = None
    LLM_API_KEY = None
    print("✓ No LLM config found - using template-based reasoning")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['SECRET_KEY'] = 'demo1-secret-key'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global state (in production, use session/database)
current_result = None


@app.route('/')
def index():
    """Serve main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle CSV upload.
    Returns preview of data.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be CSV'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and preview
        df = pd.read_csv(filepath)
        
        preview = {
            'filename': filename,
            'filepath': filepath,
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'columns': df.columns.tolist(),
            'preview': df.head(10).to_dict('records')
        }
        
        return jsonify(preview)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/run_agent', methods=['POST'])
def run_agent():
    """
    Execute data prep agent.
    Returns reasoning log and results.
    """
    global current_result
    
    data = request.json
    filepath = data.get('filepath')
    target_ratio = data.get('target_ratio', 0.05)
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 400
    
    try:
        # Run agent (with LLM if configured)
        if LLM_ENABLED:
            agent = DataPrepAgent(llm_endpoint=LLM_ENDPOINT, llm_api_key=LLM_API_KEY)
            print(f"✓ Using LLM-enhanced reasoning: {LLM_ENDPOINT}")
        else:
            agent = DataPrepAgent()
            print("✓ Using template-based reasoning")
        
        result = agent.run(filepath, target_ratio=target_ratio)
        
        # Store result
        current_result = result
        
        # Prepare response (serialize numpy arrays)
        response = {
            'observation': {
                'total_samples': result['observation']['total_samples'],
                'n_features': result['observation']['n_features'],
                'feature_names': result['observation']['feature_names'],
                'cell_types': result['observation']['cell_types'],
                'high_correlations': result['observation']['high_correlations']
            },
            'strategy': result['strategy'],
            'decision': {
                'pca': {
                    'n_components': result['decision']['pca']['n_components'],
                    'variance_explained': result['decision']['pca']['variance_explained']
                },
                'clustering': {
                    'algorithm': result['decision']['clustering']['algorithm'],
                    'n_clusters': result['decision']['clustering']['n_clusters']
                },
                'cluster_info': result['decision']['cluster_info']
            },
            'result': {
                'n_selected': result['result']['n_selected'],
                'cluster_distribution': result['result']['cluster_distribution'],
                'uncertainty_weights': result['result']['uncertainty_weights']
            },
            'reasoning_log': result['reasoning_log']
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/get_visualizations', methods=['GET'])
def get_visualizations():
    """
    Generate visualization data for Plotly.
    """
    global current_result
    
    if current_result is None:
        return jsonify({'error': 'No results available'}), 400
    
    try:
        decision = current_result['decision']
        result = current_result['result']
        
        # PCA scatter plot data
        features_pca = decision['features_pca']
        labels = decision['clustering']['labels']
        distances = decision['clustering']['distances']
        selected_indices = result['selected_indices']
        
        # Prepare PCA plot (first 2 components)
        pca_data = {
            'all_points': {
                'x': features_pca[:, 0].tolist(),
                'y': features_pca[:, 1].tolist(),
                'cluster': labels.tolist(),
                'distance': distances.tolist()
            },
            'selected_points': {
                'x': features_pca[selected_indices, 0].tolist(),
                'y': features_pca[selected_indices, 1].tolist(),
                'cluster': labels[selected_indices].tolist()
            },
            'centroids': {
                'x': decision['clustering']['centroids'][:, 0].tolist(),
                'y': decision['clustering']['centroids'][:, 1].tolist()
            }
        }
        
        # Cluster distribution
        cluster_dist = {
            'clusters': list(range(decision['clustering']['n_clusters'])),
            'total': [int(np.sum(labels == i)) for i in range(decision['clustering']['n_clusters'])],
            'selected': result['cluster_distribution']
        }
        
        # Distance histogram
        distance_hist = {
            'all_distances': distances.tolist(),
            'selected_distances': distances[selected_indices].tolist()
        }
        
        return jsonify({
            'pca_plot': pca_data,
            'cluster_distribution': cluster_dist,
            'distance_histogram': distance_hist
        })
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/download_results', methods=['GET'])
def download_results():
    """
    Download selected samples as CSV.
    """
    global current_result
    
    if current_result is None:
        return jsonify({'error': 'No results available'}), 400
    
    try:
        selected_df = current_result['result']['selected_df']
        
        # Save to temp file
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'selected_samples.csv')
        selected_df.to_csv(output_path, index=False)
        
        return send_file(output_path, as_attachment=True, download_name='selected_samples.csv')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Generate mock data if needed
    mock_path = 'mock_data/test_data.csv'
    if not os.path.exists(mock_path):
        print("Generating mock data...")
        from mock_data.generate_mock import MockDataGenerator
        generator = MockDataGenerator(n_rows=21818)
        generator.save(mock_path)
        print(f"✓ Mock data generated: {mock_path}")
    
    print("\n" + "="*80)
    print("🚀 Demo 1: Data Preparation Agent")
    print("="*80)
    print("\nServer starting at: http://localhost:5000")
    print("\nUpload your CSV or use mock data at:", mock_path)
    print("\n" + "="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
