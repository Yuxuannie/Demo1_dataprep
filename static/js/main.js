// Main JavaScript for Demo 1

let currentFilePath = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initializeUpload();
    initializeButtons();
});

// File Upload Handling
function initializeUpload() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    
    // Click to upload
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect();
        }
    });
}

async function handleFileSelect() {
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];
    
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        // Show preview
        displayFilePreview(data);
        currentFilePath = data.filepath;
        
    } catch (error) {
        alert('Upload failed: ' + error.message);
    }
}

function displayFilePreview(data) {
    const previewDiv = document.getElementById('file-preview');
    const infoDiv = document.getElementById('preview-info');
    
    infoDiv.innerHTML = `
        <p><strong>Filename:</strong> ${data.filename}</p>
        <p><strong>Samples:</strong> ${data.n_rows.toLocaleString()}</p>
        <p><strong>Features:</strong> ${data.n_cols}</p>
        <p><strong>Columns:</strong> ${data.columns.slice(0, 5).join(', ')}${data.columns.length > 5 ? '...' : ''}</p>
    `;
    
    previewDiv.classList.remove('hidden');
}

// Button Handlers
function initializeButtons() {
    document.getElementById('run-agent-btn').addEventListener('click', runAgent);
    document.getElementById('download-btn').addEventListener('click', downloadResults);
}

async function runAgent() {
    if (!currentFilePath) {
        alert('Please upload a file first');
        return;
    }
    
    const targetRatio = parseFloat(document.getElementById('target-ratio').value) / 100;
    
    // Show reasoning section
    const reasoningSection = document.getElementById('reasoning-section');
    reasoningSection.classList.remove('hidden');
    
    // Reset stages
    resetStages();
    
    // Disable button
    const btn = document.getElementById('run-agent-btn');
    btn.disabled = true;
    btn.textContent = 'Running...';
    
    try {
        const response = await fetch('/run_agent', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                filepath: currentFilePath,
                target_ratio: targetRatio
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert('Error: ' + data.error);
            btn.disabled = false;
            btn.textContent = 'Run Agent';
            return;
        }
        
        // Display reasoning
        displayReasoning(data);
        
        // Load visualizations
        await loadVisualizations(data);
        
        // Show results section
        document.getElementById('results-section').classList.remove('hidden');
        
    } catch (error) {
        alert('Execution failed: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run Agent';
    }
}

function resetStages() {
    const stages = ['observe', 'think', 'decide', 'act'];
    stages.forEach(stage => {
        document.getElementById(`${stage}-status`).textContent = '⏳';
        document.getElementById(`${stage}-text`).textContent = '';
        document.getElementById(`${stage}-content`).classList.remove('active');
    });
}

function displayReasoning(data) {
    // Map reasoning log to stages
    const stageMap = {
        'OBSERVE': 'observe',
        'THINK': 'think',
        'DECIDE': 'decide',
        'ACT': 'act'
    };
    
    data.reasoning_log.forEach(entry => {
        const stageId = stageMap[entry.stage];
        if (stageId) {
            document.getElementById(`${stageId}-text`).textContent = entry.content;
            document.getElementById(`${stageId}-status`).textContent = '✅';
            document.getElementById(`${stageId}-content`).classList.add('active');
        }
    });
    
    // Update metrics
    document.getElementById('total-samples').textContent = data.observation.total_samples.toLocaleString();
    document.getElementById('selected-samples').textContent = data.result.n_selected.toLocaleString();
    document.getElementById('selection-ratio').textContent = ((data.result.n_selected / data.observation.total_samples) * 100).toFixed(1) + '%';
    document.getElementById('n-clusters').textContent = data.decision.clustering.n_clusters;
}

async function loadVisualizations(data) {
    try {
        const response = await fetch('/get_visualizations');
        const vizData = await response.json();
        
        if (vizData.error) {
            console.error('Visualization error:', vizData.error);
            return;
        }
        
        // PCA scatter plot
        plotPCAScatter(vizData.pca_plot, data.decision.clustering.n_clusters);
        
        // Cluster distribution
        plotClusterDistribution(vizData.cluster_distribution);
        
        // Distance histogram
        plotDistanceHistogram(vizData.distance_histogram);
        
    } catch (error) {
        console.error('Failed to load visualizations:', error);
    }
}

function plotPCAScatter(data, nClusters) {
    const colors = Plotly.d3.scale.category10().range();
    
    // All points (by cluster)
    const traces = [];
    
    for (let i = 0; i < nClusters; i++) {
        const indices = data.all_points.cluster.map((c, idx) => c === i ? idx : -1).filter(idx => idx !== -1);
        
        traces.push({
            x: indices.map(idx => data.all_points.x[idx]),
            y: indices.map(idx => data.all_points.y[idx]),
            mode: 'markers',
            type: 'scatter',
            name: `Cluster ${i}`,
            marker: {
                size: 4,
                color: colors[i % colors.length],
                opacity: 0.3
            }
        });
    }
    
    // Selected points
    traces.push({
        x: data.selected_points.x,
        y: data.selected_points.y,
        mode: 'markers',
        type: 'scatter',
        name: 'Selected',
        marker: {
            size: 8,
            color: 'red',
            symbol: 'x',
            line: { width: 2, color: 'darkred' }
        }
    });
    
    // Centroids
    traces.push({
        x: data.centroids.x,
        y: data.centroids.y,
        mode: 'markers',
        type: 'scatter',
        name: 'Centroids',
        marker: {
            size: 15,
            color: 'black',
            symbol: 'star',
            line: { width: 2, color: 'yellow' }
        }
    });
    
    const layout = {
        xaxis: { title: 'PC1' },
        yaxis: { title: 'PC2' },
        hovermode: 'closest',
        showlegend: true,
        height: 500
    };
    
    Plotly.newPlot('pca-plot', traces, layout);
}

function plotClusterDistribution(data) {
    const trace1 = {
        x: data.clusters,
        y: data.total,
        type: 'bar',
        name: 'Total',
        marker: { color: '#3498db' }
    };
    
    const trace2 = {
        x: data.clusters,
        y: data.selected,
        type: 'bar',
        name: 'Selected',
        marker: { color: '#e74c3c' }
    };
    
    const layout = {
        xaxis: { title: 'Cluster ID' },
        yaxis: { title: 'Count' },
        barmode: 'group',
        height: 400
    };
    
    Plotly.newPlot('cluster-dist-plot', [trace1, trace2], layout);
}

function plotDistanceHistogram(data) {
    const trace1 = {
        x: data.all_distances,
        type: 'histogram',
        name: 'All Samples',
        marker: { color: '#3498db' },
        opacity: 0.6,
        nbinsx: 50
    };
    
    const trace2 = {
        x: data.selected_distances,
        type: 'histogram',
        name: 'Selected Samples',
        marker: { color: '#e74c3c' },
        opacity: 0.7,
        nbinsx: 50
    };
    
    const layout = {
        xaxis: { title: 'Distance from Centroid' },
        yaxis: { title: 'Frequency' },
        barmode: 'overlay',
        height: 400
    };
    
    Plotly.newPlot('distance-hist-plot', [trace1, trace2], layout);
}

function toggleStage(stageId) {
    const content = document.getElementById(`${stageId}-content`);
    content.classList.toggle('active');
}

async function downloadResults() {
    window.location.href = '/download_results';
}
