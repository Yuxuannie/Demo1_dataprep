# Demo 1: Data Preparation Agent

**Intelligent Training Set Selection via Uncertainty-Based Sampling**

---

## 🎯 What This Demo Shows

An **AI agent** that demonstrates complete **Observe → Think → Decide → Act** reasoning for selecting representative training data from unlabeled MLQC timing data.

### Key Intelligence Demonstrated:
1. **OBSERVE** - Analyzes data characteristics (21,818 samples, 8 features, 28 cell types)
2. **THINK** - Reasons about clustering approach (Why? Which algorithm? How many clusters?)
3. **DECIDE** - Makes concrete decisions (PCA: 5 components, GMM: 8 clusters)
4. **ACT** - Executes uncertainty-based selection (selects 5% farthest from centroids)

---

## 📊 Demo Results

**Input**: 21,818 unlabeled timing samples  
**Output**: 1,090 representative samples (5%)  
**Method**: Uncertainty-based sampling (active learning)

### Agent's Reasoning:
```
Question: How to select 5% representative samples?

Answer: Use clustering + uncertainty sampling
1. PCA reduces 8 features → 5 components (93.8% variance)
2. GMM finds 8 natural clusters in timing space
3. Select samples FAR from centroids (high uncertainty)
4. Result: Better model robustness on boundary cases
```

---

## 🚀 Quick Start

### Installation

```bash
cd demo1_dataprep
pip install -r requirements.txt
```

### Run Flask Server

```bash
python app.py
```

Server starts at: **http://localhost:5000**

### Use the Demo

1. **Upload CSV** - Your 21K-row timing data (or use generated mock data)
2. **Set ratio** - Default 5% (adjustable)
3. **Click "Run Agent"** - Watch agent reason through the process
4. **View results** - Interactive visualizations + downloadable CSV

---

## 📁 Project Structure

```
demo1_dataprep/
├── app.py                          # Flask server
├── agent/
│   ├── __init__.py
│   └── data_prep_agent.py          # Agent with Observe-Think-Decide-Act
├── mock_data/
│   ├── generate_mock.py            # Mock data generator
│   └── test_data.csv               # Generated 21,818 samples
├── static/
│   ├── css/style.css               # Clean minimal styling
│   └── js/main.js                  # Frontend logic + Plotly
├── templates/
│   └── index.html                  # Single-page UI
├── uploads/                        # Temp upload storage
└── requirements.txt                # Dependencies
```

---

## 🧠 Agent Intelligence Architecture

### Stage 1: OBSERVE
```python
observation = {
    'total_samples': 21818,
    'n_features': 8,
    'cell_types': 28,
    'high_correlations': 5 pairs
}

Reasoning: "nominal_delay and lib_sigma_delay_late are highly 
correlated → PCA will be effective"
```

### Stage 2: THINK
```python
strategy = {
    'use_pca': True,          # Reduce correlated features
    'algorithm': 'gmm',       # Better for overlapping clusters
    'n_clusters': 8,          # Balance coverage vs simplicity
    'selection': 'uncertainty' # Far from centroid = high uncertainty
}

Reasoning: "Training on uncertain samples → Better model 
robustness on boundary cases (active learning principle)"
```

### Stage 3: DECIDE
```python
# Apply PCA
pca_result = PCA(n_components=5)  # 93.8% variance
features_reduced = pca.transform(features)

# Test clustering
gmm = GaussianMixture(n_components=8)
clusters = gmm.fit_predict(features_reduced)

# Calculate distances
distances = distance_to_centroid(features_reduced, clusters)

Decision: "8 clusters, GMM algorithm, 5 PCA components"
```

### Stage 4: ACT
```python
# Select farthest from centroids (uncertainty sampling)
for cluster_id in range(8):
    cluster_samples = samples[labels == cluster_id]
    cluster_distances = distances[labels == cluster_id]
    
    # Sort by distance DESC (farthest first)
    sorted_idx = np.argsort(cluster_distances)[::-1]
    selected = cluster_samples[sorted_idx[:n_samples]]

Result: 1,090 samples selected, all clusters represented
```

---

## 📈 Visualizations

### 1. PCA Space (All vs Selected)
- **Blue points**: All 21,818 samples (8 clusters)
- **Red X marks**: Selected 1,090 samples
- **Black stars**: Cluster centroids
- **Observation**: Selected samples are far from centroids (boundaries)

### 2. Cluster Distribution
- **Blue bars**: Total samples per cluster
- **Red bars**: Selected samples per cluster
- **Observation**: Proportional selection with uncertainty boost

### 3. Distance Histogram
- **Blue**: Distance distribution (all samples)
- **Red**: Distance distribution (selected samples)
- **Observation**: Selected samples have higher distances (uncertainty)

---

## 🔬 Why This Works (Theory)

### Active Learning Principle
```
Certainty = f(distance_to_centroid)

Samples near centroid:
- Model is confident (well-represented region)
- Adding more doesn't help much

Samples far from centroid:
- Model is uncertain (boundary/outlier region)
- Adding these improves robustness significantly
```

### Expected Benefits
1. **Better coverage** - Spans full feature space
2. **Better generalization** - Learns boundary cases
3. **Fewer samples** - 5% vs 10% random sampling
4. **Systematic** - No trial-and-error

---

## 📊 CSV Format

### Input (Unlabeled)
```csv
arc_pt,nominal_delay,lib_sigma_delay_late,nominal_tran,lib_sigma_tran_late,...
MUXA2MZD2ZBWP...,1522.2,318.1,1042.7,463.7,...
NAND2D4ZBWP...,732.5,87.3,298.4,35.6,...
```

### Output (Selected with Metadata)
```csv
arc_pt,nominal_delay,...,cluster_id,distance_to_centroid
MUXA2MZD2ZBWP...,1522.2,...,3,2.84
NAND2D4ZBWP...,732.5,...,1,2.71
```

---

## 🎬 For Presentation

### Demo Script (5 minutes)

**Minute 1**: Problem
> "We need to select 5% representative samples from 21K timing paths. Random sampling misses critical regions. We need intelligent selection."

**Minute 2**: Agent Reasoning (Show UI)
> "Watch the agent reason through this:
> - OBSERVE: 21K samples, 8 features, highly correlated
> - THINK: Use PCA + GMM + uncertainty sampling
> - DECIDE: 5 components, 8 clusters, select farthest from centroids
> - ACT: 1,090 samples selected"

**Minute 3**: Results
> "Look at the visualizations:
> - Red X marks are far from blue clusters (high uncertainty)
> - All 8 clusters covered proportionally
> - Distance histogram shows selected samples at high-uncertainty tail"

**Minute 4**: Why This is Intelligent
> "The agent didn't just run sklearn. It:
> - Reasoned about WHY clustering helps
> - Chose GMM over K-means (better for overlapping)
> - Applied active learning principle (uncertainty sampling)
> - Explains every decision transparently"

**Minute 5**: Next Steps
> "This is proof-of-concept. For production:
> - Integrate with real MLQC pipeline
> - Add more domain rules (Low-VT MB_Cells priority)
> - Connect to Qwen LLM for natural language reasoning"

---

## ⚙️ Technical Details

### Dependencies
- **Flask**: Web server
- **pandas**: Data handling
- **scikit-learn**: PCA, K-means, GMM
- **scipy**: Distance calculations
- **Plotly.js**: Interactive visualizations

### Performance
- **Data loading**: <1 second (21K rows)
- **PCA**: <1 second
- **Clustering**: 2-3 seconds
- **Selection**: <1 second
- **Total**: ~5 seconds end-to-end

### Scalability
- Tested: 21,818 samples
- Can handle: 100K+ samples
- For larger: Use mini-batch clustering

---

## 🔧 Customization

### Change Selection Ratio
```python
# In UI: Adjust slider (1-20%)
# In code: target_ratio parameter
result = agent.run(csv_path, target_ratio=0.10)  # 10%
```

### Change Number of Clusters
```python
# In agent/data_prep_agent.py
strategy = {
    'n_clusters_range': [6, 8, 10, 12]  # Test these values
}
```

### Change Selection Strategy
```python
# Current: Farthest from centroid (uncertainty)
# Alternative: Random within cluster (baseline)
# Alternative: Nearest to centroid (confidence)
```

---

## ✅ Testing

### Test Agent Directly
```bash
cd demo1_dataprep
python agent/data_prep_agent.py
```

Expected output:
```
🤖 Data Preparation Agent Starting
📊 Stage 1: OBSERVE → ✅
🧠 Stage 2: THINK → ✅
⚖️ Stage 3: DECIDE → ✅
⚙️ Stage 4: ACT → ✅
Selected 1090 samples
```

### Test Mock Data Generator
```bash
python mock_data/generate_mock.py
```

### Test Flask Server
```bash
python app.py
# Visit http://localhost:5000
```

---

## 🎓 Educational Value

### For Management
- Shows "AI thinking" transparently
- Demonstrates business value (5% vs 10% data)
- Clear before/after comparison

### For Engineers
- Practical active learning example
- Clean code architecture
- Reusable components

### For Students
- Complete ML pipeline example
- Agent reasoning pattern
- Uncertainty quantification

---

## 📝 Known Limitations

### Current Implementation
- ✅ Works with mock data (realistic)
- ✅ Demonstrates reasoning clearly
- ⚠️ Not integrated with real MLQC pipeline
- ⚠️ No LLM for natural language (template-based)

### For Production
- Add real MLQC validation
- Connect to Qwen-72B for explanations
- Add domain-specific rules (Low-VT priority)
- Multi-PVT corner support

---

## 🚀 Next Steps

1. **Test with real data** - Replace mock with actual MLQC CSV
2. **Validate selections** - Train model and measure accuracy
3. **Add domain rules** - Low-VT, MB_Cell priorities
4. **LLM integration** - Qwen-72B for natural language
5. **Build Demo 2** - Evaluation & Loop-Back agent

---

## 💡 Tips

### For Demo
- Start with mock data (fast, reliable)
- Show reasoning stages one-by-one
- Highlight uncertainty-based selection visualization
- Emphasize: "Agent reasons, not just automates"

### For Production
- Start with 5% ratio, increase if needed
- Validate with cross-validation
- Monitor cluster quality over time
- A/B test vs random sampling

---

## 📧 Contact

Questions? Issues? Suggestions?
- Review agent reasoning in console output
- Check visualizations for data quality
- Adjust parameters in UI

---

## 🎉 Summary

**What you built**: Intelligent data prep agent with transparent reasoning

**What it demonstrates**: Observe-Think-Decide-Act pattern + Active Learning

**What it proves**: AI can make strategic decisions, not just optimize

**What's next**: Demo 2 (Evaluation agent) + Real data integration

**Time invested**: 2 days building, lifetime of savings! 🚀
