# Timing Data Selection Agent

Agentic AI system for intelligent Monte Carlo timing characterization sample selection. Features autonomous exploration, conversational Q&A, and interactive visualization for semiconductor timing analysis.

## Quick Start

1. **Install & Run**
   ```bash
   pip install -r requirements.txt
   ollama serve && ollama pull qwen:32b
   python run_agent.py
   ```

2. **Access Interface**: http://localhost:8501
3. **Upload CSV**: Timing data with delay, sigma, slew, load columns
4. **Chat**: "Select 5% of timing data for ML training"

## Repository Structure

```
demo1_dataprep/
├── agent/
│   ├── timing_data_selection_agent.py  # Complete agentic agent
│   └── __init__.py
├── app_ui.py                           # Streamlit chatbox UI
├── run_agent.py                        # Simple launcher
├── requirements.txt                    # Dependencies
├── mock_data/generate_mock.py          # Test data generator
└── README.md                          # This file
```

## Core Features

### Agentic Intelligence
- **Autonomous Exploration**: Self-guided statistical analysis
- **Strategic Reasoning**: Business impact consideration
- **Self-Validation**: Built-in quality checkpoints
- **Domain Expertise**: Senior timing engineer knowledge

### Conversational Interface
- **Intent Classification**: Distinguishes Q&A from execution
- **Context Preservation**: Answers follow-ups without re-execution
- **Parameter Modification**: "Change to 8%" modifies and re-runs
- **Methodology Explanation**: Technical approach details on demand

### Interactive Visualization
- **Plotly Dashboards**: Zoom, pan, hover for selection analysis
- **Coverage Assessment**: Visual feature space validation
- **Cluster Analysis**: Interactive timing path groupings
- **HTML Export**: Standalone reports for offline review

## Usage Examples

```
"Select 5% of timing data"          → Executes sampling analysis
"Why did you pick those samples?"   → Explains reasoning from context
"Change to 8% instead"             → Modifies parameters and re-runs
"Show visualization"               → Generates interactive dashboard
"How does uncertainty sampling work?" → Explains methodology
```

## Technical Approach

- **PCA Dimensionality Reduction**: Handle high-dimensional timing features
- **Uncertainty Sampling**: Focus on samples far from cluster centroids
- **Safe Allocation**: None-type protection and error recovery
- **Business Optimization**: Balance simulation cost vs model accuracy

## Configuration

Set environment variables for customization:
```bash
OLLAMA_MODEL=qwen:32b           # LLM model
LLM_TEMPERATURE=0.25            # Reasoning consistency
LLM_NUM_PREDICT=2500            # Extended context
```

## Troubleshooting

**Ollama Issues**: `ollama serve && ollama list`
**Import Errors**: `pip install -r requirements.txt`
**Memory Issues**: Sample large datasets before processing
**Slow Response**: Use smaller model `ollama pull qwen:7b`

## Development

The agent combines all functionality in a single class:
- Safe sample allocation with None-type protection
- Intent classification for conversational Q&A
- Interactive Plotly visualization with fallbacks
- Consolidated prompts and LLM configuration

All features are accessible through the `TimingDataSelectionAgent` class without external dependencies.