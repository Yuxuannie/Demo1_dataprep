# Demo 1: Timing-Aware Data Selection Agent

**Senior Timing Engineer Expertise for Monte Carlo Sample Selection**

## Overview

An AI agent with senior timing engineer domain knowledge (15+ years TSMC/Samsung/Intel experience) that demonstrates intelligent Monte Carlo sample selection for semiconductor library characterization. The agent reduces characterization costs by 50% while maintaining signoff accuracy through strategic uncertainty-based sampling.

### Key Intelligence Demonstrated:
1. **OBSERVE** - Timing domain analysis (process variation, delay correlations, timing corners)
2. **THINK** - Strategic reasoning (active learning principles, algorithm trade-offs)
3. **DECIDE** - Technical optimization (PCA for timing features, GMM for overlapping distributions)
4. **ACT** - Uncertainty sampling (samples far from centroids for critical boundary cases)

## Key Features

- **Senior Timing Engineer Domain Knowledge**: Process variation analysis, timing corner detection, library characterization expertise
- **Active Learning Principles**: Uncertainty-based sampling targeting critical boundary cases
- **Business Impact Awareness**: 50% Monte Carlo cost reduction (10% to 5% coverage)
- **Strategic Reasoning**: Algorithm selection with timing-specific trade-off analysis
- **Clean Architecture**: No special symbols, organized structure, proven LLM foundation

## Quick Start

### Installation

```bash
git clone https://github.com/Yuxuannie/Demo1_dataprep.git
cd Demo1_dataprep
pip install -r requirements.txt
```

### Setup Ollama

```bash
ollama pull qwen2.5:32b-instruct
ollama serve
```

### Configure Environment

Create `.env` file:
```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:32b-instruct
LLM_TEMPERATURE=0.2
LLM_TOP_P=0.9
LLM_NUM_PREDICT=2500
```

### Run Agent

```bash
python chatbot.py
```

### Example Usage

```
Timing Engineer: Select 8% of timing data for Monte Carlo characterization
```

The agent will demonstrate senior timing engineer expertise through detailed reasoning about process variation, active learning principles, and business impact.

## Repository Structure

```
Demo1_dataprep/
├── chatbot.py                      # Main terminal interface
├── timing_prompts.py               # Timing domain expertise prompts
├── agent/
│   ├── timing_data_selection_agent.py   # Core timing-aware agent
│   ├── timing_llm_config.py            # Optimized LLM configuration
│   ├── data_selection_agent.py         # Original agent (reference)
│   └── llm_config.py                   # Original LLM config
├── mock_data/                      # Sample timing data for testing
├── requirements.txt                # Python dependencies
└── setup.py                      # Installation script
```

## Agent Architecture

### Four-Stage Reasoning Process

#### 1. OBSERVE: Timing Domain Analysis
- **Process Variation Detection**: Analyzes correlation between delay and variability
- **Timing Corner Assessment**: Identifies fast/slow path clusters
- **Feature Analysis**: Evaluates timing-critical characteristics with domain knowledge

Example Reasoning:
```
"Analysis of 21,817 timing arc samples reveals strong delay-variability correlation
(r=0.89) between nominal_delay and lib_sigma_delay_late, confirming process variation
scaling. The sigma_by_nominal range 0.02-0.15 indicates mix of stable and
high-variation paths critical for timing signoff."
```

#### 2. THINK: Strategic Reasoning
- **Clustering Justification**: Explains why clustering beats random sampling for timing data
- **Algorithm Selection**: Compares K-means vs GMM for overlapping timing distributions
- **Active Learning Application**: Connects uncertainty sampling to timing robustness

Example Reasoning:
```
"Timing paths form natural clusters (fast/slow, stable/variable). Random sampling
treats all regions equally, risking under-representation of critical corners.
GMM handles overlapping distributions better than K-means for cell type boundaries.
Uncertainty sampling targets samples far from centroids = boundary conditions
where silicon fails."
```

#### 3. DECIDE: Technical Optimization
- **PCA for Timing Features**: Compresses correlated timing characteristics
- **Algorithm Comparison**: Evaluates metrics with timing domain interpretation
- **Parameter Selection**: Chooses optimal cluster count for timing diversity

Example Reasoning:
```
"GMM BIC=1234 optimal for timing data complexity vs K-means inertia.
10 clusters capture timing diversity without over-segmentation.
PCA preserves 93.9% variance while eliminating timing correlation redundancy."
```

#### 4. ACT: Uncertainty-Based Selection
- **Critical Corner Selection**: Samples far from centroids for high model uncertainty
- **Proportional Coverage**: Ensures all timing regions represented
- **Business Impact**: Quantifies cost reduction and signoff benefits

Example Reasoning:
```
"Selected 1,745 samples far from centroids = boundary cases critical for timing signoff.
This uncertainty-based approach enables 5% Monte Carlo coverage vs current 10%,
delivering 50% characterization cost reduction while maintaining accuracy on
critical timing corners."
```

## Timing Domain Expertise

### What the Agent Understands

1. **Process Variation Principles**
   - Delay variability scales with nominal delay
   - Sigma_by_nominal ratios indicate process corners
   - Correlation patterns reveal timing dependencies

2. **Timing Path Characteristics**
   - Fast vs slow path natural clustering
   - Cell type impact on timing distribution
   - Setup/hold timing relationship patterns

3. **Active Learning for Timing**
   - Boundary samples = critical corner cases
   - Model uncertainty = signoff risk areas
   - Representative coverage vs cost trade-offs

4. **Business Context**
   - Monte Carlo characterization costs
   - Signoff accuracy requirements
   - Silicon failure risk management

### Expected Response Quality

| Aspect | Generic Agent | Timing-Aware Agent |
|--------|---------------|-------------------|
| **Correlation Analysis** | "High correlations found" | "r=0.89 delay-variability correlation confirms process variation scaling" |
| **Algorithm Selection** | "Use GMM based on BIC" | "GMM handles timing distribution overlaps better for cell type boundaries" |
| **Sampling Strategy** | "Uncertainty-based sampling" | "Samples far from centroids capture boundary conditions critical for signoff" |
| **Business Impact** | Missing | "Enables 5% vs 10% Monte Carlo = 50% cost reduction while maintaining accuracy" |

## Technical Configuration

### LLM Optimization for Timing Domain

The agent uses optimized LLM parameters for consistent technical reasoning:

```python
TIMING_LLM_PARAMETERS = {
    "temperature": 0.2,   # Lower for consistent technical reasoning
    "top_p": 0.9,         # Focused sampling for domain expertise
    "top_k": 40,          # Reduced for technical precision
    "num_predict": 2500,  # More tokens for detailed explanations
    "repeat_penalty": 1.1 # Avoid repetitive generic phrases
}
```

### Timing-Specific Features

The agent focuses on timing-critical features:
- `nominal_delay`, `lib_sigma_delay_late`: Core delay and variability
- `sigma_by_nominal`: Variability ratio for process corners
- `early_sigma_by_late_sigma`: Skewness indicators
- `cross_signal_enc`: Signal integrity effects
- `mc_err_delay_late`: Monte Carlo error characteristics

## Usage Examples

### Basic Monte Carlo Selection
```
Input: "Select 8% of timing data for library characterization"
Output: Detailed timing domain analysis with uncertainty-based sampling
```

### Custom Percentage
```
Input: "I need 5% representative samples for Monte Carlo validation"
Output: Strategic reasoning about 5% vs other percentages for timing coverage
```

### Algorithm Comparison Request
```
Input: "Compare clustering approaches for timing path selection"
Output: Technical analysis of K-means vs GMM for overlapping timing distributions
```

## Development Notes

### Design Principles
1. **Domain Expertise First**: Senior timing engineer knowledge guides all decisions
2. **Active Learning Focus**: Uncertainty sampling for critical corner cases
3. **Business Impact Awareness**: Connect technical choices to cost reduction
4. **Clean Architecture**: No special symbols, organized structure
5. **Proven Foundation**: Built on working LLM configuration

### Quality Improvements Over Generic ML
- **Specific Number Citations**: Correlation values, exact percentages
- **Timing Domain Terminology**: Process variation, timing corners, signoff accuracy
- **Strategic Trade-off Analysis**: Algorithm selection with domain justification
- **Business Value Articulation**: Cost reduction and risk quantification

## Future Extensions

This foundation enables:
- **Demo 2**: Multi-constraint evaluation agent with timing expertise
- **Adaptive Reasoning**: Data-driven strategy adjustment
- **TSMC Integration**: Real workflow tool connectivity
- **Advanced Active Learning**: Multi-objective uncertainty sampling

## Troubleshooting

### Common Issues

**LLM Connection Errors**:
```bash
# Check Ollama status
ollama serve
curl http://localhost:11434/api/tags
```

**Generic Responses**:
- Lower temperature to 0.1 in `.env`
- Increase `num_predict` to 3000
- Check timing domain prompts are loaded

**Import Errors**:
```bash
pip install -r requirements.txt
python -c "import langchain; print('LangChain available')"
```

## Contributing

This is part of the AIQC (AI Quality Control) project for semiconductor timing analysis. The agent serves as a foundation for intelligent sample selection in library characterization workflows.

For questions or improvements, focus on:
- Timing domain accuracy and expertise depth
- Active learning principle implementation
- Business impact quantification
- Clean, maintainable architecture

## License

Internal TSMC project. Please refer to corporate guidelines for usage and distribution.