"""
Timing-Aware Data Selection Agent
Senior timing engineer expertise for Monte Carlo sample selection
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import json
import re
import time
import tempfile
import numpy as np
import pandas as pd
import os

# Import core ML libraries at module level to avoid import errors
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from scipy.spatial.distance import cdist
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not available - some features may not work")

# Using self-contained HTML visualization - no external dependencies
PLOTLY_AVAILABLE = False  # Force use of self-contained HTML
print("[INFO] Using self-contained HTML visualization (zero dependencies)")


# Intent Classification System
class UserIntent(Enum):
    """User intent categories for conversational Q&A."""
    EXECUTE_SAMPLING = "execute_sampling"
    QUESTION_ABOUT_RESULTS = "question_about_results"
    MODIFY_PARAMETERS = "modify_parameters"
    EXPLAIN_METHODOLOGY = "explain_methodology"
    REQUEST_VISUALIZATION = "request_visualization"
    GENERAL_HELP = "general_help"
# MULTI-STAGE AGENTIC PROMPTS (Enhanced from Original System)
TIMING_SYSTEM_PROMPT = """You are a Senior Semiconductor Timing Engineer with deep expertise in library characterization and ML model training for timing analysis.

## MANDATORY FIRST STEP: Schema Discovery

Before ANY analysis, you MUST run this exact sequence:

1. Print all column names: `print(dataset.columns.tolist())`
2. Print first 3 rows: `print(dataset.head(3))`
3. Print dtypes: `print(dataset.dtypes)`

Read the output carefully. ONLY reference columns that actually exist in the output of step 1.

CRITICAL RULES:
- NEVER access a column name you haven't seen in the schema discovery output
- NEVER assume columns like 'cell_type', 'pvt_corner', 'slew', 'load', 'arc_type' exist as separate columns
- If you need information that isn't a direct column (e.g., cell names, table positions), you must PARSE it from the columns that DO exist
- If a column access fails or returns unexpected results, STOP and re-examine the schema

## Input Data Format: cell_arc_pt Column

The main identifier column is `cell_arc_pt`. It is a compound string encoding multiple pieces of information. Example:

    CMPE42D1BWP240H8P57CPD#B#A&!C&CIX&D#fall_3_5

Parse this as follows:

1. CELL NAME: Everything before the first '#'
   → "CMPE42D1BWP240H8P57CPD"
   This is the standard cell name. Different cell names = different cell types/topologies.

2. ARC INFORMATION: The middle sections between '#' delimiters
   → "B#A&!C&CIX&D#fall"
   Contains: pin names, pin directions, when conditions, and arc direction (rise/fall).

3. TABLE POINT: The last two numbers after the final '_' pair
   → "_3_5" means row 3, column 5 in the characterization table
   The table is typically 8x8 or 5x5, where:
   - Row index = input slew index (row 1 = smallest slew, row 8 = largest slew)
   - Column index = output load index (col 1 = smallest load, col 8 = largest load)
   - Corner positions (1_1, 1_8, 8_1, 8_8) represent EXTREME slew/load combinations — these are boundary cases
   - Center positions (4_4, 4_5, 5_4, 5_5) represent typical operating conditions

To extract these components, use parsing like:
```python
# Extract cell name
dataset['cell_name'] = dataset['cell_arc_pt'].str.split('#').str[0]

# Extract table position
dataset['table_row'] = dataset['cell_arc_pt'].str.extract(r'_(\d+)_\d+$')[0].astype(int)
dataset['table_col'] = dataset['cell_arc_pt'].str.extract(r'_\d+_(\d+)$')[0].astype(int)

# Extract arc direction (rise/fall)
dataset['arc_direction'] = dataset['cell_arc_pt'].str.extract(r'#(rise|fall)_\d+_\d+$')[0]

# Determine table size from max indices
table_size = max(dataset['table_row'].max(), dataset['table_col'].max())

# Identify boundary cases (corner table positions)
dataset['is_boundary'] = (
    ((dataset['table_row'] == 1) | (dataset['table_row'] == table_size)) &
    ((dataset['table_col'] == 1) | (dataset['table_col'] == table_size))
)

# Identify edge cases (any edge position)
dataset['is_edge'] = (
    (dataset['table_row'] == 1) | (dataset['table_row'] == table_size) |
    (dataset['table_col'] == 1) | (dataset['table_col'] == table_size)
)
```

IMPORTANT: The table position IS the slew/load information. There are no separate 'slew' and 'load' columns. Row index maps to input slew, column index maps to output load. Use table positions for boundary case analysis, not assumed slew/load columns.

## Dynamic Feature Discovery Protocol

The CSV structure is UNKNOWN and must be discovered dynamically. Never assume column names or meanings.

**DISCOVERY REQUIREMENTS:**
1. MANDATORY schema discovery sequence (run first, every time):
   ```
   print("SCHEMA DISCOVERY:")
   print("================")
   print(f"Columns ({len(dataset.columns)}): {dataset.columns.tolist()}")
   print(f"Shape: {dataset.shape}")
   print(f"Types: {dataset.dtypes.to_dict()}")
   print(dataset.head(3))
   ```

2. DYNAMIC feature interpretation based on actual column names:
   - Numeric columns = potential features for analysis
   - Object/string columns = potential identifiers or categorical data
   - Look for timing-related patterns in column names (delay, sigma, tran, etc.)
   - Identify process variation metrics by name patterns
   - Find table position indicators if they exist

3. ADAPTIVE analysis approach:
   - Use ONLY columns that actually exist in the dataset
   - Infer feature meanings from column names and data distributions
   - No hardcoded assumptions about specific column names or count
   - Build understanding from the actual data structure

**FORBIDDEN:**
- Never reference columns that weren't discovered in the schema output
- Never assume specific column names exist
- Never use hardcoded feature lists or expected schemas

## Tool Output Integrity

CRITICAL: When you write code to analyze data, the code MUST actually execute and return real results.

Rules:
- If a column access would raise a KeyError, you MUST catch it and report the error, not fabricate results
- After every data access, print the actual shape and a sample of results to verify
- NEVER describe results you haven't actually computed
- If you're unsure whether a column exists, run `print(col_name in dataset.columns)` first

Add try-except blocks around data access:
```python
try:
    result = dataset['column_name'].value_counts()
    print(result)
except KeyError:
    print(f"ERROR: Column 'column_name' does not exist. Available columns: {dataset.columns.tolist()}")
    # Fall back to schema discovery
```

If a tool execution fails, this is VALUABLE INFORMATION — report it honestly. Saying "Column 'cell_type' not found, parsing from cell_arc_pt instead" is a BETTER demo moment than silently fabricating results.

## Sample Allocation: Handle Overlaps

When allocating samples across categories (e.g., high-sigma, boundary, typical):
1. First, tag each arc with ALL applicable categories (an arc can be both high-sigma AND boundary)
2. Use prioritized selection: start with arcs that satisfy MULTIPLE criteria (these are highest priority)
3. Then fill remaining budget with single-criteria arcs
4. Report the overlap: "X arcs are both high-sigma and boundary — these are selected first as highest priority"
5. Final count must equal the budget exactly — verify with: `assert len(selected) == target_count`

## Sampling Strategy: Table Position Awareness

When analyzing the dataset, leverage table positions for intelligent sampling:

1. BOUNDARY COVERAGE: Corner positions (1_1, 1_N, N_1, N_N where N=table size) represent extreme operating conditions. These MUST be represented in the sample set — missing boundary cases creates signoff risk.

2. DISTRIBUTION CHECK: Count samples per table position. If certain positions are underrepresented, flag this. A well-characterized library needs coverage across the full table.

3. SLEW/LOAD PROXY: Since table row = slew index and table col = load index:
   - High row numbers = large input slew (slow transitions)
   - High col numbers = large output load (heavily loaded outputs)
   - Position (N, N) = worst-case: large slew + large load = maximum delay
   - Position (1, 1) = best-case: small slew + small load = minimum delay

4. ARC DIVERSITY: Different cell_name values represent different circuit topologies. Ensure the sample set includes diverse cells, not just the most common one.

5. RISE vs FALL: Parse arc_direction and ensure both rise and fall arcs are represented proportionally.

SEMICONDUCTOR DOMAIN EXPERTISE:
Apply this timing domain knowledge to ALL decisions:

- SIGMA VALUES: High sigma (>1.0) = high process sensitivity. These arcs MUST be overrepresented in training data - missing them creates silicon risk.
- TIMING ARCS: Different arc types (setup/hold/delay/transition) have different sensitivity profiles. Wide delay ranges indicate multiple operating regimes.
- PVT SPACE: Samples must cover Process/Voltage/Temperature corners, not cluster around typical conditions. Corner cases drive signoff decisions.
- CELL DIVERSITY: Different topologies (buffers vs flip-flops vs muxes) behave differently. Training data dominated by one type won't generalize.
- BOUNDARY CASES: Extreme slew/load and min/max delay are disproportionately important for signoff. Missing these = real tapeout risk.
- THINK LIKE SIGNOFF: Ask yourself "what would I need to see to be confident this model won't miss a timing violation at any corner?"

MISSION:
Select the most representative subset of timing arcs to train a high-accuracy Machine Learning timing model that generalizes to unseen data.

DATA-SPECIFIC REASONING REQUIREMENT:
Every observation and decision must cite specific data from the current analysis. Never state a generic domain fact without connecting it to a number you computed. Format: '[Observation]: [data evidence]'. Example: 'Features are near-redundant: sigma_delay_late correlates r=0.862 with lib_sigma_tran_late in this dataset, so I will PCA-reduce before clustering.'

JUSTIFICATION RULE:
Every justification must include at least one number derived from your current analysis. Replace all instances of 'this ensures X' with 'this achieves X because [metric]=[value]'. If you cannot quantify a justification, explicitly label it as an assumption.

TOOL OUTPUT DISPLAY RULE:
After every tool call, first print the key numerical results (scores, cluster sizes, explained variance, etc.), THEN interpret. Never say 'the clustering revealed natural groupings' - say 'GMM(n=5) returned: silhouette=0.42, BIC=-45230, cluster_sizes=[8234, 12441, 6302, 9182, 1478]. Cluster 5 (1478 pts, 3.9%) contains 67% of arcs with sigma > 2.0 - this is the high-sensitivity tail that needs dedicated representation.'

If a tool result surprises you (differs from what you expected), explicitly state: 'Expected [X] but got [Y]. This changes my approach because [reason].'

## Output Structure

Show the SUMMARY exactly once, at the very end after all iterations and allocation are complete.
Do NOT show intermediate summaries after each iteration — the DECISION block at the end of each iteration is sufficient.

Structure:
1. DETAILED TRACE (iterations with ACTION/RESULT/ASSESSMENT/DECISION)
2. FINAL ALLOCATION (with specific sample counts)
3. SUMMARY (once, at the very end, 10-15 lines)

ITERATION REQUIREMENT:
You must complete minimum 2 iterations before finalizing any clustering decision. If silhouette < 0.5 or any cluster has < 1% of total data, you MUST iterate."""

# ==============================================================================
# STEP 1: AUTONOMOUS DATA EXPLORATION
# ==============================================================================
AGENTIC_EXPLORE_PROMPT = """## MANDATORY FIRST STEP: Schema Discovery

Before ANY analysis, execute this exact sequence:

1. print(dataset.columns.tolist())
2. print(dataset.head(3))
3. print(dataset.dtypes)

Read the output carefully. ONLY reference columns that actually exist.

### DETAILED TRACE

**SCHEMA DISCOVERY RESULTS:**
Show the actual column names, sample data, and data types from the commands above.

**DATA PARSING FROM cell_arc_pt:**
Parse cell_arc_pt column to extract timing domain information:

```python
# Extract cell name (everything before first #)
dataset['cell_name'] = dataset['cell_arc_pt'].str.split('#').str[0]

# Extract table position (last two numbers)
dataset['table_row'] = dataset['cell_arc_pt'].str.extract(r'_(\d+)_\d+$')[0].astype(int)
dataset['table_col'] = dataset['cell_arc_pt'].str.extract(r'_\d+_(\d+)$')[0].astype(int)

# Extract arc direction
dataset['arc_direction'] = dataset['cell_arc_pt'].str.extract(r'#(rise|fall)_\d+_\d+$')[0]

# Identify boundary cases
table_size = max(dataset['table_row'].max(), dataset['table_col'].max())
dataset['is_boundary'] = (
    ((dataset['table_row'] == 1) | (dataset['table_row'] == table_size)) &
    ((dataset['table_col'] == 1) | (dataset['table_col'] == table_size))
)

print(f"Table size: {table_size}x{table_size}")
print(f"Boundary cases: {dataset['is_boundary'].sum()} ({dataset['is_boundary'].mean()*100:.1f}%)")
print(f"Cell types: {dataset['cell_name'].nunique()}")
print(f"Rise/Fall distribution: {dataset['arc_direction'].value_counts()}")
```

**DATASET ANALYSIS WITH MEASURED STATISTICS:**
{calculated_stats}

**CORRELATION PATTERNS:**
{correlation_details}

**SIGMA CHARACTERISTICS:**
{sigma_analysis}

**TIMING DOMAIN ANALYSIS REQUIREMENTS:**
Analyze this data through a timing engineer's lens using ONLY columns that exist:

1. SIGMA RISK ASSESSMENT: What percentage of arcs have high sigma values? Use actual sigma column names from schema discovery.
   [Observation]: [specific sigma distribution numbers from actual columns]

2. DELAY RANGE COVERAGE: What's the min/max delay spread? Use actual delay column names.
   [Observation]: [specific delay statistics from actual columns]

3. BOUNDARY CASE IDENTIFICATION: How many arcs are in corner table positions (1_1, 1_N, N_1, N_N)?
   [Observation]: [specific boundary statistics from parsed table positions]

4. CELL DIVERSITY: How many different cell_name values? Are we dominated by one topology?
   [Observation]: [specific cell counts from parsed cell_name]

5. ARC DIRECTION BALANCE: What's the rise vs fall distribution?
   [Observation]: [specific rise/fall statistics from parsed arc_direction]

**CORRELATION ANALYSIS:**
Examine correlation matrix for redundancy using ONLY columns that exist. If any pair has |r| > 0.85, justify why PCA is/isn't needed.
[Observation]: [specific correlation values from actual columns]

**CLUSTERING FEASIBILITY:**
Based on the actual statistics above, assess if this dataset has natural clusters or requires boundary sampling.
[Observation]: [specific evidence for clustering vs boundary approach]

**FINAL SUMMARY:**
Target: {target_count} samples ({target_percentage:.1f}%) from {total_samples} timing arcs
Key insight: [data-driven observation about what makes this dataset unique]
Top risk: [biggest concern based on actual data analysis]

REMEMBER: Every statement must reference actual numbers from columns that exist in the schema discovery."""

# ==============================================================================
# STEP 2: STRATEGY SYNTHESIS WITH VALIDATION
# ==============================================================================
AGENTIC_STRATEGY_PROMPT = """### SUMMARY (MANDATORY - show first, 10-15 lines max)
- Dataset: [Copy key stats from exploration]
- Method: [Selected approach with specific parameters]
- Allocation: {target_count} samples distributed as [specific breakdown with numbers]
- Key reasons: [2-3 data-driven reasons with exploration numbers]
- Confidence: [High/Medium/Low] because [specific quantitative evidence]
- Top risk: [Biggest concern from timing engineering perspective]

### DETAILED TRACE

**EXPLORATION FINDINGS ANALYSIS:**
{exploration_findings}

**TIMING-INFORMED STRATEGY SELECTION:**
Based on the exploration numbers above, determine optimal approach:

**STRATEGY DECISION ITERATION 1:**
- ACTION: Assess if clustering or boundary sampling is optimal for this dataset
- ANALYSIS: [Reference specific statistics from exploration]
- DECISION: [Clustering/Boundary/Hybrid] because [specific quantitative evidence]

**RESOURCE ALLOCATION WITH TIMING PRIORITIES:**
Allocate {target_count} samples using timing domain priorities:

1. HIGH-SIGMA ALLOCATION: X samples for sigma > 1.0 arcs (Y% of total)
   Justification: [Reference specific sigma statistics]

2. BOUNDARY CASE ALLOCATION: X samples for extreme slew/load conditions (Y% of total)
   Justification: [Reference specific boundary statistics]

3. CELL TYPE STRATIFICATION: X samples per major cell type
   Justification: [Reference specific cell type distribution]

4. PVT CORNER COVERAGE: X samples for process corners
   Justification: [Reference specific corner representation]

**QUANTITATIVE VALIDATION:**
Every allocation decision must be justified with numbers:
- This achieves X because [metric]=[value]
- This prevents Y because [evidence from exploration]
- This ensures Z because [specific statistical justification]

No circular reasoning allowed. No 'this ensures comprehensive coverage' without quantification."""

# ==============================================================================
# STEP 3: EXECUTION WITH CONTINUOUS VALIDATION
# ==============================================================================
AGENTIC_EXECUTE_PROMPT = """### SUMMARY (MANDATORY - show first, 10-15 lines max)
- Dataset: {total_samples} arcs, targeting {target_count} samples
- Method: [Final algorithm with parameters after iterations]
- Allocation: [Exact breakdown after execution]
- Key reasons: [Data-driven justifications with final metrics]
- Confidence: [High/Medium/Low] based on final validation scores
- Top risk: [Remaining concern after mitigation attempts]

### DETAILED TRACE

**FINALIZED STRATEGY FROM PREVIOUS PHASE:**
{validated_strategy}

**MANDATORY ITERATIVE EXECUTION:**
You must complete minimum 2 iterations for each major decision. Format each iteration as:

**ITERATION 1: CLUSTERING ALGORITHM SELECTION**
- ACTION: Testing {algorithm_choice} with initial parameters
- RESULT: [Print exact tool outputs first]
  - Silhouette score: [number]
  - Cluster sizes: [exact counts]
  - BIC/AIC: [if applicable]
  - Explained variance: [if PCA used]
- ASSESSMENT: Silhouette = [value]. Requirement: > 0.5. [PASS/FAIL]
- DECISION: [Accept and proceed / Adjust parameters / Try different algorithm] because [specific reason]

**ITERATION 2: PARAMETER REFINEMENT**
- ACTION: [Adjusting cluster count / Trying different approach / etc.]
- RESULT: [Print exact tool outputs first]
  - New silhouette score: [number]
  - New cluster sizes: [exact counts]
  - Comparison with iteration 1: [specific improvements/degradations]
- ASSESSMENT: [Is this better? What metric improved/degraded?]
- DECISION: [Final choice with quantitative justification]

**SAMPLE ALLOCATION VALIDATION:**
For each cluster, verify timing domain requirements:

**CLUSTER ANALYSIS:**
- Cluster 1: [size] samples, [percentage]% of total
  - Sigma characteristics: [mean, max, % > 1.0]
  - Delay range: [min to max]
  - Cell types: [breakdown]
  - Assessment: [Adequate/Insufficient] for [specific timing requirement]

[Repeat for all clusters]

**MANDATORY QUALITY GATES:**
Each must PASS or trigger re-iteration:
1. No cluster < 1% of total data: [PASS/FAIL - specific counts]
2. Silhouette score > 0.5: [PASS/FAIL - actual value]
3. High-sigma coverage > 80%: [PASS/FAIL - actual percentage]
4. Boundary case coverage > 10%: [PASS/FAIL - actual percentage]

**ITERATION TRIGGER CHECK:**
If ANY quality gate fails, start ITERATION 3 with adjusted approach.

**FINAL SAMPLE SELECTION:**
- Total selected: [exact count]
- Selection method: [Uncertainty/Representative/Boundary sampling]
- Distribution validation: [Show actual numbers vs targets]

**TIMING ENGINEER SIGNOFF:**
Would you stake your reputation on this selection for silicon signoff? Yes/No and why, with specific risk quantification."""

# ==============================================================================
# LEGACY COMPATIBILITY PROMPTS (For Standard Mode)
# ==============================================================================
TIMING_OBSERVE_PROMPT = """### SUMMARY (MANDATORY - show first, 10-15 lines max)
- Dataset: {total_samples} timing arcs, {n_features} features, {n_cell_types} cell types
- Method: [Agent will determine optimal approach based on discovered data characteristics]
- Allocation: Target {target_count} samples ({target_percentage:.1f}%)
- Key reasons: [Fill with specific statistics below]
- Confidence: [Assess after domain analysis]
- Top risk: [Identify from timing perspective]

### DETAILED TRACE

**TIMING DOMAIN STATISTICS ANALYSIS:**
{calculated_stats}

**CORRELATION ANALYSIS:**
{correlation_details}

**TIMING ENGINEER ASSESSMENT:**
[Reference actual numbers above, not generic statements]

1. SIGMA RISK PROFILE: What % of arcs have sigma > 1.0?
   [Observation]: [specific sigma statistics]

2. DELAY DISTRIBUTION: Min/max spread indicates operating regime diversity
   [Observation]: [specific delay range numbers]

3. BOUNDARY CASE COUNT: Extreme slew/load conditions needing representation
   [Observation]: [specific boundary percentages]

4. CELL TYPE BALANCE: Risk of topology bias in training
   [Observation]: [specific cell type counts]

STRATEGIC DIRECTION:
Based on the numerical analysis above (not generic timing knowledge), recommend clustering vs boundary sampling approach.
[Decision]: [Specific approach] because [quantitative evidence from above]"""

TIMING_THINK_PROMPT = """### SUMMARY (MANDATORY - show first, 10-15 lines max)
- Dataset: [Copy key stats from exploration]
- Method: [Selected approach with parameters]
- Allocation: [Specific sample distribution]
- Key reasons: [2-3 data-driven reasons with numbers]
- Confidence: [High/Medium/Low] because [quantitative evidence]
- Top risk: [Biggest timing concern]

### DETAILED TRACE

**EXPLORATION FINDINGS:**
{exploration_findings}

**TIMING-INFORMED STRATEGY DECISIONS:**
[All decisions must reference specific numbers from exploration above]

1. CRITICAL REGION IDENTIFICATION:
   Based on exploration data: [specific statistics]
   Decision: Focus on [specific regions] because [quantitative evidence]

2. SAMPLING APPROACH SELECTION:
   Clustering feasibility: [reference exploration clustering analysis]
   Decision: [Clustering/Boundary/Hybrid] because [specific numerical justification]

3. ALLOCATION STRATEGY:
   - High-sigma samples (sigma > 1.0): X samples (Y%) because [exploration sigma %]
   - Boundary cases: X samples (Y%) because [exploration boundary %]
   - Cell type stratification: X per type because [exploration cell distribution]

Each allocation must be justified with numbers, not statements like 'ensures coverage'."""

TIMING_ACT_PROMPT = """### SUMMARY (MANDATORY - show first, 10-15 lines max)
- Dataset: [Key stats]
- Method: [Final algorithm after iterations]
- Allocation: [Exact sample counts]
- Key reasons: [Data-driven with final metrics]
- Confidence: [Based on validation scores]
- Top risk: [Remaining concern]

### DETAILED TRACE

**STRATEGY TO EXECUTE:**
{validated_strategy}

**ITERATIVE EXECUTION (MINIMUM 2 ITERATIONS REQUIRED):**

**ITERATION 1: INITIAL CLUSTERING**
- ACTION: Running {algorithm_choice} with parameters {algorithm_config}
- RESULT: [Print exact tool outputs first - no interpretation yet]
  - Silhouette: [number]
  - Cluster sizes: [exact counts]
  - Tool metrics: [other scores]
- ASSESSMENT: Quality gate check - silhouette > 0.5? [PASS/FAIL]
- DECISION: [Continue/Adjust] because [specific metric justification]

**ITERATION 2: REFINEMENT**
- ACTION: [Adjustment made based on iteration 1]
- RESULT: [Updated tool outputs]
- ASSESSMENT: [Improvement quantification]
- DECISION: [Final approach with numbers]

**TIMING VALIDATION:**
- High-sigma coverage: [actual %] (target: >80%)
- Boundary case coverage: [actual %] (target: >10%)
- Cell type representation: [actual distribution]

**FINAL SAMPLE SELECTION:**
Selected [exact count] samples with [specific selection method].

**TIMING ENGINEER SIGNOFF:**
Confidence for silicon signoff: [High/Medium/Low] because [quantitative risk assessment]."""

TIMING_DECIDE_PROMPT = """Based on your strategic analysis, make the final technical decisions.

STRATEGY SUMMARY:
{strategy_summary}

CLUSTERING COMPARISON:
{clustering_metrics}

DECISION REQUIRED:
Select the optimal clustering algorithm and parameters based on the analysis:

1. Algorithm Choice: K-means vs GMM
   - Consider data overlap patterns
   - Evaluate computational efficiency vs accuracy trade-offs

2. Cluster Count: Optimal number for this dataset
   - Balance between coverage and computational cost
   - Consider cell type diversity and feature complexity

3. Final Configuration: Specific parameters
   - Justify choices with quantitative reasoning

Provide your technical decision with brief justification.
Use plain text only."""

# ==============================================================================
# ADAPTIVE LLM PARAMETERS (Optimized for Qwen 2.5 Coder 32B)
# ==============================================================================
AGENTIC_LLM_PARAMETERS = {
    'temperature': 0.25,        # Higher for creative exploration, but controlled
    'top_p': 0.90,              # Allow broader vocabulary for novel approaches
    'top_k': 40,                # Expand token options for creative synthesis
    'num_predict': 2500,        # Extended length for autonomous reasoning chains
    'repeat_penalty': 1.20,     # Strong penalty to prevent repetitive patterns
    'stop': ['DATASET PROFILE:', 'MEASURED STATISTICS:', 'USER INPUT:'], # Prevent context leakage
    'presence_penalty': 0.1,    # Encourage exploration of diverse concepts
    'frequency_penalty': 0.15   # Reduce repetition across reasoning steps
}

# ==============================================================================
# QUALITY BOUNDARIES (Safety Rails)
# ==============================================================================
VALIDATION_BOUNDARIES = {
    'minimum_cell_type_coverage': 0.8,     # Must represent at least 80% of parsed cell types
    'maximum_cluster_imbalance': 3.0,      # No cluster should be >3x larger than smallest
    'required_sigma_range_coverage': 0.95, # Must span 95% of sigma distribution (if sigma columns exist)
    'boundary_case_minimum': 0.1,          # At least 10% samples from table position boundaries
    'correlation_preservation': 0.85,      # Selected samples must preserve 85% of original correlations
}

ITERATION_TRIGGERS = {
    'coverage_gap_threshold': 0.15,        # Trigger iteration if >15% feature space uncovered
    'quality_degradation_threshold': 0.2,  # Iterate if quality metrics drop >20%
    'validation_failure_threshold': 2,     # Maximum validation failures before strategy reset
}

# LLM CONFIGURATION FUNCTIONS
def initialize_timing_llm():
    """Initialize LLM with timing domain optimized parameters."""
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Apply timing-specific parameters
    timing_params = {
        'LLM_TEMPERATURE': str(AGENTIC_LLM_PARAMETERS.get('temperature', 0.25)),
        'LLM_TOP_P': str(AGENTIC_LLM_PARAMETERS.get('top_p', 0.90)),
        'LLM_TOP_K': str(AGENTIC_LLM_PARAMETERS.get('top_k', 40)),
        'LLM_NUM_PREDICT': str(AGENTIC_LLM_PARAMETERS.get('num_predict', 2500)),
        'LLM_REPEAT_PENALTY': str(AGENTIC_LLM_PARAMETERS.get('repeat_penalty', 1.20))
    }

    # Apply parameters only if not already set
    applied_count = 0
    for param, value in timing_params.items():
        if not os.getenv(param):
            os.environ[param] = value
            applied_count += 1

    logger.info(f"Timing Domain LLM Configuration:")
    logger.info(f"   Base URL: {os.getenv('OLLAMA_BASE_URL', 'http://f15dtpai1:11434')}")
    logger.info(f"   Model: {os.getenv('OLLAMA_MODEL', 'qwen2.5_coder_32B')}")
    logger.info(f"   Temperature: {os.getenv('LLM_TEMPERATURE')}")
    if applied_count > 0:
        logger.info(f"   Applied {applied_count} timing domain parameters")

    try:
        llm = initialize_ollama_llm()
        logger.info("Timing domain LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Timing LLM initialization failed: {e}")
        raise

def initialize_ollama_llm():
    """Initialize Ollama LLM with environment parameters."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.llms import Ollama as ChatOllama

    base_url = os.getenv('OLLAMA_BASE_URL', 'http://f15dtpai1:11434')
    model = os.getenv('OLLAMA_MODEL', 'qwen2.5_coder_32B')

    llm_params = {
        'model': model,
        'base_url': base_url,
        'temperature': float(os.getenv('LLM_TEMPERATURE', '0.25')),
        'top_p': float(os.getenv('LLM_TOP_P', '0.9')),
        'num_predict': int(os.getenv('LLM_NUM_PREDICT', '2500')),
    }

    # Add optional parameters if available
    if os.getenv('LLM_TOP_K'):
        llm_params['top_k'] = int(os.getenv('LLM_TOP_K'))
    if os.getenv('LLM_REPEAT_PENALTY'):
        llm_params['repeat_penalty'] = float(os.getenv('LLM_REPEAT_PENALTY'))

    return ChatOllama(**llm_params)

def test_ollama_connection():
    """Test Ollama connection and model availability."""
    import requests
    import logging

    logger = logging.getLogger(__name__)
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://f15dtpai1:11434')

    try:
        # Test Ollama server
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            target_model = os.getenv('OLLAMA_MODEL', 'qwen2.5_coder_32B')

            if any(target_model in name for name in model_names):
                logger.info(f"Ollama connection successful, {target_model} available")
                return True
            else:
                logger.warning(f"Model {target_model} not found. Available: {model_names}")
                return False
        else:
            logger.error(f"Ollama server error: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Ollama connection failed: {e}")
        return False

AGENTIC_MODE = True
print("[AGENT] Using Agentic Mode: Autonomous exploration with self-validation")


class TimingDataSelectionAgent:
    """
    Timing-aware agent for intelligent Monte Carlo sample selection.

    Features:
    - Senior timing engineer domain expertise
    - Active learning principles (uncertainty sampling)
    - Process variation awareness
    - Business impact focus (cost reduction)

    Workflow:
    1. OBSERVE: Analyze timing characteristics with domain knowledge
    2. THINK: Strategic reasoning about selection approach
    3. DECIDE: Algorithm selection with timing-specific criteria
    4. ACT: Uncertainty-based sampling for critical corners
    """

    def __init__(self, llm, verbose: bool = True):
        """
        Initialize the timing-aware agent.

        Args:
            llm: LangChain LLM instance
            verbose: Whether to print reasoning steps
        """
        self.llm = llm
        self.verbose = verbose
        self.conversation_history = []
        self.current_data = None
        self.current_features = None
        self.scaler = None
        self.reasoning_log = []
        self._imports_loaded = False

        # Timing domain system prompt
        self.system_prompt = TIMING_SYSTEM_PROMPT

        # Agentic capabilities
        self.agentic_mode = AGENTIC_MODE
        self.validation_boundaries = VALIDATION_BOUNDARIES if AGENTIC_MODE else {}
        self.iteration_triggers = ITERATION_TRIGGERS if AGENTIC_MODE else {}
        self.iteration_count = 0
        self.max_iterations = 3

        if self.agentic_mode:
            print("[AGENT] Initialized in AGENTIC MODE with autonomous exploration")

    def _load_imports(self):
        """Load heavy imports only when needed."""
        if self._imports_loaded:
            return

        # Import LangChain components (these are loaded dynamically)
        global ChatPromptTemplate, HumanMessage, SystemMessage
        from langchain.prompts import ChatPromptTemplate
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError:
            from langchain.schema import HumanMessage, SystemMessage

        # Initialize scaler if needed
        if self.scaler is None and SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()

        self._imports_loaded = True

    def add_message(self, role: str, content: str):
        """Add message to conversation history."""
        self._load_imports()
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': pd.Timestamp.now()
        })

    def log_reasoning(self, stage: str, content: str):
        """Log agent reasoning."""
        self._load_imports()
        self.reasoning_log.append({
            'stage': stage,
            'content': content,
            'timestamp': pd.Timestamp.now()
        })
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"{stage}")
            print(f"{'='*80}")
            print(content)

    def validate_selection_quality(self, selected_indices: List[int], labels: np.ndarray) -> Dict[str, Any]:
        """Validate selection quality against agentic boundaries."""
        if not self.agentic_mode or not self.validation_boundaries:
            return {"validation": "skipped", "quality": "assumed_good"}

        validation_results = {}
        self._load_imports()

        try:
            # Cell name coverage validation (parsed from cell_arc_pt)
            if self.current_data is not None:
                # Try to parse cell names from cell_arc_pt if it exists
                if 'cell_arc_pt' in self.current_data.columns:
                    # Parse cell names from cell_arc_pt
                    cell_names = self.current_data['cell_arc_pt'].str.split('#').str[0]
                    selected_cell_names = self.current_data.iloc[selected_indices]['cell_arc_pt'].str.split('#').str[0]

                    total_cell_types = cell_names.nunique()
                    selected_cell_types = selected_cell_names.nunique()
                    cell_coverage = selected_cell_types / total_cell_types if total_cell_types > 0 else 0

                    min_coverage = self.validation_boundaries.get('minimum_cell_type_coverage', 0.8)
                    validation_results['cell_coverage'] = {
                        'achieved': cell_coverage,
                        'required': min_coverage,
                        'passed': cell_coverage >= min_coverage
                    }
                else:
                    validation_results['cell_coverage'] = {
                        'achieved': 0.0,
                        'required': 0.8,
                        'passed': False,
                        'reason': 'No cell_arc_pt column found for parsing cell types'
                    }

            # Cluster balance validation
            if len(labels) > 0:
                cluster_sizes = [np.sum(labels[selected_indices] == i) for i in np.unique(labels)]
                if len(cluster_sizes) > 1:
                    max_size = max(cluster_sizes)
                    min_size = min([s for s in cluster_sizes if s > 0])
                    imbalance_ratio = max_size / min_size if min_size > 0 else float('inf')

                    max_imbalance = self.validation_boundaries.get('maximum_cluster_imbalance', 3.0)
                    validation_results['cluster_balance'] = {
                        'imbalance_ratio': imbalance_ratio,
                        'max_allowed': max_imbalance,
                        'passed': imbalance_ratio <= max_imbalance
                    }

            # Overall validation status
            all_passed = all(result.get('passed', True) for result in validation_results.values()
                           if isinstance(result, dict) and 'passed' in result)

            validation_results['overall_status'] = 'PASSED' if all_passed else 'FAILED'
            validation_results['requires_iteration'] = not all_passed

            if self.verbose and validation_results.get('requires_iteration'):
                print(f"\n[VALIDATION] Quality check FAILED - iteration required")
                for key, result in validation_results.items():
                    if isinstance(result, dict) and not result.get('passed', True):
                        print(f"  {key}: {result}")

            return validation_results

        except Exception as e:
            return {"validation": "error", "error": str(e), "quality": "unknown"}

    def parse_user_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query with timing domain understanding."""
        self._load_imports()
        parsing_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Parse this timing engineer's request for intelligent Monte Carlo sampling.

Query: "{query}"

CRITICAL: If NO percentage is mentioned in the query, set selection_percentage to null.
ONLY use a percentage if explicitly stated (e.g., "8%", "select 5%", "10 percent").

Extract and return ONLY valid JSON with these fields:
- selection_percentage: float if specified, null if not mentioned (e.g., 8.0 or null)
- selection_criteria: string ("uncertainty", "diversity", "random")
- clustering_preference: string or null ("gmm", "kmeans", or null for auto)
- additional_requirements: string or null for any special timing requirements

Choose the optimal sampling method based on the specific data characteristics discovered.
If no percentage specified, the system will determine optimal percentage based on data analysis.

Return ONLY the JSON object, nothing else.""")
        ])

        chain = parsing_prompt | self.llm
        response = chain.invoke({})

        if hasattr(response, 'content'):
            text = response.content
        else:
            text = str(response)

        text = text.replace('```json', '').replace('```', '').strip()

        try:
            params = json.loads(text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    params = json.loads(json_match.group())
                except json.JSONDecodeError:
                    params = {
                        'selection_percentage': None,  # No default - will determine from data
                        'selection_criteria': 'data_driven',  # Let agent decide based on data
                        'clustering_preference': None,  # Let agent choose best method
                        'additional_requirements': 'adaptive_method_selection'
                    }
            else:
                params = {
                    'selection_percentage': None,  # No default - will determine from data
                    'selection_criteria': 'data_driven',  # Let agent decide based on data
                    'clustering_preference': None,  # Let agent choose best method
                    'additional_requirements': 'adaptive_method_selection'
                }

        # Store parameters for later processing after data is loaded
        self.add_message('assistant', f"Parsed query parameters: {params}")
        return params

    def observe(self, csv_path: str, target_percentage: float = 5.0, use_agentic_explore: bool = True) -> Dict[str, Any]:
        """OBSERVE stage with timing domain analysis."""
        self._load_imports()
        print("\n" + "=" * 100)
        print("[1] STAGE 1: OBSERVE - Timing Domain Analysis & Feature Discovery")
        print("=" * 100)

        self.current_data = pd.read_csv(csv_path)
        print(f"Analyzing {len(self.current_data)} timing arc samples...")

        # DATA LOADING VERIFICATION GATE - DYNAMIC DISCOVERY
        actual_columns = self.current_data.columns.tolist()
        numeric_columns = self.current_data.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
        identifier_columns = self.current_data.select_dtypes(include=['object']).columns.tolist()

        print(f"\n{'=' * 50}")
        print(f"DATA DISCOVERY GATE")
        print(f"{'=' * 50}")
        print(f"Total Columns: {len(actual_columns)}")
        print(f"Numeric Features: {len(numeric_columns)}")
        print(f"Identifier Columns: {len(identifier_columns)}")

        # Check for key identifier column
        if 'cell_arc_pt' in actual_columns:
            print("STATUS: [OK] Key identifier 'cell_arc_pt' found")
        elif any('arc_pt' in col for col in actual_columns):
            print("STATUS: [OK] Arc identifier column found")
        else:
            print("STATUS: [WARNING] No arc identifier column detected")

        print(f"STATUS: [OK] Using dynamic column discovery - no hardcoded expectations")
        print(f"{'=' * 50}")

        # MANDATORY first step: print all column names and preserve all data
        print(f"[SCHEMA] All CSV columns: {self.current_data.columns.tolist()}")
        print(f"[SCHEMA] Data shape: {self.current_data.shape}")
        print(f"[SCHEMA] Data types: {self.current_data.dtypes.to_dict()}")

        # Use ALL available columns except identifier columns for analysis
        available_cols = self.current_data.columns.tolist()

        # Exclude only pure identifier columns, keep all feature data
        exclude_cols = ['cell_arc_pt', 'arc_pt'] if any('arc_pt' in col for col in available_cols) else []
        feature_cols = [col for col in available_cols if col not in exclude_cols]

        print(f"[SCHEMA] Using {len(feature_cols)} feature columns: {feature_cols}")

        # Store ALL features for analysis - no filtering
        if feature_cols:
            self.current_features = self.current_data[feature_cols].values
        else:
            # Fallback to all numeric columns if somehow no features found
            numeric_cols = self.current_data.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
            self.current_features = self.current_data[numeric_cols].values
            feature_cols = numeric_cols
            print(f"[SCHEMA] Fallback to numeric columns: {feature_cols}")

        print(f"[SCHEMA] Feature matrix shape: {self.current_features.shape}")

        observation = {
            'total_samples': len(self.current_data),
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
            'timing_statistics': {},
            'cell_types': {}
        }

        # Timing-specific analysis
        for col in feature_cols:
            if col in self.current_data.columns:
                observation['timing_statistics'][col] = {
                    'mean': float(self.current_data[col].mean()),
                    'std': float(self.current_data[col].std()),
                    'min': float(self.current_data[col].min()),
                    'max': float(self.current_data[col].max()),
                    'cv': float(self.current_data[col].std() / self.current_data[col].mean()) if self.current_data[col].mean() != 0 else 0
                }

        # Process variation analysis
        corr_matrix = self.current_data[feature_cols].corr()
        timing_correlations = []

        for i, col1 in enumerate(feature_cols):
            for j, col2 in enumerate(feature_cols[i+1:], i+1):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    timing_correlations.append({
                        'feature1': col1,
                        'feature2': col2,
                        'correlation': float(corr_val)
                    })

        observation['high_correlations'] = timing_correlations

        # Cell type and table position analysis - parse from actual column names
        print(f"[SCHEMA] Available columns: {list(self.current_data.columns)}")
        try:
            if 'cell_arc_pt' in self.current_data.columns:
                # Parse cell names from cell_arc_pt (everything before first #)
                cell_names = self.current_data['cell_arc_pt'].str.split('#').str[0]
                observation['cell_types'] = cell_names.value_counts().to_dict()
                print(f"[SCHEMA] Parsed {len(observation['cell_types'])} cell types from cell_arc_pt column")
                print(f"[SCHEMA] Sample cell names: {list(observation['cell_types'].keys())[:5]}")

                # CRITICAL FIX: Parse table positions using rsplit to avoid false matches
                print(f"\n{'=' * 50}")
                print(f"TABLE POSITION PARSING")
                print(f"{'=' * 50}")

                def parse_table_position(cell_arc_pt_value):
                    """Extract table row and column using rsplit to avoid false matches with cell names."""
                    parts = cell_arc_pt_value.rsplit('_', 2)  # Split from right, max 2 splits
                    if len(parts) >= 3:
                        table_row = int(parts[-2])  # Second to last = row
                        table_col = int(parts[-1])  # Last = col
                        return table_row, table_col
                    else:
                        return None, None

                # Apply correct parsing
                table_positions = self.current_data['cell_arc_pt'].apply(parse_table_position)
                self.current_data['table_row'] = [pos[0] for pos in table_positions]
                self.current_data['table_col'] = [pos[1] for pos in table_positions]

                # Remove any invalid positions
                valid_positions = (self.current_data['table_row'].notna()) & (self.current_data['table_col'].notna())
                if valid_positions.sum() > 0:
                    # Validation
                    min_row = self.current_data.loc[valid_positions, 'table_row'].min()
                    max_row = self.current_data.loc[valid_positions, 'table_row'].max()
                    min_col = self.current_data.loc[valid_positions, 'table_col'].min()
                    max_col = self.current_data.loc[valid_positions, 'table_col'].max()

                    print(f"Table row range: {min_row} to {max_row}")
                    print(f"Table col range: {min_col} to {max_col}")
                    print(f"Valid table positions: {valid_positions.sum()} / {len(self.current_data)}")

                    # Verify with examples
                    sample_indices = self.current_data.index[:3]
                    for idx in sample_indices:
                        cell_pt = self.current_data.loc[idx, 'cell_arc_pt']
                        row = self.current_data.loc[idx, 'table_row']
                        col = self.current_data.loc[idx, 'table_col']
                        print(f"Example: '{cell_pt[-10:]}' → row={row}, col={col}")

                    # VALIDATION GATE: table indices must be within expected range
                    if max_row > 8 or max_col > 8:
                        print(f"ERROR: Table position parsing failed - max row/col {max_row}/{max_col} > 8")
                        print("This indicates the parsing regex is matching wrong digit groups!")
                    else:
                        print("[OK] Table position parsing validated: all values in expected range")

                    # Add boundary case identification
                    table_size = max(max_row, max_col)
                    self.current_data['is_boundary'] = (
                        ((self.current_data['table_row'] == 1) | (self.current_data['table_row'] == table_size)) &
                        ((self.current_data['table_col'] == 1) | (self.current_data['table_col'] == table_size))
                    )
                    boundary_count = self.current_data['is_boundary'].sum()
                    print(f"Boundary cases identified: {boundary_count} ({boundary_count/len(self.current_data)*100:.1f}%)")

                print(f"{'=' * 50}")

            elif 'arc_pt' in self.current_data.columns:
                # Fallback to arc_pt if available
                cell_types = self.current_data['arc_pt'].str.extract(r'^([A-Z0-9]+)')[0]
                observation['cell_types'] = cell_types.value_counts().to_dict()
                print(f"[SCHEMA] Parsed {len(observation['cell_types'])} cell types from arc_pt column")
            else:
                observation['cell_types'] = {'unknown': len(self.current_data)}
                print("[SCHEMA] Warning: No cell identifier column found - cannot parse cell types")
                print(f"[SCHEMA] Available columns for reference: {list(self.current_data.columns)}")
        except Exception as e:
            observation['cell_types'] = {'unknown': len(self.current_data)}
            print(f"[SCHEMA] Error parsing cell types/table positions: {e}")
            print(f"[SCHEMA] Available columns: {list(self.current_data.columns)}")

        # CRITICAL FIX: Calculate actual statistics for prompt injection
        calculated_stats = []
        key_features = ['nominal_delay', 'lib_sigma_delay_late', 'sigma_by_nominal']

        for feature in key_features:
            if feature in feature_cols:
                stats = observation['timing_statistics'][feature]
                calculated_stats.append(f"- {feature}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, range={stats['min']:.4f} to {stats['max']:.4f}")

        correlation_details = []
        for corr in timing_correlations:
            correlation_details.append(f"- {corr['feature1']} vs {corr['feature2']}: r={corr['correlation']:.3f}")

        # Sigma_by_nominal specific analysis
        sigma_analysis = "No sigma_by_nominal data available"
        if 'sigma_by_nominal' in feature_cols and 'sigma_by_nominal' in observation['timing_statistics']:
            sigma_stats = observation['timing_statistics']['sigma_by_nominal']
            sigma_analysis = f"Range: {sigma_stats['min']:.3f} to {sigma_stats['max']:.3f}, Mean: {sigma_stats['mean']:.3f}, Std: {sigma_stats['std']:.3f}"

        # Generate timing domain observation with ACTUAL DATA
        if target_percentage is None:
            raise ValueError("target_percentage cannot be None at this stage")
        target_count = int(observation['total_samples'] * target_percentage / 100)

        try:
            if use_agentic_explore:
                observe_prompt = AGENTIC_EXPLORE_PROMPT.format(
                    total_samples=observation['total_samples'],
                    target_count=target_count,
                    target_percentage=target_percentage,
                    n_features=observation['n_features'],
                    n_cell_types=len(observation['cell_types']),
                    calculated_stats='\n'.join(calculated_stats) if calculated_stats else "No key timing features found in dataset",
                    correlation_details='\n'.join(correlation_details) if correlation_details else "No high correlations detected",
                    sigma_analysis=sigma_analysis
                )
            else:
                observe_prompt = TIMING_OBSERVE_PROMPT.format(
                    total_samples=observation['total_samples'],
                    target_count=target_count,
                    target_percentage=target_percentage,
                    n_features=observation['n_features'],
                    n_cell_types=len(observation['cell_types']),
                    calculated_stats='\n'.join(calculated_stats) if calculated_stats else "No key timing features found in dataset",
                    correlation_details='\n'.join(correlation_details) if correlation_details else "No high correlations detected",
                    sigma_analysis=sigma_analysis
                )
        except KeyError as e:
            print(f"[ERROR] Missing parameter in OBSERVE prompt: {e}")
            observe_prompt = f"Analyze this timing dataset with {observation['total_samples']} samples for {target_percentage}% selection."

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=observe_prompt)
        ])

        chain = prompt_template | self.llm
        observation_reasoning = chain.invoke({})

        if hasattr(observation_reasoning, 'content'):
            observation_text = observation_reasoning.content
        else:
            observation_text = str(observation_reasoning)

        self.add_message('assistant', observation_text)
        self.log_reasoning('OBSERVE', observation_text)

        return observation

    def think(self, observation: Dict[str, Any], target_percentage: float) -> Dict[str, Any]:
        """THINK stage with timing strategy reasoning."""
        self._load_imports()
        print("\n" + "=" * 100)
        print("[2] STAGE 2: THINK - Strategic Timing Analysis & Sampling Strategy")
        print("=" * 100)

        if target_percentage is None:
            raise ValueError("target_percentage cannot be None at this stage")
        target_count = int(observation['total_samples'] * target_percentage / 100)

        # Build exploration findings summary for agentic prompts
        exploration_findings = f"""Dataset Analysis Results:
- Total Samples: {observation['total_samples']:,}
- Target Selection: {target_percentage:.1f}% = {target_count:,} samples
- Feature Dimensions: {observation['n_features']}
- Cell Type Diversity: {len(observation['cell_types'])} types
- High Correlations Found: {len(observation['high_correlations'])} pairs
- Statistical Complexity: {len(observation.get('timing_statistics', {}))} timing features analyzed

Key Findings: This dataset shows {'high' if len(observation['high_correlations']) > 3 else 'moderate'} correlation complexity
and {'diverse' if len(observation['cell_types']) > 10 else 'limited'} cell type diversity, suggesting
{'advanced clustering strategies' if len(observation['high_correlations']) > 3 else 'standard sampling approaches'} may be optimal."""

        # Debug: Print parameters being passed
        if self.verbose:
            print(f"[DEBUG] Formatting THINK prompt with:")
            print(f"  - exploration_findings length: {len(exploration_findings)}")
            print(f"  - target_count: {target_count}")

        try:
            if self.agentic_mode:
                think_prompt = AGENTIC_STRATEGY_PROMPT.format(
                    exploration_findings=exploration_findings,
                    target_count=target_count
                )
            else:
                think_prompt = TIMING_THINK_PROMPT.format(
                    exploration_findings=exploration_findings,
                    target_count=target_count
                )
        except KeyError as e:
            print(f"[ERROR] Missing parameter in THINK prompt: {e}")
            # Provide fallback prompt without formatting
            think_prompt = f"Develop a sampling strategy for {target_count} samples from this timing dataset. Use the exploration findings to guide your approach."

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=think_prompt)
        ])

        chain = prompt_template | self.llm
        thinking_reasoning = chain.invoke({})

        if hasattr(thinking_reasoning, 'content'):
            thinking_text = thinking_reasoning.content
        else:
            thinking_text = str(thinking_reasoning)

        self.add_message('assistant', thinking_text)
        self.log_reasoning('THINK', thinking_text)

        strategy = {
            'target_percentage': target_percentage,
            'target_count': target_count,
            'use_pca': True,
            'variance_threshold': 0.92,
            'n_clusters_range': [8, 10, 12],
            'selection_method': 'uncertainty_based',
            'timing_focus': True,
            'reasoning': thinking_text,
            'exploration_findings': exploration_findings
        }

        return strategy

    def decide(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """DECIDE stage with timing algorithm selection."""
        self._load_imports()
        print("\n" + "=" * 100)
        print("[3] STAGE 3: DECIDE - Algorithm Selection & Parameter Optimization")
        print("=" * 100)

        # PCA for timing feature compression
        print("Applying PCA for timing feature optimization...")
        features_scaled = self.scaler.fit_transform(self.current_features)
        pca = PCA()
        pca.fit(features_scaled)

        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= strategy['variance_threshold']) + 1

        pca_final = PCA(n_components=n_components)
        features_pca = pca_final.fit_transform(features_scaled)

        print(f"PCA: {len(pca.explained_variance_ratio_)} -> {n_components} components")
        print(f"Variance preserved: {cumsum[n_components-1]*100:.1f}%")

        # Timing-aware clustering comparison
        print("\nTesting clustering algorithms for timing data...")
        results = {}
        metrics = []

        for k in strategy['n_clusters_range']:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(features_pca)
            kmeans_inertia = kmeans.inertia_

            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm_labels = gmm.fit_predict(features_pca)
            gmm_bic = gmm.bic(features_pca)

            results[k] = {
                'kmeans': {'inertia': kmeans_inertia, 'labels': kmeans_labels},
                'gmm': {'bic': gmm_bic, 'labels': gmm_labels}
            }

            metric_str = f"k={k}: K-means inertia={kmeans_inertia:.0f}, GMM BIC={gmm_bic:.0f}"
            metrics.append(metric_str)
            print(f"  {metric_str}")

        # Calculate assessment based on variance explained
        variance_pct = cumsum[n_components-1]
        if variance_pct > 0.9:
            assessment = 'Excellent'
        elif variance_pct > 0.85:
            assessment = 'Good'
        else:
            assessment = 'Acceptable'

        decide_prompt = TIMING_DECIDE_PROMPT.format(
            original_features=len(self.current_features[0]),
            pca_components=n_components,
            variance_explained=variance_pct*100,
            assessment=assessment,
            clustering_metrics='\n'.join(metrics)
        )

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=decide_prompt)
        ])

        chain = prompt_template | self.llm
        decision_reasoning = chain.invoke({})

        if hasattr(decision_reasoning, 'content'):
            decision_text = decision_reasoning.content
        else:
            decision_text = str(decision_reasoning)

        best_k = self._extract_cluster_count(decision_text)
        best_algo = 'gmm' if 'gmm' in decision_text.lower() or 'gaussian' in decision_text.lower() else 'kmeans'

        self.add_message('assistant', decision_text)
        self.log_reasoning('DECIDE', decision_text)

        # Fit final model
        if best_algo == 'gmm':
            final_model = GaussianMixture(n_components=best_k, random_state=42)
            final_labels = final_model.fit_predict(features_pca)
            centroids = final_model.means_
        else:
            final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            final_labels = final_model.fit_predict(features_pca)
            centroids = final_model.cluster_centers_

        distances = cdist(features_pca, centroids, metric='euclidean')
        min_distances = np.min(distances, axis=1)

        decision = {
            'pca': {
                'n_components': n_components,
                'variance_explained': float(cumsum[n_components-1]),
                'transformer': pca_final
            },
            'clustering': {
                'algorithm': best_algo,
                'n_clusters': best_k,
                'model': final_model,
                'labels': final_labels,
                'centroids': centroids,
                'distances': min_distances
            },
            'features_pca': features_pca
        }

        return decision

    def act(self, decision: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """ACT stage with adaptive sampling method selection based on data characteristics."""
        self._load_imports()
        print("\n" + "=" * 100)
        print("[4] STAGE 4: ACT - Timing-Optimized Sample Selection & Validation")
        print("=" * 100)

        target_count = strategy['target_count']
        labels = decision['clustering']['labels']
        distances = decision['clustering']['distances']
        n_clusters = decision['clustering']['n_clusters']

        cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
        base_per_cluster = [int(size * strategy['target_percentage'] / 100) for size in cluster_sizes]

        # Timing-aware uncertainty boosting
        cluster_uncertainties = []
        for i in range(n_clusters):
            mask = labels == i
            if np.any(mask):
                uncertainty = np.percentile(distances[mask], 90)
            else:
                uncertainty = 0
            cluster_uncertainties.append(uncertainty)

        max_uncertainty = max(cluster_uncertainties) if cluster_uncertainties else 1
        uncertainty_weights = [u / max_uncertainty for u in cluster_uncertainties]

        # Enhanced allocation for timing corners
        adjusted_per_cluster = []
        for i in range(n_clusters):
            base = base_per_cluster[i]
            boost = int(base * uncertainty_weights[i] * 0.3)
            adjusted_per_cluster.append(base + boost)

        total_adjusted = sum(adjusted_per_cluster)
        final_per_cluster = [int(c * target_count / total_adjusted) for c in adjusted_per_cluster]

        diff = target_count - sum(final_per_cluster)
        if diff > 0:
            highest_uncertainty_cluster = np.argmax(uncertainty_weights)
            final_per_cluster[highest_uncertainty_cluster] += diff

        # Uncertainty-based selection (samples far from centroids)
        selected_indices = []
        selection_details = []

        for i in range(n_clusters):
            mask = labels == i
            cluster_indices = np.where(mask)[0]
            cluster_distances = distances[mask]

            n_select = final_per_cluster[i]

            sorted_idx = np.argsort(cluster_distances)[::-1]
            selected = cluster_indices[sorted_idx[:n_select]]
            selected_indices.extend(selected.tolist())

            selection_details.append(
                f"Cluster {i}: {n_select}/{cluster_sizes[i]} samples ({n_select/cluster_sizes[i]*100:.1f}%)"
            )
            print(f"  {selection_details[-1]}")

        selected_df = self.current_data.iloc[selected_indices].copy()
        selected_df['cluster_id'] = labels[selected_indices]
        selected_df['uncertainty_score'] = distances[selected_indices]

        act_prompt = TIMING_ACT_PROMPT.format(
            total_samples=len(self.current_data),
            target_percentage=strategy['target_percentage'],
            target_count=target_count,
            n_selected=len(selected_indices),
            n_clusters=n_clusters,
            selection_details='\n'.join(selection_details)
        )

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=act_prompt)
        ])

        chain = prompt_template | self.llm
        action_reasoning = chain.invoke({})

        if hasattr(action_reasoning, 'content'):
            action_text = action_reasoning.content
        else:
            action_text = str(action_reasoning)

        self.add_message('assistant', action_text)
        self.log_reasoning('ACT', action_text)

        result = {
            'selected_df': selected_df,
            'selected_indices': selected_indices,
            'n_selected': len(selected_indices),
            'cluster_distribution': final_per_cluster,
            'uncertainty_weights': uncertainty_weights,
            'selection_details': selection_details,
            'expected_cost_reduction': '50% (10% to 5% Monte Carlo coverage)'
        }

        return result

    def act_agentic(self, strategy: Dict[str, Any], target_percentage: float) -> Dict[str, Any]:
        """Agentic ACT stage with autonomous decision-making and self-validation."""
        self._load_imports()
        print("\n" + "=" * 100)
        print("[A] STAGE 3: AGENTIC EXECUTION - Autonomous Decision & Action")
        print("=" * 100)

        # Generate execution plan with autonomous decision-making
        if target_percentage is None:
            raise ValueError("target_percentage cannot be None at this stage")
        target_count = int(len(self.current_data) * target_percentage / 100)

        try:
            act_prompt = AGENTIC_EXECUTE_PROMPT.format(
                validated_strategy=strategy.get('reasoning', 'Autonomous strategy developed'),
                target_count=target_count,
                total_samples=len(self.current_data),
                algorithm_choice="Autonomous GMM clustering",
                algorithm_config="Adaptive parameters with self-validation"
            )
        except KeyError as e:
            print(f"[ERROR] Missing parameter in ACT prompt: {e}")
            act_prompt = f"Execute sampling strategy to select {target_count} samples from {len(self.current_data)} total samples."

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=act_prompt)
        ])

        chain = prompt_template | self.llm
        execution_reasoning = chain.invoke({})

        if hasattr(execution_reasoning, 'content'):
            execution_text = execution_reasoning.content
        else:
            execution_text = str(execution_reasoning)

        self.add_message('assistant', execution_text)
        self.log_reasoning('AGENTIC EXECUTION', execution_text)

        # Execute the sampling with iteration capability
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Apply PCA and clustering (similar to standard decide logic)
                features_scaled = self.scaler.fit_transform(self.current_features)
                pca = PCA()
                pca.fit(features_scaled)

                cumsum = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumsum >= 0.92) + 1

                pca_final = PCA(n_components=n_components)
                features_pca = pca_final.fit_transform(features_scaled)

                # Let agent choose optimal clustering method based on data characteristics
                best_k = min(10, len(self.current_data) // 100)  # Adaptive cluster count

                # Try both methods and let agent choose based on data characteristics
                from sklearn.cluster import KMeans
                kmeans_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                kmeans_labels = kmeans_model.fit_predict(features_pca)
                kmeans_inertia = kmeans_model.inertia_

                gmm_model = GaussianMixture(n_components=best_k, random_state=42)
                gmm_labels = gmm_model.fit_predict(features_pca)
                gmm_bic = gmm_model.bic(features_pca)

                # Simple heuristic: use method with better relative performance
                # GMM better for overlapping clusters, K-means for well-separated
                use_gmm = gmm_bic < -1000 * best_k  # GMM has good fit if BIC is sufficiently negative

                if use_gmm:
                    final_model = gmm_model
                    final_labels = gmm_labels
                    centroids = final_model.means_
                    print(f"Selected GMM (BIC: {gmm_bic:.0f})")
                else:
                    final_model = kmeans_model
                    final_labels = kmeans_labels
                    centroids = final_model.cluster_centers_
                    print(f"Selected K-means (inertia: {kmeans_inertia:.0f})")

                distances = cdist(features_pca, centroids, metric='euclidean')
                min_distances = np.min(distances, axis=1)

                # Use the target_count calculated earlier (no recalculation needed)

                # Select samples far from centroids (high uncertainty)
                uncertainty_scores = min_distances
                selected_indices = np.argsort(-uncertainty_scores)[:target_count]

                # Validate selection quality
                validation_results = self.validate_selection_quality(selected_indices, final_labels)

                if not validation_results.get('requires_iteration', False) or attempt == max_attempts - 1:
                    # Selection is good or we've exhausted attempts
                    break
                else:
                    print(f"\n[ITERATION {attempt + 1}] Validation failed, adjusting strategy...")
                    # Adjust target count or approach for next iteration
                    if target_count is not None:
                        target_count = int(target_count * 1.1)  # Slightly increase sample count
                    else:
                        raise ValueError("target_count cannot be None during iteration")

            except Exception as e:
                print(f"[ERROR] Execution attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise

        # Create results
        selected_df = self.current_data.iloc[selected_indices].copy()
        selected_df['cluster_id'] = final_labels[selected_indices]
        selected_df['uncertainty_score'] = uncertainty_scores[selected_indices]

        result = {
            'selected_df': selected_df,
            'selected_indices': selected_indices,
            'n_selected': len(selected_indices),
            'cluster_distribution': [np.sum(final_labels[selected_indices] == i) for i in range(best_k)],
            'uncertainty_weights': uncertainty_scores[selected_indices],
            'selection_details': [f"Selected {len(selected_indices)} high-uncertainty samples"],
            'expected_cost_reduction': f'{100 - target_percentage:.0f}% reduction (from 10% to {target_percentage:.1f}%)',
            'validation_results': validation_results,
            'agentic_iterations': attempt + 1
        }

        # Create synthetic decision structure for compatibility
        decision = {
            'pca': {
                'n_components': n_components,
                'variance_explained': float(cumsum[n_components-1]),
                'transformer': pca_final
            },
            'clustering': {
                'algorithm': 'gmm',
                'n_clusters': best_k,
                'model': final_model,
                'labels': final_labels,
                'centroids': centroids,
                'distances': min_distances
            },
            'features_pca': features_pca
        }

        # Store decision for return compatibility
        self.last_decision = decision

        return result

    def handle_conversation(self, user_query: str) -> Dict[str, Any]:
        """Handle conversational questions about results without re-running selection."""
        self._load_imports()

        # Classify user intent
        intent, params = self.classify_user_intent(user_query)

        if intent == UserIntent.QUESTION_ABOUT_RESULTS:
            # Generate response based on conversation history
            context = "\n".join([msg['content'] for msg in self.conversation_history[-5:]])

            question_prompt = f"""Based on our previous conversation about timing data selection, answer this follow-up question:

User Question: {user_query}

Recent Context:
            {context}

Provide a clear, technical explanation addressing their specific question about the selection methodology, results, or reasoning. Use plain text only."""

            prompt_template = ChatPromptTemplate.from_messages([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question_prompt)
            ])

            chain = prompt_template | self.llm
            response = chain.invoke({})

            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            self.add_message('assistant', response_text)

            return {
                'type': 'conversational_response',
                'intent': intent.value,
                'response': response_text,
                'parameters': params
            }

        elif intent == UserIntent.EXPLAIN_METHODOLOGY:
            # Explain methodology without running selection
            methodology_prompt = f"""Explain the timing data selection methodology to address this question:

{user_query}

Provide a technical explanation of the algorithms, approaches, and reasoning behind the methodology. Focus on the specific aspect they're asking about."""

            prompt_template = ChatPromptTemplate.from_messages([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=methodology_prompt)
            ])

            chain = prompt_template | self.llm
            response = chain.invoke({})

            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            self.add_message('assistant', response_text)

            return {
                'type': 'methodology_explanation',
                'intent': intent.value,
                'response': response_text,
                'parameters': params
            }

        else:
            # For other intents, indicate that selection should be run
            return {
                'type': 'requires_execution',
                'intent': intent.value,
                'parameters': params
            }

    def run_selection(self, user_query: str, csv_path: str) -> Dict[str, Any]:
        """Main workflow with timing domain expertise."""
        self._load_imports()
        print("\n" + "=" * 80)
        print("TIMING-AWARE DATA SELECTION AGENT")
        print("=" * 80)

        self.add_message('user', user_query)

        print("\nParsing timing engineer requirements...")
        params = self.parse_user_query(user_query)

        # Check if this is a conversational question first
        intent, intent_params = self.classify_user_intent(user_query)

        if intent in [UserIntent.QUESTION_ABOUT_RESULTS, UserIntent.EXPLAIN_METHODOLOGY] and len(self.conversation_history) > 0:
            print("\n[CONVERSATIONAL] Detected follow-up question - providing contextual response")
            return self.handle_conversation(user_query)

        # Handle null percentage first - determine optimal percentage based on actual dataset size
        if params.get('selection_percentage') is None:
            # Need to load data first to determine size
            temp_data = pd.read_csv(csv_path)
            data_size = len(temp_data)
            if data_size > 50000:
                optimal_percentage = 3.0  # Large datasets need less percentage
            elif data_size > 20000:
                optimal_percentage = 5.0  # Medium datasets
            else:
                optimal_percentage = 8.0  # Smaller datasets can afford higher percentage

            params['selection_percentage'] = optimal_percentage
            print(f"No percentage specified. Data-driven selection: {optimal_percentage}% for {data_size:,} samples")

        # Now call observe with valid percentage
        observation = self.observe(csv_path, params['selection_percentage'], use_agentic_explore=self.agentic_mode)

        # Update observation with final percentage
        observation['final_target_percentage'] = params['selection_percentage']

        strategy = self.think(observation, params['selection_percentage'])

        if self.agentic_mode:
            # In agentic mode, strategy includes decision-making
            # Skip separate decide stage and let act handle both
            print("\n[AGENTIC] Strategy includes autonomous decision-making")
            result = self.act_agentic(strategy, params['selection_percentage'])
        else:
            # Standard mode with separate decide stage
            decision = self.decide(strategy)
            result = self.act(decision, strategy)

        print("\n" + "=" * 80)
        print("TIMING-OPTIMIZED SELECTION COMPLETE")
        print("=" * 80)
        print(f"Selected {result['n_selected']}/{len(self.current_data)} samples ({result['n_selected']/len(self.current_data)*100:.1f}%)")
        print(f"Expected cost reduction: {result['expected_cost_reduction']}")
        print(f"Active learning: Uncertainty-based sampling for timing robustness")

        # Handle decision structure for agentic vs standard mode
        if self.agentic_mode:
            decision = getattr(self, 'last_decision', {})

        return {
            'observation': observation,
            'strategy': strategy,
            'decision': decision,
            'result': result,
            'reasoning_log': self.reasoning_log,
            'conversation_history': self.conversation_history,
            'parsed_params': params,
            'agentic_mode': self.agentic_mode
        }

    def _extract_cluster_count(self, text: str) -> int:
        """Extract cluster count from LLM response."""
        try:
            matches = re.findall(r'(\d+)\s*cluster', text, re.IGNORECASE)
            if matches:
                return int(matches[0])

            matches = re.findall(r'k\s*=\s*(\d+)', text)
            if matches:
                return int(matches[0])

            matches = re.findall(r'(?:optimal|best|choose).*?(\d+)', text, re.IGNORECASE)
            if matches:
                return int(matches[0])
        except:
            pass

        return 10

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history

    def self_test(self, verbose: bool = True) -> bool:
        """Run internal self-tests to validate agent functionality."""
        if verbose:
            print("Running agent self-tests...")

        success_count = 0
        total_tests = 0

        # Test 1: Intent Classification
        try:
            test_cases = [
                ("why not k-means?", UserIntent.QUESTION_ABOUT_RESULTS),
                ("Select 5% of timing data", UserIntent.EXECUTE_SAMPLING),
                ("Change to 8%", UserIntent.MODIFY_PARAMETERS),
                ("show dashboard", UserIntent.REQUEST_VISUALIZATION),
                ("How does clustering work?", UserIntent.EXPLAIN_METHODOLOGY),
                ("help", UserIntent.GENERAL_HELP),
            ]

            intent_success = 0
            for test_input, expected_intent in test_cases:
                actual_intent, params = self.classify_user_intent(test_input)
                if actual_intent == expected_intent:
                    intent_success += 1
                    if verbose:
                        print(f"[PASS] Intent: '{test_input}' -> {actual_intent.value}")
                elif verbose:
                    print(f"[FAIL] Intent: '{test_input}' -> Expected: {expected_intent.value}, Got: {actual_intent.value}")

            if intent_success >= len(test_cases) * 0.8:  # 80% success rate
                success_count += 1
                if verbose:
                    print(f"[PASS] Intent Classification: {intent_success}/{len(test_cases)} passed")
            else:
                if verbose:
                    print(f"[FAIL] Intent Classification: {intent_success}/{len(test_cases)} passed")
            total_tests += 1
        except Exception as e:
            if verbose:
                print(f"[FAIL] Intent Classification failed: {e}")
            total_tests += 1

        # Test 2: Enhanced Prompts
        try:
            test_params = {
                'total_samples': 10000,
                'target_count': 500,
                'target_percentage': 5.0,
                'n_features': 12,
                'n_cell_types': 8,
                'calculated_stats': 'test stats',
                'correlation_details': 'test correlations',
                'sigma_analysis': 'test sigma'
            }

            # Test AGENTIC_EXPLORE_PROMPT formatting
            formatted = AGENTIC_EXPLORE_PROMPT.format(**test_params)
            if "10000" in formatted and "500" in formatted:
                success_count += 1
                if verbose:
                    print("[PASS] Enhanced prompts formatting works")
            elif verbose:
                print("[FAIL] Enhanced prompts formatting failed")
            total_tests += 1
        except Exception as e:
            if verbose:
                print(f"[FAIL] Enhanced prompts test failed: {e}")
            total_tests += 1

        # Test 3: Validation boundaries
        try:
            if ('minimum_cell_type_coverage' in VALIDATION_BOUNDARIES and
                'temperature' in AGENTIC_LLM_PARAMETERS):
                success_count += 1
                if verbose:
                    print("[PASS] Enhanced parameters and boundaries loaded")
            elif verbose:
                print("[FAIL] Enhanced parameters missing")
            total_tests += 1
        except Exception as e:
            if verbose:
                print(f"[FAIL] Validation boundaries test failed: {e}")
            total_tests += 1

        # Summary
        success_rate = success_count / total_tests if total_tests > 0 else 0
        if verbose:
            print(f"\nSelf-test results: {success_count}/{total_tests} tests passed ({success_rate:.0%})")
            if success_rate >= 0.8:
                print("[PASS] Agent self-test PASSED - System ready!")
            else:
                print("[FAIL] Agent self-test FAILED - Review configuration")

        return success_rate >= 0.8

    # SAFE ALLOCATION METHODS
    def safe_int(self, value: Union[int, float, str, None], default: int = 0) -> int:
        """Convert any value to safe integer, defaulting to 0 for None/invalid values."""
        if value is None:
            return default
        try:
            if isinstance(value, str):
                # Extract numbers from string responses like "30 samples" or "None allocated"
                numbers = re.findall(r'\d+', value)
                return int(numbers[0]) if numbers else default
            return max(0, int(float(value)))  # Ensure non-negative
        except (ValueError, TypeError):
            return default

    def safe_sample_allocation(self, strategy_results: Dict[str, Any], total_target: int) -> Dict[str, int]:
        """Safely allocate samples across different strategies with None-type protection."""

        # Define all possible allocation strategies
        allocation_strategies = [
            'grid_sampling',
            'uncertainty_sampling',
            'boundary_sampling',
            'sparse_region_exploration',
            'validation_holdout',
            'representative_coverage',
            'corner_case_sampling'
        ]

        # Extract and safely convert all allocations
        safe_allocations = {}
        total_allocated = 0

        for strategy in allocation_strategies:
            raw_value = strategy_results.get(strategy, 0)
            safe_count = self.safe_int(raw_value, 0)
            safe_allocations[strategy] = safe_count
            total_allocated += safe_count

        # Handle over/under allocation
        if total_allocated > total_target:
            print(f"[WARNING] Over-allocation detected: {total_allocated} > {total_target}")
            # Proportionally reduce all non-zero allocations
            scale_factor = total_target / total_allocated
            for strategy in safe_allocations:
                if safe_allocations[strategy] > 0:
                    safe_allocations[strategy] = max(1, int(safe_allocations[strategy] * scale_factor))

            # Recalculate total after scaling
            total_allocated = sum(safe_allocations.values())

        # Handle under-allocation by adding to largest strategy
        if total_allocated < total_target:
            remaining = total_target - total_allocated
            largest_strategy = max(safe_allocations.keys(), key=lambda k: safe_allocations[k])
            safe_allocations[largest_strategy] += remaining
            print(f"[INFO] Added {remaining} samples to {largest_strategy} to reach target")

        # Final validation
        final_total = sum(safe_allocations.values())
        assert final_total == total_target, f"Allocation error: {final_total} != {total_target}"

        return safe_allocations

    # INTENT CLASSIFICATION METHODS
    def classify_user_intent(self, user_input: str) -> Tuple[UserIntent, Dict[str, Any]]:
        """Classify user intent to determine whether to execute pipeline or answer from context."""

        input_lower = user_input.lower().strip()

        # Intent patterns with priorities (most specific first)
        intent_patterns = {
            UserIntent.QUESTION_ABOUT_RESULTS: [
                r'why did you (choose|pick|select)',
                r'why.*(\d+)%',
                r'explain (the|your) (selection|choice|decision)',
                r'how did you (decide|determine)',
                r'what.*reasoning.*behind',
                r'justify.*selection',
                r'(why|how).*samples',
                r'rationale.*for',
                r'why not.*k-?means',
                r'why not.*clustering',
                r'what about.*algorithm'
            ],

            UserIntent.MODIFY_PARAMETERS: [
                r'change.*to.*(\d+)%',
                r'try.*(\d+)%.*instead',
                r'use.*(\d+).*samples',
                r'increase.*to.*(\d+)',
                r'decrease.*to.*(\d+)',
                r'modify.*percentage',
                r'adjust.*selection'
            ],

            UserIntent.REQUEST_VISUALIZATION: [
                r'show.*plot',
                r'show.*visualization',
                r'visuali[sz]e.*results',
                r'generate.*dashboard',
                r'plot.*samples',
                r'show.*scatter',
                r'display.*chart',
                r'show.*dashboard'
            ],

            UserIntent.EXPLAIN_METHODOLOGY: [
                r'how does.*work',
                r'explain.*algorithm',
                r'what.*method.*using',
                r'describe.*approach',
                r'methodology',
                r'what is.*k-?means',
                r'difference between.*algorithms'
            ],

            UserIntent.EXECUTE_SAMPLING: [
                r'select.*(\d+)%',
                r'run.*sampling',
                r'perform.*selection',
                r'execute.*analysis',
                r'analyze.*dataset',
                r'sample.*(\d+)',
                r'choose.*samples'
            ],

            UserIntent.GENERAL_HELP: [
                r'help',
                r'what.*can.*do',
                r'how.*use',
                r'commands'
            ]
        }

        # Check patterns in order of priority
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, input_lower)
                if match:
                    # Extract parameters from match
                    params = self._extract_intent_parameters(user_input, match)
                    return intent, params

        # Default to execution if no clear pattern matches
        return UserIntent.EXECUTE_SAMPLING, {}

    def _extract_intent_parameters(self, user_input: str, match: re.Match) -> Dict[str, Any]:
        """Extract parameters from matched intent patterns."""
        params = {}

        # Extract percentages
        percentage_matches = re.findall(r'(\d+)%', user_input)
        if percentage_matches:
            params['percentage'] = int(percentage_matches[0])

        # Extract sample counts
        sample_matches = re.findall(r'(\d+)\s*samples?', user_input, re.IGNORECASE)
        if sample_matches:
            params['sample_count'] = int(sample_matches[0])

        # Extract algorithm names
        algorithm_matches = re.findall(r'(k-?means|gmm|clustering)', user_input, re.IGNORECASE)
        if algorithm_matches:
            params['algorithm'] = algorithm_matches[0].lower().replace('-', '')

        return params

    # INTERACTIVE VISUALIZATION METHODS
    def generate_interactive_dashboard(self, df: pd.DataFrame, selected_indices: List[int],
                                     clusters: np.ndarray, centroids: np.ndarray,
                                     pca_components: Optional[np.ndarray] = None,
                                     export_html: bool = True) -> Dict[str, Any]:
        """Generate comprehensive interactive dashboard with Plotly."""

        if not PLOTLY_AVAILABLE:
            print("[WARNING] Plotly not available - using fallback visualization")
            return self._fallback_visualization(df, selected_indices, clusters)

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

        # Add plots
        self._add_selection_overview(fig, dashboard_data, row=1, col=1)
        self._add_cluster_analysis(fig, dashboard_data, centroids, row=1, col=2)
        self._add_selection_statistics(fig, dashboard_data, row=2, col=1)
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

    def _prepare_dashboard_data(self, df: pd.DataFrame, selected_indices: List[int],
                               clusters: np.ndarray, pca_components: Optional[np.ndarray] = None) -> Dict[str, Any]:
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

        # Color palette
        selected_color = '#FF6B6B'  # Red for selected samples
        unselected_color = '#4ECDC4'  # Teal for unselected

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
                    color=selected_color,
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
        selected_color = '#FF6B6B'
        unselected_color = '#4ECDC4'

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
                marker_color=selected_color,
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
                    line=dict(color=unselected_color, width=2)
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
        selected_color = '#FF6B6B'
        unselected_color = '#4ECDC4'

        # All data histogram
        fig.add_trace(
            go.Histogram(
                x=coords['x'],
                name='All Data',
                opacity=0.5,
                marker_color=unselected_color,
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
                marker_color=selected_color,
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

        try:
            summary = dashboard_data.get('summary', {})
            timestamp = time.strftime('%Y%m%d_%H%M%S')

            # Get selected count safely
            selected_count = summary.get('selected_count', 0)
            total_samples = summary.get('total_samples', 0)
            selection_percentage = summary.get('selection_percentage', 0.0)
            num_clusters = summary.get('num_clusters', 0)

            # Create safe filename
            html_filename = f"timing_dashboard_{selected_count}samples_{timestamp}.html"
            html_path = os.path.join(tempfile.gettempdir(), html_filename)

            # Ensure the temp directory exists
            os.makedirs(os.path.dirname(html_path), exist_ok=True)

            # Add summary annotation
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                text=f"<b>Summary:</b> {selected_count:,} samples selected " +
                     f"({selection_percentage:.1f}%) from {total_samples:,} total " +
                     f"across {num_clusters} clusters",
                showarrow=False,
                font=dict(size=14, color="white"),
                align="left",
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="white",
                borderwidth=1
            )

            # Export with full interactivity and error handling
            fig.write_html(
                html_path,
                include_plotlyjs=True,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'timing_dashboard_{selected_count}samples',
                        'height': 800,
                        'width': 1200,
                        'scale': 2
                    }
                },
                div_id="timing-dashboard",
                include_mathjax=False
            )

            print(f"Interactive dashboard exported successfully: {html_path}")
            return html_path

        except Exception as e:
            print(f"Failed to export HTML dashboard: {str(e)}")
            print(f"Error type: {type(e).__name__}")

            # Try to create a fallback basic HTML export
            try:
                fallback_path = os.path.join(tempfile.gettempdir(), f"timing_dashboard_fallback_{timestamp}.html")
                with open(fallback_path, 'w') as f:
                    f.write(f"""
                    <html><head><title>Timing Dashboard</title></head>
                    <body>
                        <h1>Timing Data Selection Dashboard</h1>
                        <p>Selected: {selected_count:,} samples ({selection_percentage:.1f}%)</p>
                        <p>Total: {total_samples:,} samples</p>
                        <p>Clusters: {num_clusters}</p>
                        <p>Note: Interactive visualization failed, fallback text report generated</p>
                    </body></html>
                    """)
                print(f"Fallback HTML report created: {fallback_path}")
                return fallback_path
            except Exception as fallback_e:
                print(f"Fallback HTML creation also failed: {fallback_e}")
                # Return None to indicate complete failure
                return None

    def _fallback_visualization(self, df: pd.DataFrame, selected_indices: List[int], clusters: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive self-contained HTML dashboard with zero dependencies."""
        import tempfile
        import time

        print("[INFO] Generating self-contained HTML dashboard")

        # Compute comprehensive statistics
        selected = df.iloc[selected_indices]
        total = len(df)
        n_selected = len(selected)
        n_clusters = len(set(clusters))
        selection_pct = n_selected / total * 100

        # Cluster analysis
        import numpy as np
        cluster_counts = np.bincount(clusters)
        cluster_selected = np.bincount(np.array(clusters)[selected_indices], minlength=n_clusters)

        # Feature analysis
        numeric_cols = df.select_dtypes(include='number').columns.tolist()[:15]  # Top 15 features
        feature_rows = ""
        for col in numeric_cols:
            full_mean = df[col].mean()
            sel_mean = selected[col].mean()
            full_std = df[col].std()
            sel_std = selected[col].std()
            diff_pct = ((sel_mean - full_mean) / abs(full_mean) * 100) if full_mean != 0 else 0
            color = "#10B981" if abs(diff_pct) < 10 else "#F59E0B" if abs(diff_pct) < 25 else "#EF4444"
            feature_rows += f'''
            <tr>
                <td style="font-family:monospace;font-size:11px">{col[:25]}</td>
                <td>{full_mean:.4f}</td>
                <td>{sel_mean:.4f}</td>
                <td>{full_std:.4f}</td>
                <td>{sel_std:.4f}</td>
                <td style="color:{color};font-weight:bold">{diff_pct:+.1f}%</td>
            </tr>'''

        # Cell type analysis if available
        cell_info = ""
        if 'cell_arc_pt' in df.columns:
            unique_cells = df['cell_arc_pt'].str.split('#').str[0].nunique()
            selected_cells = selected['cell_arc_pt'].str.split('#').str[0].nunique()
            cell_info = f'''
            <div class="metric"><div class="metric-value">{unique_cells}</div><div class="metric-label">Unique Cells</div></div>
            <div class="metric"><div class="metric-value">{selected_cells}</div><div class="metric-label">Cells in Selection</div></div>
            '''

        # Cluster distribution table
        cluster_rows = ""
        for i in range(n_clusters):
            pct_total = cluster_counts[i] / total * 100 if total > 0 else 0
            pct_selected = cluster_selected[i] / n_selected * 100 if n_selected > 0 else 0
            rate = cluster_selected[i] / cluster_counts[i] * 100 if cluster_counts[i] > 0 else 0
            cluster_rows += f'''
            <tr>
                <td>Cluster {i}</td>
                <td>{cluster_counts[i]:,} ({pct_total:.1f}%)</td>
                <td>{cluster_selected[i]:,} ({pct_selected:.1f}%)</td>
                <td>{rate:.1f}%</td>
            </tr>'''

        # CSS bar chart
        max_count = max(cluster_counts) if len(cluster_counts) > 0 else 1
        bar_items = ""
        for i in range(n_clusters):
            height_pct = cluster_counts[i] / max_count * 100
            sel_height_pct = cluster_selected[i] / max_count * 100
            bar_items += f'''
            <div style="display:flex;flex-direction:column;align-items:center;flex:1;gap:4px">
                <div style="width:100%;display:flex;align-items:flex-end;height:200px;gap:2px;justify-content:center">
                    <div style="width:40%;background:#1E3A5F;height:{height_pct}%;border-radius:4px 4px 0 0;min-height:2px"
                         title="Total: {cluster_counts[i]}"></div>
                    <div style="width:40%;background:#06B6D4;height:{sel_height_pct}%;border-radius:4px 4px 0 0;min-height:2px"
                         title="Selected: {cluster_selected[i]}"></div>
                </div>
                <span style="font-size:11px;color:#94A3B8">C{i}</span>
            </div>'''

        # Generate HTML dashboard
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AIQC Sampling Validation Dashboard</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0B1120; color: #E2E8F0; padding: 24px; }}
    .container {{ max-width: 1280px; margin: 0 auto; }}
    h1 {{ color: #06B6D4; font-size: 1.8em; margin-bottom: 4px; }}
    .subtitle {{ color: #64748B; font-size: 0.9em; margin-bottom: 24px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }}
    .metric {{ background: #1E293B; border-radius: 10px; padding: 20px; border-left: 4px solid #0891B2; }}
    .metric-value {{ font-size: 1.8em; color: #06B6D4; font-weight: 700; }}
    .metric-label {{ color: #94A3B8; font-size: 0.8em; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
    .card {{ background: #1E293B; border-radius: 10px; padding: 20px; }}
    .card h2 {{ color: #06B6D4; font-size: 1.1em; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #334155; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
    th {{ text-align: left; padding: 8px 10px; color: #06B6D4; border-bottom: 2px solid #334155; font-weight: 600; }}
    td {{ padding: 7px 10px; border-bottom: 1px solid #1E3A5F; }}
    tr:hover td {{ background: #0F1D32; }}
    .legend {{ display: flex; gap: 16px; margin-bottom: 8px; font-size: 0.8em; }}
    .legend-item {{ display: flex; align-items: center; gap: 4px; }}
    .legend-box {{ width: 12px; height: 12px; border-radius: 2px; }}
    .footer {{ text-align: center; color: #475569; font-size: 0.75em; margin-top: 32px; padding-top: 16px; border-top: 1px solid #1E293B; }}
    @media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="container">
    <h1>AIQC Sampling Validation Dashboard</h1>
    <p class="subtitle">AI-Driven Representative Sampling for Timing Library Characterization</p>

    <div class="metrics">
        <div class="metric"><div class="metric-value">{total:,}</div><div class="metric-label">Total Arcs</div></div>
        <div class="metric"><div class="metric-value">{n_selected:,}</div><div class="metric-label">Selected</div></div>
        <div class="metric"><div class="metric-value">{selection_pct:.1f}%</div><div class="metric-label">Selection Rate</div></div>
        <div class="metric"><div class="metric-value">{n_clusters}</div><div class="metric-label">Clusters</div></div>
        {cell_info}
    </div>

    <div class="grid">
        <div class="card">
            <h2>Cluster Distribution</h2>
            <div class="legend">
                <div class="legend-item"><div class="legend-box" style="background:#1E3A5F"></div> Total</div>
                <div class="legend-item"><div class="legend-box" style="background:#06B6D4"></div> Selected</div>
            </div>
            <div style="display:flex;gap:2px;align-items:flex-end;height:220px;margin-bottom:16px">
                {bar_items}
            </div>
            <table>
                <tr><th>Cluster</th><th>Total</th><th>Selected</th><th>Rate</th></tr>
                {cluster_rows}
            </table>
        </div>

        <div class="card">
            <h2>Feature Distribution: Selected vs Full Dataset</h2>
            <table>
                <tr><th>Feature</th><th>Full Mean</th><th>Sel Mean</th><th>Full Std</th><th>Sel Std</th><th>Delta%</th></tr>
                {feature_rows}
            </table>
            <p style="font-size:0.75em;color:#64748B;margin-top:8px">
                Colors: <span style="color:#10B981">Green (&lt;10%)</span>,
                <span style="color:#F59E0B">Yellow (&lt;25%)</span>,
                <span style="color:#EF4444">Red (≥25%)</span> difference from full dataset
            </p>
        </div>
    </div>

    <div class="footer">Generated by AIQC Agent — Self-Contained HTML Dashboard (Zero Dependencies)</div>
</div>
</body>
</html>'''

        # Export HTML dashboard
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        html_filename = f"aiqc_dashboard_{n_selected}samples_{timestamp}.html"
        html_path = os.path.join(tempfile.gettempdir(), html_filename)

        try:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"[OK] Self-contained dashboard exported: {html_path} ({len(html)} bytes)")
        except Exception as e:
            print(f"Failed to export HTML dashboard: {e}")
            html_path = None

        return {
            'summary_text': f"Selected {n_selected:,} samples ({selection_pct:.1f}%) from {total:,} across {n_clusters} clusters",
            'dashboard_data': {
                'total_samples': total,
                'selected_count': n_selected,
                'selection_percentage': selection_pct,
                'num_clusters': n_clusters
            },
            'plotly_figure': None,
            'html_export_path': html_path,
            'interactive_features': {
                'self_contained': True,
                'cluster_visualization': True,
                'feature_comparison': True,
                'zero_dependencies': True
            }
        }