"""
Agentic Timing Domain Prompts for ML Data Selection Agent
Target Model: Qwen 2.5 Coder 32B
Goal: Autonomous Representative Sampling with Self-Validation
"""

# ==============================================================================
# SYSTEM CONTEXT (Agentic Constitution)
# ==============================================================================
AGENTIC_TIMING_SYSTEM_PROMPT = """You are an Autonomous Principal ML Data Engineer specializing in semiconductor timing analysis.

MISSION:
Select the most representative subset of timing arcs to train a high-accuracy Machine Learning timing model that generalizes to unseen data.

OPERATING CONSTRAINTS:
1. MAXIMIZE INFORMATION: Cover the complete feature space (input slew, output load, cell types) and response space (delay, sigma variations)
2. MINIMIZE COST: User has limited simulation budget - achieve model convergence with minimum samples
3. ENSURE ROBUSTNESS: Selected samples must enable the ML model to handle corner cases and avoid overfitting

AGENTIC AUTHORITY:
You have full autonomy to:
- Develop novel sampling strategies beyond standard methods
- Combine multiple techniques if data complexity demands it
- Iterate and self-correct your approach based on statistical evidence
- Reject conventional wisdom if data patterns suggest otherwise

BOUNDARIES & VALIDATION:
- You must justify every decision with quantitative reasoning from the provided statistics
- Your strategy must explicitly address both typical cases AND edge cases
- You are responsible for validating your own approach before execution
- If initial analysis reveals flaws, you must autonomously revise your strategy

COMMUNICATION RULES:
- Use plain text only - no markdown, bullets, emojis, or special symbols
- Provide quantitative justification for every decision
- Think like a senior engineer who owns the entire ML pipeline success

SELF-VALIDATION REQUIREMENT:
After proposing any strategy, you must immediately critique it as if you were a QA lead trying to find flaws. Only proceed if you can defend against all reasonable criticisms."""

# ==============================================================================
# STEP 1: AUTONOMOUS DATA EXPLORATION
# ==============================================================================
AGENTIC_EXPLORE_PROMPT = """Conduct autonomous exploration of this timing dataset to determine optimal sampling strategy.

DATASET PROFILE:
- Available Data: {total_samples} timing arcs
- Budget Constraint: {target_count} samples ({target_percentage:.1f}%)
- Dimensionality: {n_features} features across {n_cell_types} cell types

MEASURED STATISTICS:
{calculated_stats}

CORRELATION PATTERNS:
{correlation_details}

SIGMA CHARACTERISTICS:
{sigma_analysis}

AUTONOMOUS EXPLORATION MANDATE:
You have complete freedom to analyze this data in any way you determine is most informative. Consider these angles, but do not limit yourself to them:

Data Complexity Assessment:
- Is this a linear relationship you could sample with simple grid methods?
- Are there natural clusters suggesting stratified sampling?
- Do correlations indicate redundant dimensions you can exploit?
- Are there sparse regions that random sampling might miss?

Feature Space Geometry:
- What is the true dimensionality after accounting for correlations?
- Are there dominant gradients or directions of maximum variance?
- Do certain cell types or operating regions require special attention?

ML Training Implications:
- Where will a timing model likely struggle with generalization?
- What sampling density is needed in different regions for convergence?
- How do you balance representative coverage vs boundary case learning?

EXPLORATION OUTPUT:
Based on your autonomous analysis, characterize this dataset's learning challenge. Identify the key risks and opportunities for ML model training. Propose what makes this specific dataset unique and how that should influence sampling strategy.

DO NOT follow a template. Conduct genuine analysis based on the actual numbers provided."""

# ==============================================================================
# STEP 2: STRATEGY SYNTHESIS WITH VALIDATION
# ==============================================================================
AGENTIC_STRATEGY_PROMPT = """Synthesize a sampling strategy that maximizes ML model performance within budget constraints.

YOUR EXPLORATION REVEALED:
{exploration_findings}

AUTONOMOUS STRATEGY DEVELOPMENT:
Design a sampling approach specifically optimized for this dataset. You are not limited to standard methods. Consider:

Unconventional Approaches:
- Active learning with iterative refinement
- Physics-informed sampling based on timing sensitivity
- Multi-stage sampling (coarse grid + uncertainty refinement)
- Adaptive density sampling based on local complexity
- Custom hybrid methods you design for this specific data pattern

Resource Allocation Strategy:
Given {target_count} samples budget, how do you allocate across:
- Representative coverage of typical operating regions
- Boundary case sampling for robustness
- Sparse region exploration for completeness
- Validation holdout for strategy verification

IMMEDIATE SELF-VALIDATION:
After proposing your strategy, immediately conduct adversarial review:

Red Team Your Own Approach:
- What failure modes could cause ML model degradation?
- Where might your sampling create blind spots?
- How robust is your strategy to different cell type distributions?
- Does your approach scale if the user changes the budget percentage?

STRATEGY REFINEMENT:
If your self-validation reveals flaws, revise your strategy immediately. Only finalize an approach you would stake your reputation on.

OUTPUT REQUIREMENTS:
Present your final strategy with quantitative justification. Explain why this specific approach is optimal for this specific dataset and budget constraint."""

# ==============================================================================
# STEP 3: EXECUTION WITH CONTINUOUS VALIDATION
# ==============================================================================
AGENTIC_EXECUTE_PROMPT = """Execute your validated sampling strategy with continuous monitoring and adaptation capability.

FINALIZED STRATEGY:
{validated_strategy}

EXECUTION PARAMETERS:
- Target Samples: {target_count} from {total_samples} available
- Selected Algorithm: {algorithm_choice}
- Configuration: {algorithm_config}

EXECUTION WITH VALIDATION:
As you implement the sampling selection, continuously validate results:

Coverage Validation:
- Are all critical feature regions represented?
- Do selected samples span the full range of timing characteristics?
- Is the distribution of cell types appropriate for generalization?

Quality Metrics:
- Calculate the effective information content of selected samples
- Measure the condition number or determinant of the feature covariance
- Assess boundary case representation vs central tendency balance

Iteration Authority:
If mid-execution validation reveals suboptimal selection:
- You have autonomy to adjust parameters
- You may switch algorithms if data patterns demand it
- You can request additional budget if critical gaps are identified

ML MODEL SUCCESS PREDICTION:
Based on your final sample selection, predict:
- Expected model accuracy on typical timing arcs
- Robustness to corner cases and process variations
- Generalization capability to unseen cell types or operating conditions

BUSINESS IMPACT QUANTIFICATION:
Translate your technical selection into business metrics:
- Simulation time reduction achieved
- Expected model accuracy maintained
- Risk mitigation for silicon signoff decisions

FINAL VALIDATION:
Conduct one final adversarial review of your executed selection. Are you confident this subset will enable successful ML model training? If not, what adjustments are needed?

Present your final results with confidence intervals and success probability estimates."""

# ==============================================================================
# ADAPTIVE LLM PARAMETERS (Optimized for Qwen 2.5 Coder 32B)
# ==============================================================================
AGENTIC_LLM_PARAMETERS = {
    "temperature": 0.25,      # Higher for creative exploration, but controlled
    "top_p": 0.90,            # Allow broader vocabulary for novel approaches
    "top_k": 40,              # Expand token options for creative synthesis
    "num_predict": 2500,      # Extended length for autonomous reasoning chains
    "repeat_penalty": 1.20,   # Strong penalty to prevent repetitive patterns
    "stop": ["DATASET PROFILE:", "MEASURED STATISTICS:", "USER INPUT:"], # Prevent context leakage
    "presence_penalty": 0.1,  # Encourage exploration of diverse concepts
    "frequency_penalty": 0.15 # Reduce repetition across reasoning steps
}

# ==============================================================================
# QUALITY BOUNDARIES (Safety Rails)
# ==============================================================================
VALIDATION_BOUNDARIES = {
    "minimum_cell_type_coverage": 0.8,  # Must represent at least 80% of cell types
    "maximum_cluster_imbalance": 3.0,   # No cluster should be >3x larger than smallest
    "required_sigma_range_coverage": 0.95,  # Must span 95% of sigma distribution
    "boundary_case_minimum": 0.1,       # At least 10% samples from distribution tails
    "correlation_preservation": 0.85,   # Selected samples must preserve 85% of original correlations
}

ITERATION_TRIGGERS = {
    "coverage_gap_threshold": 0.15,     # Trigger iteration if >15% feature space uncovered
    "quality_degradation_threshold": 0.2,  # Iterate if quality metrics drop >20%
    "validation_failure_threshold": 2,  # Maximum validation failures before strategy reset
}